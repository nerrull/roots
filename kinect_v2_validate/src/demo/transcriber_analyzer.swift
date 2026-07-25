// SpeechAnalyzer / SpeechTranscriber transcription backend (macOS 26+).
//
// ⚠️ COMPILED ONLY AGAINST THE macOS 26 SDK. CMake adds this file, and defines
// KV2_WITH_SPEECH_ANALYZER, only when the selected SDK is new enough. On an
// older SDK the whole backend is absent and transcriber_sf.mm's
// SFSpeechRecognizer path is the only one built.
//
// Why Swift at all: SpeechAnalyzer has no Objective-C interface. It is an actor
// with an AsyncSequence result stream, so it is reachable only from Swift.
//
// Why a C ABI rather than Swift/C++ interop: `@_cdecl` functions are plain C
// symbols, which means the C++ side needs nothing but the declarations in
// transcriber_analyzer_shim.mm -- no generated bridging header, no module map,
// and no ordering constraint between the Swift and C++ halves of the build.
// The cost is marshalling strings through caller-provided buffers, which is a
// few lines and entirely mechanical.
//
// Lifetime: the handle is an Unmanaged reference the C++ side owns. It is
// retained at create and released at destroy, exactly once each.

import AVFAudio
import Foundation
import Speech

@available(macOS 26.0, *)
private final class AnalyzerSession {
  private var analyzer: SpeechAnalyzer?
  private var transcriber: SpeechTranscriber?
  private var inputBuilder: AsyncStream<AnalyzerInput>.Continuation?
  private var analyzerFormat: AVAudioFormat?
  private var resultsTask: Task<Void, Never>?
  private var converter: AVAudioConverter?
  private var sourceFormat: AVAudioFormat?

  // Results and errors cross from the analyzer's task back to the polling C++
  // thread through here. A plain lock is right: contention is two threads at a
  // few hertz, and neither side is realtime.
  private let lock = NSLock()
  private var pending: [(String, Bool)] = []
  private var errorText: String = ""

  private func push(_ text: String, _ isFinal: Bool) {
    lock.lock()
    defer { lock.unlock() }
    pending.append((text, isFinal))
    // Bound it, in case the UI stops polling entirely.
    if pending.count > 256 { pending.removeFirst(pending.count - 256) }
  }

  private func setError(_ text: String) {
    lock.lock()
    errorText = text
    lock.unlock()
  }

  func takeResult() -> (String, Bool)? {
    lock.lock()
    defer { lock.unlock() }
    return pending.isEmpty ? nil : pending.removeFirst()
  }

  func takeError() -> String {
    lock.lock()
    defer { lock.unlock() }
    return errorText
  }

  func start(sampleRate: Double, locale localeID: String) async -> String {
    let locale = Locale(identifier: localeID)

    // .volatileResults is what makes this feel live: the transcriber emits
    // revisable hypotheses as it goes rather than only settled text.
    let t = SpeechTranscriber(locale: locale,
                              transcriptionOptions: [],
                              reportingOptions: [.volatileResults],
                              attributeOptions: [])
    transcriber = t

    // The locale's model is a downloadable asset, not something bundled with
    // the OS. Missing it is the single most likely first-run failure, so it is
    // handled rather than reported.
    if !(await SpeechTranscriber.installedLocales).contains(locale) {
      guard await SpeechTranscriber.supportedLocales.contains(locale) else {
        return "locale '\(localeID)' is not supported by SpeechTranscriber"
      }
      do {
        if let request =
            try await AssetInventory.assetInstallationRequest(supporting: [t]) {
          try await request.downloadAndInstall()
        }
      } catch {
        return "could not install the speech model for '\(localeID)': \(error)"
      }
    }

    guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(
        compatibleWith: [t]) else {
      return "no audio format compatible with the transcriber"
    }
    analyzerFormat = format

    // The beam is mono float32 at the mic array's rate; the analyzer wants its
    // own. Convert rather than assume they match -- they generally do not.
    guard let src = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                  sampleRate: sampleRate,
                                  channels: 1,
                                  interleaved: false) else {
      return "could not build a 1ch float32 source format at \(sampleRate) Hz"
    }
    sourceFormat = src
    converter = AVAudioConverter(from: src, to: format)
    if converter == nil {
      return "no converter from \(sampleRate) Hz mono to the analyzer format"
    }

    let (inputSequence, builder) = AsyncStream<AnalyzerInput>.makeStream()
    inputBuilder = builder

    let a = SpeechAnalyzer(modules: [t])
    analyzer = a

    resultsTask = Task { [weak self] in
      do {
        for try await result in t.results {
          self?.push(String(result.text.characters), result.isFinal)
        }
      } catch {
        self?.setError("transcription stream failed: \(error)")
      }
    }

    do {
      try await a.start(inputSequence: inputSequence)
    } catch {
      return "could not start the analyzer: \(error)"
    }
    return ""  // empty == success
  }

  func feed(_ samples: UnsafePointer<Float>, _ count: Int) {
    guard let builder = inputBuilder,
          let src = sourceFormat,
          let dst = analyzerFormat,
          let converter = converter,
          count > 0 else { return }

    guard let inBuf = AVAudioPCMBuffer(pcmFormat: src,
                                       frameCapacity: AVAudioFrameCount(count)),
          let channel = inBuf.floatChannelData else { return }
    inBuf.frameLength = AVAudioFrameCount(count)
    channel[0].update(from: samples, count: count)

    // Ceil the output capacity, and add a frame of slack: a resampling
    // converter can emit one more frame than the exact ratio suggests, and a
    // short buffer makes convertToBuffer fail outright.
    let ratio = dst.sampleRate / src.sampleRate
    let capacity = AVAudioFrameCount((Double(count) * ratio).rounded(.up)) + 1
    guard let outBuf = AVAudioPCMBuffer(pcmFormat: dst,
                                        frameCapacity: capacity) else { return }

    var consumed = false
    var convError: NSError?
    converter.convert(to: outBuf, error: &convError) { _, status in
      if consumed {
        status.pointee = .noDataNow
        return nil
      }
      consumed = true
      status.pointee = .haveData
      return inBuf
    }
    if let convError {
      setError("audio conversion failed: \(convError.localizedDescription)")
      return
    }
    guard outBuf.frameLength > 0 else { return }
    builder.yield(AnalyzerInput(buffer: outBuf))
  }

  func stop() async {
    inputBuilder?.finish()
    inputBuilder = nil
    if let a = analyzer {
      // No "through" sample to finalise against on a live stream that the user
      // simply switched off, so cancel rather than wait for a flush.
      await a.cancelAndFinishNow()
    }
    resultsTask?.cancel()
    resultsTask = nil
    analyzer = nil
    transcriber = nil
    converter = nil
  }
}

// --- C ABI -------------------------------------------------------------------

// Always NUL-terminates and never overruns; truncates if the caller's buffer is
// too small. Callers pass buffers sized for a sentence, so truncation is a
// non-event in practice.
private func writeCString(_ s: String, _ buf: UnsafeMutablePointer<CChar>?,
                          _ len: Int32) {
  guard let buf, len > 0 else { return }
  s.withCString { _ = strlcpy(buf, $0, Int(len)) }
}

@_cdecl("kv2_sa_available")
public func kv2_sa_available() -> Bool {
  if #available(macOS 26.0, *) { return true }
  return false
}

@_cdecl("kv2_sa_create")
public func kv2_sa_create() -> UnsafeMutableRawPointer? {
  guard #available(macOS 26.0, *) else { return nil }
  return Unmanaged.passRetained(AnalyzerSession()).toOpaque()
}

@_cdecl("kv2_sa_destroy")
public func kv2_sa_destroy(_ handle: UnsafeMutableRawPointer?) {
  guard #available(macOS 26.0, *), let handle else { return }
  Unmanaged<AnalyzerSession>.fromOpaque(handle).release()
}

// Synchronous by design: the C++ caller is on the UI thread during setup, and
// start() is a one-off that must either succeed or produce a message before the
// UI decides what to show. Returns true on success; on failure `err` holds why.
@_cdecl("kv2_sa_start")
public func kv2_sa_start(_ handle: UnsafeMutableRawPointer?,
                         _ sampleRate: Double,
                         _ locale: UnsafePointer<CChar>?,
                         _ err: UnsafeMutablePointer<CChar>?,
                         _ errLen: Int32) -> Bool {
  guard #available(macOS 26.0, *), let handle else {
    writeCString("SpeechAnalyzer requires macOS 26", err, errLen)
    return false
  }
  let session = Unmanaged<AnalyzerSession>.fromOpaque(handle)
      .takeUnretainedValue()
  let localeID = locale.map { String(cString: $0) } ?? "en-US"

  let sem = DispatchSemaphore(value: 0)
  var message = ""
  Task {
    message = await session.start(sampleRate: sampleRate, locale: localeID)
    sem.signal()
  }
  // Generous: the first run for a locale downloads a model.
  if sem.wait(timeout: .now() + 180) == .timedOut {
    writeCString("timed out starting SpeechAnalyzer (model download?)",
                 err, errLen)
    return false
  }
  if !message.isEmpty {
    writeCString(message, err, errLen)
    return false
  }
  return true
}

@_cdecl("kv2_sa_feed")
public func kv2_sa_feed(_ handle: UnsafeMutableRawPointer?,
                        _ samples: UnsafePointer<Float>?,
                        _ count: Int32) {
  guard #available(macOS 26.0, *), let handle, let samples, count > 0 else {
    return
  }
  Unmanaged<AnalyzerSession>.fromOpaque(handle).takeUnretainedValue()
      .feed(samples, Int(count))
}

@_cdecl("kv2_sa_stop")
public func kv2_sa_stop(_ handle: UnsafeMutableRawPointer?) {
  guard #available(macOS 26.0, *), let handle else { return }
  let session = Unmanaged<AnalyzerSession>.fromOpaque(handle)
      .takeUnretainedValue()
  let sem = DispatchSemaphore(value: 0)
  Task {
    await session.stop()
    sem.signal()
  }
  _ = sem.wait(timeout: .now() + 5)
}

// Pops one result. Returns false when the queue is empty.
@_cdecl("kv2_sa_poll")
public func kv2_sa_poll(_ handle: UnsafeMutableRawPointer?,
                        _ text: UnsafeMutablePointer<CChar>?,
                        _ textLen: Int32,
                        _ isFinal: UnsafeMutablePointer<Bool>?) -> Bool {
  guard #available(macOS 26.0, *), let handle else { return false }
  let session = Unmanaged<AnalyzerSession>.fromOpaque(handle)
      .takeUnretainedValue()
  guard let (s, final) = session.takeResult() else { return false }
  writeCString(s, text, textLen)
  isFinal?.pointee = final
  return true
}

@_cdecl("kv2_sa_error")
public func kv2_sa_error(_ handle: UnsafeMutableRawPointer?,
                         _ buf: UnsafeMutablePointer<CChar>?,
                         _ bufLen: Int32) {
  guard #available(macOS 26.0, *), let handle else {
    writeCString("", buf, bufLen)
    return
  }
  let session = Unmanaged<AnalyzerSession>.fromOpaque(handle)
      .takeUnretainedValue()
  writeCString(session.takeError(), buf, bufLen)
}
