// SpeechAnalyzer / SpeechTranscriber transcription backend (macOS 26+).
//
// ⚠️ COMPILED ONLY AGAINST THE macOS 26 SDK. CMake compiles this file, and
// defines KV2_WITH_SPEECH_ANALYZER, only when the selected SDK is new enough.
// On an older SDK the whole backend is absent and transcriber_sf.mm's
// SFSpeechRecognizer path is the only one built.
//
// Why Swift at all: SpeechAnalyzer has no Objective-C interface. It is an actor
// with an AsyncSequence result stream, so it is reachable only from Swift.
//
// Why a C ABI rather than Swift/C++ interop: `@_cdecl` functions are plain C
// symbols, which means the C++ side needs nothing but the declarations in
// transcriber_analyzer_shim.cpp -- no generated bridging header, no module map,
// and no ordering constraint between the Swift and C++ halves of the build.
// The cost is marshalling strings through caller-provided buffers, which is a
// few lines and entirely mechanical.
//
// Lifetime: the handle is an Unmanaged reference the C++ side owns. It is
// retained at create and released at destroy, exactly once each.
//
// --- Bilingual recognition ---------------------------------------------------
//
// start() accepts several locales. Each gets its own SpeechTranscriber and
// SpeechAnalyzer over the *same* audio, and Merger picks per utterance.
//
// This exists because a recogniser handed the wrong language does not fail --
// it returns fluent, confident-looking nonsense. Measured on French speech, the
// en-US model produced "je m'appelle la Millie, le jardinis plan de fleur" at
// mean confidence 0.29 while fr-CA produced the correct text at 0.95, so
// confidence separates them cleanly. It is not symmetric: on English speech the
// gap was 0.94 (en) against 0.79 (fr), the French model coping with English far
// better than the reverse. That asymmetry is why the winner is chosen by
// *comparing* the streams rather than by any fixed threshold.

import AVFAudio
import Foundation
import Speech

// One locale's worth of recognition: its own transcriber, analyzer, converter
// and results task, all fed from the same samples as every other stream.
@available(macOS 26.0, *)
private final class Stream {
  let localeID: String
  let transcriber: SpeechTranscriber
  let analyzer: SpeechAnalyzer
  let format: AVAudioFormat
  let converter: AVAudioConverter
  var builder: AsyncStream<AnalyzerInput>.Continuation?
  var task: Task<Void, Never>?

  init(localeID: String, transcriber: SpeechTranscriber,
       analyzer: SpeechAnalyzer, format: AVAudioFormat,
       converter: AVAudioConverter) {
    self.localeID = localeID
    self.transcriber = transcriber
    self.analyzer = analyzer
    self.format = format
    self.converter = converter
  }
}

// A finalised utterance from one stream, waiting to be raced against whatever
// the other streams made of the same stretch of audio.
//
// `chars` is carried because streams do not agree on utterance boundaries (see
// mergeLocked): several candidates from one stream may have to be combined to
// line up with one from another, and combining mean confidences requires the
// weights that produced them.
private struct Candidate {
  let stream: Int
  let text: String
  let confidence: Double
  let chars: Int
  let start: Double  // seconds; CMTimeRange.start of the result
  let end: Double
  let arrived: Date
}

@available(macOS 26.0, *)
private final class AnalyzerSession {
  private var streams: [Stream] = []
  private var sourceFormat: AVAudioFormat?

  // Results and errors cross from the analyzers' tasks back to the polling C++
  // thread through here. A plain lock is right: contention is a few threads at
  // a few hertz, and none of them is realtime.
  private let lock = NSLock()
  private var pending: [(String, Bool, String)] = []  // text, isFinal, locale
  private var errorText: String = ""

  // Finals waiting to be raced, per stream, oldest first.
  private var candidates: [[Candidate]] = []
  // The stream currently allowed to drive the volatile (in-flight) text. Only
  // one may, or the live line would flip between two languages' guesses several
  // times a second. It follows whichever stream last won an utterance.
  private var lead: Int = 0

  // Audio time (seconds) already emitted. Finals arriving entirely behind it
  // are dropped: a straggler from a stream that lost a forced race would
  // otherwise re-emit words the user has already read.
  private var emittedThrough: Double = -1

  // How long to wait for the slower streams to finalise up to the end of a span
  // before racing with whoever has answered. Not a tie-break window: streams
  // genuinely endpoint at different times, and one that has run two sentences
  // together answers a good deal later than one that split them. The live
  // volatile text keeps the UI moving meanwhile, so this only delays *settled*
  // text.
  private let graceS: Double = 1.5

  // --- result plumbing -------------------------------------------------------

  private func push(_ text: String, _ isFinal: Bool, _ locale: String) {
    lock.lock()
    defer { lock.unlock() }
    pending.append((text, isFinal, locale))
    // Bound it, in case the UI stops polling entirely.
    if pending.count > 256 { pending.removeFirst(pending.count - 256) }
  }

  private func setError(_ text: String) {
    lock.lock()
    errorText = text
    lock.unlock()
  }

  func takeResult() -> (String, Bool, String)? {
    lock.lock()
    defer { lock.unlock() }
    return pending.isEmpty ? nil : pending.removeFirst()
  }

  func takeError() -> String {
    lock.lock()
    defer { lock.unlock() }
    return errorText
  }

  // --- confidence ------------------------------------------------------------

  // Mean confidence over the utterance, weighted by how much text each run
  // covers. Unweighted, a one-character run counts as much as a whole clause,
  // which is exactly backwards for judging a whole utterance.
  private func confidence(of s: AttributedString) -> Double {
    var weighted = 0.0
    var chars = 0
    for run in s.runs {
      guard let c = run.transcriptionConfidence else { continue }
      let n = s.characters[run.range].count
      weighted += Double(c) * Double(n)
      chars += n
    }
    // No confidence attributes at all (a locale whose model does not report
    // them): treat as neutral so a lone stream still emits, and so it never
    // silently beats a stream that does report.
    return chars == 0 ? 0.5 : weighted / Double(chars)
  }

  // --- merge -----------------------------------------------------------------

  // Races the buffered finals and emits the winners. Called with the lock held.
  //
  // `force` skips the grace period, for stop(): whatever has arrived by then is
  // all there will ever be.
  //
  // Each utterance is raced on its own: the oldest unresolved final anchors a
  // group, every stream's overlapping final joins it, and the most confident
  // wins. Emission is by *audio time* -- once a stretch has been emitted no
  // stream may emit over it again, which is what keeps a straggler from
  // re-showing words the user has already read.
  //
  // The streams do not always segment alike. On audio that switches language
  // mid-session, fr-CA ran three of en-US's utterances together and answered
  // seconds late; that late result is discarded rather than merged, because
  // waiting for the slowest stream means unbounded latency on the transcript.
  // See the README: this is why language switching lags.
  private func mergeLocked(force: Bool) {
    guard !streams.isEmpty else { return }
    let eps = 0.001

    while true {
      // The oldest unresolved final across all streams anchors the group.
      var anchor: Candidate?
      for list in candidates {
        guard let c = list.first else { continue }
        if anchor == nil || c.start < anchor!.start { anchor = c }
      }
      guard let head = anchor else { return }

      var group: [Candidate] = []
      var waiting = false
      for i in 0..<streams.count {
        if let c = candidates[i].first, c.start < head.end - eps,
           c.end > head.start + eps {
          group.append(c)
        } else if candidates[i].isEmpty {
          waiting = true  // may still be coming
        }
      }
      // Give the slower streams a bounded chance to answer the same utterance.
      if waiting && !force &&
          Date().timeIntervalSince(head.arrived) < graceS {
        return
      }
      guard let winner = group.max(by: { $0.confidence < $1.confidence }) else {
        return
      }

      pending.append((winner.text, true, streams[winner.stream].localeID))
      if pending.count > 256 { pending.removeFirst(pending.count - 256) }
      lead = winner.stream

      // Everything through the winner's end is now spoken for. Anything a
      // stream still has that *starts* behind that edge would re-cover emitted
      // audio, so it is dropped whole -- the emitted stream will produce its
      // own finals for the remainder of that stretch.
      emittedThrough = max(emittedThrough, winner.end)
      for i in 0..<streams.count {
        candidates[i].removeAll { $0.start < emittedThrough - eps }
      }
    }
  }

  // Clears everything carried by a finished session. A session may be started,
  // stopped and started again -- the demo's `transcribe` toggle does exactly
  // that -- and every analyzer timestamps its results from zero, so leftover
  // state is not merely stale but actively wrong: an emittedThrough left at the
  // old session's end silently swallows *every* result of the new one, and
  // leftover pending text reappears beneath the fresh transcript.
  //
  // Not async, for the same reason as mergeNow().
  private func resetForNewSession() {
    lock.lock()
    pending.removeAll()
    candidates.removeAll()
    emittedThrough = -1
    lead = 0
    errorText = ""
    lock.unlock()
  }

  // Takes the lock and merges. Deliberately *not* async: NSLock may not be
  // taken from an async context (an error under Swift 6), and there is nothing
  // to await here anyway.
  private func mergeNow(force: Bool) {
    lock.lock()
    mergeLocked(force: force)
    lock.unlock()
  }

  // Wakes the merge after the grace period, so an utterance that only one
  // stream ever finalised still gets emitted instead of sitting in the buffer.
  private func scheduleMerge() {
    Task { [weak self] in
      guard let grace = self?.graceS else { return }
      try? await Task.sleep(nanoseconds: UInt64((grace + 0.05) * 1_000_000_000))
      self?.mergeNow(force: false)
    }
  }

  private func onResult(stream i: Int, _ r: SpeechTranscriber.Result) {
    let text = String(r.text.characters)
    if r.isFinal {
      let start = r.range.start.seconds
      let end = r.range.end.seconds
      lock.lock()
      let c = Candidate(
          stream: i, text: text, confidence: confidence(of: r.text),
          chars: text.count,
          start: start.isFinite ? start : 0,
          end: end.isFinite ? end : 0,
          arrived: Date())
      // Starts behind the emitted edge: this stretch was already raced and
      // shown, so this result can only duplicate it.
      if c.start >= emittedThrough - 0.001 {
        candidates[i].append(c)
        mergeLocked(force: false)
      }
      lock.unlock()
      scheduleMerge()
    } else {
      // Volatile text from the lead stream only; see `lead`.
      lock.lock()
      let isLead = (i == lead)
      lock.unlock()
      if isLead { push(text, false, streams[i].localeID) }
    }
  }

  // --- lifecycle -------------------------------------------------------------

  func start(sampleRate: Double, locales localeIDs: [String]) async -> String {
    guard !localeIDs.isEmpty else { return "no locale requested" }

    resetForNewSession()

    // The beam is mono float32 at the mic array's rate; each analyzer wants its
    // own format. Convert rather than assume they match -- they generally do
    // not.
    guard let src = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                  sampleRate: sampleRate,
                                  channels: 1,
                                  interleaved: false) else {
      return "could not build a 1ch float32 source format at \(sampleRate) Hz"
    }
    sourceFormat = src

    for localeID in localeIDs {
      // supportedLocale(equivalentTo:) is the framework's own matcher. Doing it
      // by hand is a trap: the locale lists come back with ICU-style
      // identifiers ("en_US") while a BCP-47 one ("en-US") keeps its hyphen,
      // and Locale equality is identifier equality, so a plain `contains`
      // reports every locale as unsupported -- including installed ones.
      guard let locale = await SpeechTranscriber.supportedLocale(
          equivalentTo: Locale(identifier: localeID)) else {
        return "locale '\(localeID)' is not supported by SpeechTranscriber"
      }

      // .volatileResults is what makes this feel live: the transcriber emits
      // revisable hypotheses as it goes rather than only settled text.
      // .transcriptionConfidence is what lets several locales be raced.
      let t = SpeechTranscriber(locale: locale,
                                transcriptionOptions: [],
                                reportingOptions: [.volatileResults],
                                attributeOptions: [.transcriptionConfidence])

      // The locale's model is a downloadable asset, not something bundled with
      // the OS. Missing it is the single most likely first-run failure, so it
      // is handled rather than reported -- but it can take the better part of a
      // minute, which is why kv2_sa_start's timeout is what it is.
      let installed = (await SpeechTranscriber.installedLocales)
          .contains { $0.identifier(.bcp47) == locale.identifier(.bcp47) }
      if !installed {
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
        return "no audio format compatible with the '\(localeID)' transcriber"
      }
      guard let converter = AVAudioConverter(from: src, to: format) else {
        return "no converter from \(sampleRate) Hz mono to the '\(localeID)' " +
               "analyzer format"
      }

      streams.append(Stream(localeID: localeID, transcriber: t,
                            analyzer: SpeechAnalyzer(modules: [t]),
                            format: format, converter: converter))
    }

    candidates = Array(repeating: [], count: streams.count)
    lead = 0

    for (i, s) in streams.enumerated() {
      let (inputSequence, builder) = AsyncStream<AnalyzerInput>.makeStream()
      s.builder = builder
      s.task = Task { [weak self] in
        do {
          for try await result in s.transcriber.results {
            self?.onResult(stream: i, result)
          }
        } catch is CancellationError {
          // stop() cancels these tasks deliberately; that is not a failure, and
          // reporting it would light up the UI's error line on every stop.
        } catch {
          self?.setError("transcription stream failed: \(error)")
        }
      }
      do {
        try await s.analyzer.start(inputSequence: inputSequence)
      } catch {
        return "could not start the analyzer for '\(s.localeID)': \(error)"
      }
    }
    return ""  // empty == success
  }

  func feed(_ samples: UnsafePointer<Float>, _ count: Int) {
    guard let src = sourceFormat, count > 0, !streams.isEmpty else { return }

    guard let inBuf = AVAudioPCMBuffer(pcmFormat: src,
                                       frameCapacity: AVAudioFrameCount(count)),
          let channel = inBuf.floatChannelData else { return }
    inBuf.frameLength = AVAudioFrameCount(count)
    channel[0].update(from: samples, count: count)

    // One conversion per stream: different locales can negotiate different
    // analyzer formats, so the converted buffer is not shareable in general.
    for s in streams {
      guard let builder = s.builder else { continue }

      // Ceil the output capacity, and add a frame of slack: a resampling
      // converter can emit one more frame than the exact ratio suggests, and a
      // short buffer makes convertToBuffer fail outright.
      let ratio = s.format.sampleRate / src.sampleRate
      let capacity = AVAudioFrameCount((Double(count) * ratio).rounded(.up)) + 1
      guard let outBuf = AVAudioPCMBuffer(pcmFormat: s.format,
                                          frameCapacity: capacity) else {
        continue
      }

      var consumed = false
      var convError: NSError?
      s.converter.convert(to: outBuf, error: &convError) { _, status in
        if consumed {
          status.pointee = .noDataNow
          return nil
        }
        consumed = true
        status.pointee = .haveData
        return inBuf
      }
      if let convError {
        setError("audio conversion failed for '\(s.localeID)': " +
                 convError.localizedDescription)
        continue
      }
      guard outBuf.frameLength > 0 else { continue }
      builder.yield(AnalyzerInput(buffer: outBuf))
    }
  }

  func stop() async {
    // Finalise rather than cancel. The transcriber emits volatile results as it
    // goes but only settles an utterance at its endpoint, so at the moment the
    // user switches the mic off there is always one utterance in flight -- the
    // one they just finished speaking. cancelAndFinishNow() discards it; this
    // flushes it out as a final result, which poll() still drains after stop()
    // because the queue outlives the session.
    for s in streams {
      s.builder?.finish()
      s.builder = nil
    }
    for s in streams {
      do {
        try await s.analyzer.finalizeAndFinishThroughEndOfInput()
      } catch {
        // Finalisation is best-effort: if it fails, stop regardless rather than
        // leave a live analyzer attached to a dead session.
        await s.analyzer.cancelAndFinishNow()
      }
    }
    // Race whatever the flush produced before the streams go away; nothing more
    // is coming, so there is no reason to keep waiting on stragglers.
    mergeNow(force: true)

    for s in streams { s.task?.cancel() }
    streams.removeAll()
    candidates.removeAll()
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

// `locales` is one BCP-47 identifier or a comma-separated list of them; see
// Transcriber::start(). Synchronous by design: the C++ caller is on the UI
// thread during setup, and start() is a one-off that must either succeed or
// produce a message before the UI decides what to show. Returns true on
// success; on failure `err` holds why.
@_cdecl("kv2_sa_start")
public func kv2_sa_start(_ handle: UnsafeMutableRawPointer?,
                         _ sampleRate: Double,
                         _ locales: UnsafePointer<CChar>?,
                         _ err: UnsafeMutablePointer<CChar>?,
                         _ errLen: Int32) -> Bool {
  guard #available(macOS 26.0, *), let handle else {
    writeCString("SpeechAnalyzer requires macOS 26", err, errLen)
    return false
  }
  let session = Unmanaged<AnalyzerSession>.fromOpaque(handle)
      .takeUnretainedValue()
  let list = (locales.map { String(cString: $0) } ?? "en-US")
      .split(separator: ",")
      .map { $0.trimmingCharacters(in: .whitespaces) }
      .filter { !$0.isEmpty }

  let sem = DispatchSemaphore(value: 0)
  var message = ""
  Task {
    message = await session.start(sampleRate: sampleRate,
                                  locales: list.isEmpty ? ["en-US"] : list)
    sem.signal()
  }
  // Generous, and per locale: the first run for a locale downloads a model,
  // which took ~40 s for fr-CA on a warm network.
  let budget = 180.0 * Double(max(1, list.count))
  if sem.wait(timeout: .now() + budget) == .timedOut {
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
  _ = sem.wait(timeout: .now() + 10)
}

// Pops one result, with the locale that produced it. Returns false when the
// queue is empty.
@_cdecl("kv2_sa_poll")
public func kv2_sa_poll(_ handle: UnsafeMutableRawPointer?,
                        _ text: UnsafeMutablePointer<CChar>?,
                        _ textLen: Int32,
                        _ isFinal: UnsafeMutablePointer<Bool>?,
                        _ locale: UnsafeMutablePointer<CChar>?,
                        _ localeLen: Int32) -> Bool {
  guard #available(macOS 26.0, *), let handle else { return false }
  let session = Unmanaged<AnalyzerSession>.fromOpaque(handle)
      .takeUnretainedValue()
  guard let (s, final, loc) = session.takeResult() else { return false }
  writeCString(s, text, textLen)
  writeCString(loc, locale, localeLen)
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
