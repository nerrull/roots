// Headless tests for the transcription and voice-cloning plumbing.
//
// Deliberately covers only the parts that do not need a microphone, an OS
// permission, an on-device speech model, or a running Python server -- i.e. the
// parts that would otherwise only ever be checked by using the app and
// squinting. That is: transcript accumulation (the volatile/final bookkeeping
// every backend feeds into), WAV round-tripping (the format both the reference
// clip and the server's output travel in), and reference-clip capture.
//
// The recognisers themselves are Apple's; their correctness is not ours to
// test. What is ours is what we do with their output.

#include "transcriber.h"
#include "voice_clone.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
  if (!ok) ++g_failures;
}

void CheckNear(double got, double want, double tol, const char* what) {
  const bool ok = std::fabs(got - want) <= tol;
  std::printf("  [%s] %s (got %.6f, want %.6f +/- %g)\n", ok ? " ok " : "FAIL",
              what, got, want, tol);
  if (!ok) ++g_failures;
}

std::string TempPath(const char* name) {
  const char* dir = std::getenv("TMPDIR");
  std::string d = dir ? dir : "/tmp/";
  if (!d.empty() && d.back() != '/') d += '/';
  return d + name;
}

// --- transcript accumulation -------------------------------------------------

void TestTranscriptLog() {
  std::printf("TranscriptLog\n");
  using transcribe::Result;
  using transcribe::TranscriptLog;

  {
    // The core behaviour: volatile results are successive guesses at the *same*
    // words, so they replace one another rather than piling up. Getting this
    // wrong yields "the the qu the quick the quick brown" on screen.
    TranscriptLog log;
    log.add(Result{"the", false});
    log.add(Result{"the quick", false});
    log.add(Result{"the quick brown", false});
    Check(log.finals().empty(), "volatile results do not finalise anything");
    Check(log.partial() == "the quick brown",
          "the newest volatile result replaces the previous one");
    Check(log.full() == "the quick brown", "full() shows the pending partial");
  }

  {
    // A final supersedes the partial it was refined from.
    TranscriptLog log;
    log.add(Result{"the quick brwn", false});
    log.add(Result{"The quick brown fox.", true});
    Check(log.finals().size() == 1, "one finalised utterance");
    Check(log.partial().empty(), "the final cleared the partial it replaced");
    Check(log.full() == "The quick brown fox.",
          "no duplicated text after finalisation");
  }

  {
    // Finals accumulate; a new partial rides on the end.
    TranscriptLog log;
    log.add(Result{"One.", true});
    log.add(Result{"Two.", true});
    log.add(Result{"thr", false});
    Check(log.finals().size() == 2, "finals accumulate");
    Check(log.full() == "One. Two. thr", "finals then partial, space-joined");
    log.clear();
    Check(log.finals().empty() && log.partial().empty() && log.full().empty(),
          "clear() empties everything");
  }

  {
    // An empty final is what a recogniser emits for a stretch of silence. It
    // must still clear the partial, but must not push a blank entry that would
    // render as a stray double space.
    TranscriptLog log;
    log.add(Result{"umm", false});
    log.add(Result{"", true});
    Check(log.finals().empty(), "an empty final adds no entry");
    Check(log.partial().empty(), "an empty final still clears the partial");
  }

  {
    // The cap keeps a long-running session bounded, dropping oldest first.
    TranscriptLog log(3);
    for (int i = 1; i <= 5; ++i) {
      log.add(Result{std::string(1, char('0' + i)), true});
    }
    Check(log.finals().size() == 3, "final count is capped");
    Check(log.finals().front() == "3" && log.finals().back() == "5",
          "the cap drops oldest first");
  }
}

// --- WAV round trip ----------------------------------------------------------

void TestWavRoundTrip() {
  std::printf("WAV round trip\n");
  const std::string path = TempPath("kv2_test_tone.wav");

  // A tone rather than noise: a sample-order or endianness bug shows up as an
  // amplitude error here, where random data would still round-trip "fine".
  const int rate = 16000;
  std::vector<float> in(rate / 2);
  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = 0.6f * std::sin(2.0 * M_PI * 440.0 * double(i) / rate);
  }

  std::string err;
  Check(voice::WriteWav16(path, in, rate, err), "WriteWav16 succeeds");

  std::vector<float> out;
  int got_rate = 0;
  Check(voice::ReadWavMono(path, out, &got_rate, err),
        "ReadWavMono succeeds on what we wrote");
  Check(got_rate == rate, "sample rate survives the round trip");
  Check(out.size() == in.size(), "sample count survives the round trip");

  double worst = 0.0;
  for (size_t i = 0; i < out.size() && i < in.size(); ++i) {
    worst = std::max(worst, double(std::fabs(out[i] - in[i])));
  }
  // 16-bit quantisation is 1/32768 ~ 3e-5; anything larger is a real bug.
  CheckNear(worst, 0.0, 1e-4, "samples survive within 16-bit quantisation");

  {
    // Clipping, not wrapping. An out-of-range sample scaled without a clamp
    // wraps int16 and turns the loudest moment into full-scale noise -- the
    // single worst-sounding possible failure.
    const std::string clip_path = TempPath("kv2_test_clip.wav");
    std::vector<float> hot = {2.0f, -2.0f, 1.0f, -1.0f, 0.0f};
    Check(voice::WriteWav16(clip_path, hot, rate, err),
          "WriteWav16 accepts out-of-range input");
    std::vector<float> back;
    Check(voice::ReadWavMono(clip_path, back, nullptr, err),
          "the clipped file reads back");
    bool in_range = true;
    for (float s : back) in_range = in_range && s >= -1.0f && s <= 1.0f;
    Check(in_range, "out-of-range samples clip rather than wrap");
    Check(back.size() >= 2 && back[0] > 0.9f && back[1] < -0.9f,
          "clipped samples keep their sign");
    std::remove(clip_path.c_str());
  }

  {
    // Not a WAV at all: must fail with a message, not read garbage.
    const std::string junk = TempPath("kv2_test_junk.wav");
    FILE* f = std::fopen(junk.c_str(), "wb");
    if (f) {
      const char* garbage = "this is definitely not a RIFF WAVE file at all!!";
      std::fwrite(garbage, 1, 48, f);
      std::fclose(f);
    }
    std::vector<float> back;
    std::string jerr;
    Check(!voice::ReadWavMono(junk, back, nullptr, jerr),
          "a non-WAV file is rejected");
    Check(!jerr.empty(), "rejection comes with an explanation");
    std::remove(junk.c_str());
  }

  std::remove(path.c_str());
}

// --- reference clip ----------------------------------------------------------

void TestReferenceRecorder() {
  std::printf("ReferenceRecorder\n");
  const int rate = 16000;
  voice::ReferenceRecorder rec;

  rec.begin(1.0, rate);
  Check(rec.recording(), "begin() starts recording");
  Check(!rec.complete(), "not complete before any audio arrives");

  // Feed in blocks, as the UI does when draining the ring.
  std::vector<float> block(512, 0.25f);
  int blocks = 0;
  bool done = false;
  while (!done && blocks < 100) {
    done = rec.feed(block.data(), int(block.size()));
    ++blocks;
  }
  Check(done, "capture completes once the target duration is reached");
  Check(rec.complete(), "complete() agrees");
  Check(!rec.recording(), "recording stops on its own");
  CheckNear(rec.capturedSeconds(), 1.0, 0.001, "captured exactly the target");
  Check(rec.samples().size() == size_t(rate),
        "no overrun past the requested length");
  CheckNear(rec.peak(), 0.25, 1e-6, "peak reflects the captured audio");

  {
    // Overrun guard: one huge block must still stop exactly on target, since
    // the ring drain hands over whatever accumulated since the last UI frame.
    voice::ReferenceRecorder r2;
    r2.begin(0.5, rate);
    std::vector<float> huge(rate * 4, 0.1f);
    Check(r2.feed(huge.data(), int(huge.size())),
          "an oversized block completes the clip");
    Check(r2.samples().size() == size_t(rate / 2),
          "an oversized block is truncated to the target");
  }

  {
    voice::ReferenceRecorder r3;
    r3.begin(0.5, rate);
    r3.feed(block.data(), int(block.size()));
    r3.cancel();
    Check(!r3.recording() && !r3.complete(),
          "cancel() discards a partial clip rather than saving it");
    std::string err;
    Check(!r3.save(TempPath("kv2_should_not_exist.wav"), err),
          "saving a cancelled clip fails");
  }

  // Saving produces something ReadWavMono accepts, which is the actual
  // contract: the server reads this file.
  const std::string ref = TempPath("kv2_test_ref.wav");
  std::string err;
  Check(rec.save(ref, err), "save() writes the clip");
  std::vector<float> back;
  int back_rate = 0;
  Check(voice::ReadWavMono(ref, back, &back_rate, err),
        "the saved reference clip is a readable WAV");
  Check(back_rate == rate && back.size() == size_t(rate),
        "the saved clip has the expected rate and length");
  std::remove(ref.c_str());
}

// --- backend reporting -------------------------------------------------------

void TestBackendReporting() {
  std::printf("backend reporting\n");
  using transcribe::Backend;

  Check(std::string(transcribe::BackendName(Backend::kSpeechAnalyzer)) ==
            "SpeechAnalyzer",
        "backend names are stable strings");

  // Not an assertion about this machine: whether SpeechAnalyzer is compiled in
  // depends on the SDK. What must hold is that the two questions -- "is it in
  // the binary?" and "can it run here?" -- never contradict each other, since
  // the UI shows both.
  const bool built = transcribe::BuiltWithSpeechAnalyzer();
  const bool avail = transcribe::BackendAvailable(Backend::kSpeechAnalyzer);
  Check(built || !avail,
        "a backend cannot be runtime-available if it was never compiled in");
  Check(transcribe::BackendAvailable(Backend::kSpeechRecognizer),
        "SFSpeechRecognizer is always available");
  Check(transcribe::BackendAvailable(Backend::kAuto),
        "kAuto is available whenever any backend is");
  std::printf("       (this build: SpeechAnalyzer compiled=%s runtime=%s)\n",
              built ? "yes" : "no", avail ? "yes" : "no");
}

}  // namespace

int main() {
  TestTranscriptLog();
  TestWavRoundTrip();
  TestReferenceRecorder();
  TestBackendReporting();

  std::printf("\n%s\n", g_failures == 0 ? "all voice tests passed"
                                        : "VOICE TESTS FAILED");
  return g_failures == 0 ? 0 : 1;
}
