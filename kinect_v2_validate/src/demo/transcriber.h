// Realtime speech-to-text over the mic array's beamformed output.
//
// Two backends sit behind one interface, because Apple's current API is not
// available on every macOS this demo has to build on:
//
//   kSpeechAnalyzer -- `SpeechAnalyzer` + `SpeechTranscriber` (WWDC25). The
//       current API: on-device by design, purpose-built for long-form and
//       streaming input, and markedly faster than the old one. Swift-only, and
//       it requires **macOS 26**, both to compile and to run.
//   kSpeechRecognizer -- `SFSpeechRecognizer` with an on-device request. The
//       previous API, available back to macOS 10.15. It still works and is
//       still on-device; it is just slower and its per-utterance model is a
//       poorer fit for a continuously open mic.
//
// Which backends exist in a given binary is a *compile-time* property of the
// SDK it was built against (see kBuiltWithSpeechAnalyzer), and which of those
// actually run is a *runtime* property of the OS. Both are reported rather than
// assumed, so a binary built on one machine and run on another degrades to a
// legible message instead of a link error or a crash.
//
// Threading: feed() is called from the UI thread with samples drained from the
// audio ring -- never from the CoreAudio render callback, since every backend
// here allocates and locks. Results arrive on some internal queue and are
// buffered until poll() picks them up, so the UI never blocks on recognition.

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace transcribe {

enum class Backend {
  kAuto,              // best available at runtime
  kSpeechAnalyzer,    // macOS 26+
  kSpeechRecognizer,  // macOS 10.15+
};

const char* BackendName(Backend b);

// True if this binary was *compiled* with the SpeechAnalyzer backend, i.e.
// against the macOS 26 SDK or newer. False here means the backend is absent
// from the build entirely -- upgrading the OS alone will not bring it back
// without a rebuild.
bool BuiltWithSpeechAnalyzer();

// True if `b` can actually be started on the machine we are running on now.
// Answers the OS-version question, not the SDK question.
bool BackendAvailable(Backend b);

// A hypothesis about what was said.
//
// Streaming recognisers emit a given stretch of speech many times over,
// refining it: `is_final` marks the point past which a segment will not be
// revised. Volatile results are what makes the display feel live; final ones
// are what you keep. Both carry the same `text` field so a caller that does not
// care about the distinction can ignore it.
struct Result {
  std::string text;
  bool is_final = false;
};

class Transcriber {
 public:
  virtual ~Transcriber() = default;

  // `locale` is a BCP-47 identifier ("en-US"). Fails, with `err` explaining
  // why, if the locale has no on-device model, if permission was refused, or if
  // the backend is unavailable on this OS.
  virtual bool start(double sample_rate, const std::string& locale,
                     std::string& err) = 0;
  virtual void stop() = 0;
  virtual bool running() const = 0;

  // Mono float samples at the rate passed to start(). Cheap and non-blocking:
  // it copies into a queue and returns. Safe to call with n == 0.
  virtual void feed(const float* samples, int n) = 0;

  // Moves any results produced since the last call into `out`. Never blocks.
  virtual void poll(std::vector<Result>& out) = 0;

  // Set once the backend fails mid-stream; empty while healthy. A recogniser
  // that dies quietly (model eviction, permission revoked) would otherwise look
  // exactly like a silent room.
  virtual std::string error() const = 0;

  virtual Backend backend() const = 0;
};

// Returns null with `err` set if no backend is usable. kAuto prefers
// SpeechAnalyzer and falls back to SFSpeechRecognizer.
std::unique_ptr<Transcriber> Create(Backend want, std::string& err);

// Prompts for speech-recognition authorisation and blocks until the user
// answers or `timeout_s` elapses. Call it before start().
//
// As with microphone access, a plain CLI binary has no bundle identity of its
// own, so macOS attributes the request to the *parent* application -- your
// terminal or IDE -- and that is where the grant shows up in System Settings.
bool RequestAuthorization(double timeout_s, std::string& err);

// --- Transcript accumulation -------------------------------------------------

// Folds a stream of Results into a stable transcript plus the one unstable tail
// being revised, which is all a UI needs to render.
//
// Kept here, out of the backends, because it is pure bookkeeping and identical
// for both -- and because it is then testable without a microphone, an OS
// permission, or an on-device model.
class TranscriptLog {
 public:
  explicit TranscriptLog(size_t max_finals = 200) : max_finals_(max_finals) {}

  void add(const Result& r);
  void clear();

  // Finalised text, oldest first, capped at max_finals entries.
  const std::vector<std::string>& finals() const { return finals_; }
  // The in-flight hypothesis, or empty when there is none pending.
  const std::string& partial() const { return partial_; }

  // Everything joined with spaces, finals then partial.
  std::string full() const;

 private:
  std::vector<std::string> finals_;
  std::string partial_;
  size_t max_finals_;
};

}  // namespace transcribe
