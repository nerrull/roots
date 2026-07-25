// Voice cloning: capture a reference clip of whoever is in front of the sensor,
// then synthesise arbitrary text in that voice.
//
// The model does not run in this process. Every good zero-shot cloning model is
// a Python artefact, and the ones worth using on this machine (Chatterbox,
// Qwen3-TTS) reach the GPU through MLX, which has no C++ inference path for
// them. Wrapping that in C++ would mean either embedding CPython in a realtime
// UI process or reimplementing a model. Both are worse than a socket.
//
// So: `tools/voice_server.py` holds the model resident and exposes it over
// loopback HTTP; this class is the client. The split has a second payoff --
// model loading (seconds to tens of seconds) and generation never touch the
// render loop, and the server can be restarted, re-pointed at a different
// model, or curl'd by hand without rebuilding the demo.
//
// Threading: speak() returns immediately and does the request on a background
// queue. The UI polls state(). Nothing here is called from the audio callback.

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace voice {

// --- WAV -----------------------------------------------------------------

// 16-bit mono PCM. Written for the reference clip, which is what the server
// wants and what you will actually want to audition by hand when a clone comes
// out wrong.
bool WriteWav16(const std::string& path, const std::vector<float>& samples,
                int sample_rate, std::string& err);

// Reads a 16-bit or 32-bit-float mono/stereo PCM WAV into mono floats. Only
// what WriteWav16 and the server produce -- not a general WAV parser.
bool ReadWavMono(const std::string& path, std::vector<float>& samples,
                 int* sample_rate, std::string& err);

// --- reference clip ----------------------------------------------------------

// Accumulates fed samples until `seconds` have been captured.
//
// The clip is taken from the *beamformed, compressed* output rather than a raw
// mic, deliberately: that is the signal already steered at the talker, and a
// cleaner reference is the single biggest lever on clone quality. The trade is
// that the beam's dynamics processing is baked into the reference -- audible as
// a slightly flatter clone, and worth it against the room noise it removes.
class ReferenceRecorder {
 public:
  void begin(double seconds, int sample_rate);
  void cancel();

  // Returns true once the clip is complete. Extra samples past the target are
  // ignored, so a late feed cannot overrun the clip.
  bool feed(const float* samples, int n);

  bool recording() const { return recording_; }
  bool complete() const { return !recording_ && !buf_.empty(); }
  double capturedSeconds() const {
    return sample_rate_ > 0 ? double(buf_.size()) / sample_rate_ : 0.0;
  }
  double targetSeconds() const { return target_s_; }
  int sampleRate() const { return sample_rate_; }
  const std::vector<float>& samples() const { return buf_; }

  // Peak amplitude of the captured clip. A reference clip that is nearly
  // silent produces a clone that sounds like nothing in particular, and the
  // failure is otherwise invisible until you hear the output.
  float peak() const;

  bool save(const std::string& path, std::string& err) const;

 private:
  std::vector<float> buf_;
  size_t target_samples_ = 0;
  double target_s_ = 0.0;
  int sample_rate_ = 16000;
  bool recording_ = false;
};

// --- synthesis client --------------------------------------------------------

struct SpeakParams {
  std::string text;
  std::string reference_wav;  // path the *server* will read; empty = default voice

  // Transcript of the reference clip. Qwen3-TTS conditions on it and clones
  // noticeably better with it; Chatterbox has no such parameter and the server
  // drops it. Optional everywhere -- an empty string is never an error.
  //
  // The demo fills this from its own transcriber when transcription is running
  // during the reference capture, which is the one place the two speech
  // features feed each other.
  std::string reference_text;

  // Emotional intensity, as a scalar. Chatterbox calls it "exaggeration": ~0.3
  // is flat and even, 0.5 natural, past ~0.8 theatrical and starting to lose
  // the speaker's identity.
  float exaggeration = 0.5f;
  // Classifier-free guidance / pace. Lower is slower and more deliberate.
  float cfg = 0.5f;
  float temperature = 0.8f;

  // Emotional direction as a *sentence* ("speak warmly and slowly"). The two
  // model families genuinely disagree about how emotion is specified --
  // Chatterbox takes the scalar above, Qwen3-TTS takes this -- and there is no
  // honest conversion between them, so both travel and each model picks up the
  // one it understands.
  std::string style;
};

// Where synthesis stands. There is exactly one request in flight at a time --
// the model is not reentrant and queuing would only build latency.
enum class State {
  kIdle,
  kGenerating,
  kPlaying,
  kError,
};

const char* StateName(State s);

struct ServerInfo {
  bool reachable = false;
  std::string model;        // what the server actually loaded
  bool model_loaded = false;
  std::string detail;       // error text when unreachable
};

class VoiceCloner {
 public:
  VoiceCloner();
  ~VoiceCloner();

  // Loopback only by default. This posts microphone audio of whoever is in
  // front of the sensor to a synthesis service; pointing it off-box would send
  // that recording over the network, so the default stays on 127.0.0.1 and any
  // other host has to be typed in deliberately.
  void setEndpoint(const std::string& base_url);
  const std::string& endpoint() const { return base_url_; }

  // Non-blocking; result lands in info() a moment later.
  void probeAsync();
  ServerInfo info() const;

  // Returns false if a request is already in flight or the text is empty.
  bool speak(const SpeakParams& p);

  // Stops playback of the current clip. Does not cancel generation -- the
  // server has no cancellation path, and a half-generated clip is discarded on
  // arrival anyway.
  void stopPlayback();

  State state() const { return state_.load(); }
  std::string errorText() const;
  // Wall-clock seconds for the last completed request, and the duration of the
  // audio it produced. Their ratio is the real-time factor, which is the number
  // that decides whether a given model is usable here.
  double lastGenerateSeconds() const { return last_generate_s_.load(); }
  double lastAudioSeconds() const { return last_audio_s_.load(); }
  // Path of the most recently generated WAV, kept so it can be replayed or
  // inspected.
  std::string lastWavPath() const;

  // Plays a WAV that already exists, e.g. the captured reference clip, so you
  // can hear what the model is being asked to imitate.
  bool playFile(const std::string& path, std::string& err);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  std::string base_url_ = "http://127.0.0.1:8765";
  std::atomic<State> state_{State::kIdle};
  std::atomic<double> last_generate_s_{0.0};
  std::atomic<double> last_audio_s_{0.0};

  mutable std::mutex mu_;
  std::string error_text_;
  std::string last_wav_;
  ServerInfo info_;
};

}  // namespace voice
