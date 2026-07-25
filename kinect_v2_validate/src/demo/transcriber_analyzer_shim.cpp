// Adapts the Swift SpeechAnalyzer backend's C entry points to the Transcriber
// interface. Built only when KV2_WITH_SPEECH_ANALYZER is set, i.e. against the
// macOS 26 SDK or newer -- see transcriber_analyzer.swift and CMakeLists.txt.
//
// Nothing here knows anything about Speech; it is pure marshalling. The
// declarations below must stay in step with the @_cdecl signatures on the Swift
// side, which is the one cost of using a C ABI instead of Swift/C++ interop.

#include "transcriber.h"

#include <string>
#include <vector>

extern "C" {
bool kv2_sa_available();
void* kv2_sa_create();
void kv2_sa_destroy(void* handle);
bool kv2_sa_start(void* handle, double sample_rate, const char* locale,
                  char* err, int err_len);
void kv2_sa_feed(void* handle, const float* samples, int count);
void kv2_sa_stop(void* handle);
bool kv2_sa_poll(void* handle, char* text, int text_len, bool* is_final);
void kv2_sa_error(void* handle, char* buf, int buf_len);
}

namespace transcribe {
namespace {

// One utterance's worth of text, with room to spare.
constexpr int kTextBuf = 8192;
constexpr int kErrBuf = 1024;

class AnalyzerTranscriber final : public Transcriber {
 public:
  ~AnalyzerTranscriber() override {
    stop();
    if (handle_) kv2_sa_destroy(handle_);
  }

  bool create(std::string& err) {
    handle_ = kv2_sa_create();
    if (!handle_) {
      err = "SpeechAnalyzer requires macOS 26 or newer";
      return false;
    }
    return true;
  }

  bool start(double sample_rate, const std::string& locale,
             std::string& err) override {
    if (running_) return true;
    std::vector<char> buf(kErrBuf, 0);
    if (!kv2_sa_start(handle_, sample_rate, locale.c_str(), buf.data(),
                      kErrBuf)) {
      err = buf.data();
      if (err.empty()) err = "SpeechAnalyzer failed to start";
      return false;
    }
    running_ = true;
    return true;
  }

  void stop() override {
    if (!running_) return;
    running_ = false;
    kv2_sa_stop(handle_);
  }

  bool running() const override { return running_; }

  void feed(const float* samples, int n) override {
    if (!running_ || n <= 0 || !samples) return;
    kv2_sa_feed(handle_, samples, n);
  }

  void poll(std::vector<Result>& out) override {
    if (!handle_) return;
    std::vector<char> buf(kTextBuf, 0);
    bool is_final = false;
    // Bounded so a flood of volatile results can never stall the UI thread.
    for (int i = 0; i < 256; ++i) {
      if (!kv2_sa_poll(handle_, buf.data(), kTextBuf, &is_final)) break;
      out.push_back(Result{std::string(buf.data()), is_final});
    }
  }

  std::string error() const override {
    if (!handle_) return {};
    std::vector<char> buf(kErrBuf, 0);
    kv2_sa_error(handle_, buf.data(), kErrBuf);
    return std::string(buf.data());
  }

  Backend backend() const override { return Backend::kSpeechAnalyzer; }

 private:
  void* handle_ = nullptr;
  bool running_ = false;
};

}  // namespace

bool SpeechAnalyzerRuntimeAvailable() { return kv2_sa_available(); }

std::unique_ptr<Transcriber> CreateSpeechAnalyzer(std::string& err) {
  std::unique_ptr<AnalyzerTranscriber> t(new AnalyzerTranscriber());
  if (!t->create(err)) return nullptr;
  return t;
}

}  // namespace transcribe
