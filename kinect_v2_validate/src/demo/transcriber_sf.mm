// SFSpeechRecognizer transcription backend, and the factory that chooses
// between backends.
//
// This is the API available on macOS 10.15 through 15.x. It is on-device (we
// require it to be -- see below), but its model is per-*utterance*: a task
// accumulates one recognition and its hypotheses only ever grow. Left running
// on a permanently open mic it drifts, slows, and eventually hits Apple's
// undocumented per-task duration limit. So the task is recycled on silence,
// which is what turns a one-shot API into a continuous one. See kRestartAfter.
//
// The newer SpeechAnalyzer backend has none of these problems and is preferred
// when the SDK and OS provide it; see transcriber_analyzer.swift.

#include "transcriber.h"

#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <Speech/Speech.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <mutex>

#if defined(__has_include)
#if __has_include(<Speech/SFSpeechRecognitionResult.h>)
#define KV2_HAVE_SPEECH 1
#endif
#endif

// Objective-C declarations are only legal at global scope, so the result sink's
// interface sits out here while everything else lives in `transcribe`. The
// class it talks back into is named (transcribe::detail) rather than
// anonymous for the same reason: the @implementation below has to name it.
@class KV2SpeechSink;

namespace transcribe {

// Declared by the Swift backend when it is part of the build. Guarded by
// KV2_WITH_SPEECH_ANALYZER (set by CMake only when the SDK is macOS 26+), so
// the symbols are referenced only when they exist.
#if KV2_WITH_SPEECH_ANALYZER
std::unique_ptr<Transcriber> CreateSpeechAnalyzer(std::string& err);
bool SpeechAnalyzerRuntimeAvailable();
#endif

bool BuiltWithSpeechAnalyzer() {
#if KV2_WITH_SPEECH_ANALYZER
  return true;
#else
  return false;
#endif
}

bool BackendAvailable(Backend b) {
  switch (b) {
    case Backend::kSpeechAnalyzer:
#if KV2_WITH_SPEECH_ANALYZER
      return SpeechAnalyzerRuntimeAvailable();
#else
      return false;
#endif
    case Backend::kSpeechRecognizer:
      // Present on every macOS this project targets.
      return true;
    case Backend::kAuto:
      return BackendAvailable(Backend::kSpeechAnalyzer) ||
             BackendAvailable(Backend::kSpeechRecognizer);
  }
  return false;
}

namespace {

// Recycle the recognition task once this much silence has passed. Long enough
// not to chop a speaker mid-sentence at a natural pause, short enough that a
// task never accumulates the many minutes of audio that degrade it.
constexpr double kRestartAfterSilenceS = 1.2;
// Below this peak amplitude a block counts as silence for the purpose above.
// Deliberately generous: the beam output is already compressed and gained up,
// so genuine room silence still sits well under it.
constexpr float kSilencePeak = 0.012f;
// Hard ceiling on a single task regardless of pauses, for the speaker who
// simply does not stop. SFSpeechRecognizer's own limit is around a minute.
constexpr double kMaxTaskS = 50.0;

}  // namespace

namespace detail {

class SfTranscriber final : public Transcriber {
 public:
  ~SfTranscriber() override { stop(); }

  bool start(double sample_rate, const std::string& locales,
             std::string& err) override;
  void stop() override;
  bool running() const override { return running_; }
  void feed(const float* samples, int n) override;
  void poll(std::vector<Result>& out) override;
  std::string error() const override {
    std::lock_guard<std::mutex> lk(mu_);
    return error_;
  }
  Backend backend() const override { return Backend::kSpeechRecognizer; }

  // Called from the Speech delegate queue.
  void onResult(const std::string& text, bool is_final);
  void onError(const std::string& msg);

 private:
  // Tears down the current task and opens a fresh one. Any text the old task
  // had not yet finalised is emitted as final first, so recycling never eats a
  // partial utterance.
  void restartTask();
  bool beginTask(std::string& err);
  void endTask(bool flush_partial);

  mutable std::mutex mu_;
  std::deque<Result> pending_;
  std::string error_;
  std::string last_partial_;
  // The single locale this recogniser was actually opened with, stamped onto
  // every Result. Set once in start() and only read afterwards.
  std::string locale_;

  SFSpeechRecognizer* recognizer_ = nil;
  SFSpeechAudioBufferRecognitionRequest* request_ = nil;
  SFSpeechRecognitionTask* task_ = nil;
  KV2SpeechSink* sink_ = nil;
  AVAudioFormat* format_ = nil;

  double sample_rate_ = 16000.0;
  bool running_ = false;

  // Silence/duration bookkeeping for task recycling, touched only by feed().
  double silence_s_ = 0.0;
  double task_s_ = 0.0;
};

}  // namespace detail
}  // namespace transcribe

// --- Objective-C delegate ----------------------------------------------------

// Results arrive on an arbitrary queue. This hands them to a C++ sink that is
// mutex-guarded, so the UI thread can poll without coordinating with Speech.
@interface KV2SpeechSink : NSObject
@property(nonatomic, assign) void* owner;
- (void)handleResult:(SFSpeechRecognitionResult*)result error:(NSError*)error;
@end

@implementation KV2SpeechSink
- (void)handleResult:(SFSpeechRecognitionResult*)result error:(NSError*)error {
  using transcribe::detail::SfTranscriber;
  SfTranscriber* self_owner = static_cast<SfTranscriber*>(self.owner);
  if (!self_owner) return;
  if (result) {
    const std::string text =
        result.bestTranscription.formattedString
            ? std::string(result.bestTranscription.formattedString.UTF8String)
            : std::string();
    self_owner->onResult(text, result.isFinal);
  }
  if (error) {
    // kAFAssistantErrorDomain 216/301 are the ordinary "task cancelled"
    // codes we provoke ourselves every time we recycle; they are not faults.
    const BOOL benign = [error.domain isEqualToString:@"kAFAssistantErrorDomain"] &&
                        (error.code == 216 || error.code == 301 || error.code == 203);
    if (!benign) {
      self_owner->onError(std::string(error.localizedDescription.UTF8String));
    }
  }
}
@end

namespace transcribe {
namespace detail {

bool SfTranscriber::start(double sample_rate, const std::string& locales,
                          std::string& err) {
  if (running_) return true;
  sample_rate_ = sample_rate > 0 ? sample_rate : 16000.0;

  // One recogniser, one language. Racing several locales the way the analyzer
  // backend does would mean several SFSpeechRecognizers each recycling their
  // own task on their own silence timer, which is a great deal of machinery for
  // the backend we prefer *not* to use. Taking the first is the honest
  // degradation, and locale_ is reported on every Result so the caller can see
  // which one it got.
  const std::vector<std::string> list = SplitLocales(locales);
  const std::string locale = list.empty() ? std::string("en-US") : list.front();
  locale_ = locale;

  NSString* loc = [NSString stringWithUTF8String:locale.c_str()];
  recognizer_ = [[SFSpeechRecognizer alloc]
      initWithLocale:[NSLocale localeWithLocaleIdentifier:loc]];
  if (!recognizer_) {
    err = "no speech recogniser for locale '" + locale + "'";
    return false;
  }
  if (!recognizer_.isAvailable) {
    err = "speech recogniser for '" + locale + "' is not available right now";
    return false;
  }
  // On-device only, and not as a preference: the alternative is shipping mic
  // audio of whoever walks past the sensor to Apple's servers. If the device
  // model is missing we fail loudly rather than quietly going online.
  if (!recognizer_.supportsOnDeviceRecognition) {
    err = "no on-device model for '" + locale +
          "' (Settings > Keyboard > Dictation languages installs it); "
          "refusing to fall back to server recognition";
    return false;
  }

  // The beam is mono float32 at the array's rate. Non-interleaved so we can
  // hand SFSpeech the channel pointer directly.
  format_ = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                             sampleRate:sample_rate_
                                               channels:1
                                            interleaved:NO];
  if (!format_) {
    err = "could not build a 1ch float32 audio format";
    return false;
  }

  sink_ = [[KV2SpeechSink alloc] init];
  sink_.owner = this;

  running_ = true;
  silence_s_ = 0.0;
  task_s_ = 0.0;
  if (!beginTask(err)) {
    running_ = false;
    return false;
  }
  return true;
}

bool SfTranscriber::beginTask(std::string& err) {
  request_ = [[SFSpeechAudioBufferRecognitionRequest alloc] init];
  if (!request_) {
    err = "could not create a recognition request";
    return false;
  }
  request_.shouldReportPartialResults = YES;
  request_.requiresOnDeviceRecognition = YES;
  if (@available(macOS 13.0, *)) {
    // Punctuation makes the transcript readable, which is the whole point of
    // showing it on screen rather than logging it.
    request_.addsPunctuation = YES;
  }
  // Dictation hint: this is a person talking to a camera, not a search query.
  request_.taskHint = SFSpeechRecognitionTaskHintDictation;

  KV2SpeechSink* sink = sink_;
  task_ = [recognizer_ recognitionTaskWithRequest:request_
                                    resultHandler:^(SFSpeechRecognitionResult* r,
                                                    NSError* e) {
                                      [sink handleResult:r error:e];
                                    }];
  if (!task_) {
    err = "could not start a recognition task";
    return false;
  }
  task_s_ = 0.0;
  silence_s_ = 0.0;
  return true;
}

void SfTranscriber::endTask(bool flush_partial) {
  if (request_) [request_ endAudio];
  if (task_) [task_ cancel];
  request_ = nil;
  task_ = nil;

  if (flush_partial) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!last_partial_.empty()) {
      // Promote it rather than dropping it: the words were spoken, and this
      // task will never get the chance to finalise them itself.
      pending_.push_back(Result{last_partial_, true, locale_});
      last_partial_.clear();
    }
  }
}

void SfTranscriber::restartTask() {
  endTask(/*flush_partial=*/true);
  std::string err;
  if (!beginTask(err)) {
    std::lock_guard<std::mutex> lk(mu_);
    error_ = "could not restart recognition: " + err;
    running_ = false;
  }
}

void SfTranscriber::stop() {
  if (!running_) return;
  running_ = false;
  endTask(/*flush_partial=*/true);
  if (sink_) sink_.owner = nullptr;
  sink_ = nil;
  recognizer_ = nil;
  format_ = nil;
}

void SfTranscriber::feed(const float* samples, int n) {
  if (!running_ || !request_ || n <= 0 || !samples) return;

  AVAudioPCMBuffer* buf =
      [[AVAudioPCMBuffer alloc] initWithPCMFormat:format_
                                    frameCapacity:(AVAudioFrameCount)n];
  if (!buf) return;
  buf.frameLength = (AVAudioFrameCount)n;
  std::copy(samples, samples + n, buf.floatChannelData[0]);
  [request_ appendAudioPCMBuffer:buf];

  // Recycle on a pause, so each task covers roughly one utterance.
  float peak = 0.f;
  for (int i = 0; i < n; ++i) peak = std::max(peak, std::fabs(samples[i]));
  const double dt = double(n) / sample_rate_;
  task_s_ += dt;
  silence_s_ = (peak < kSilencePeak) ? silence_s_ + dt : 0.0;

  // Only recycle on silence once there is something to close off; restarting
  // every 1.2 s in an empty room would churn tasks for no reason.
  const bool idle_break = silence_s_ >= kRestartAfterSilenceS &&
                          (!last_partial_.empty() || task_s_ > 8.0);
  if (idle_break || task_s_ >= kMaxTaskS) restartTask();
}

void SfTranscriber::onResult(const std::string& text, bool is_final) {
  std::lock_guard<std::mutex> lk(mu_);
  if (is_final) {
    last_partial_.clear();
  } else {
    last_partial_ = text;
  }
  pending_.push_back(Result{text, is_final, locale_});
  // Bound the queue: if the UI stops polling we would otherwise grow forever.
  while (pending_.size() > 256) pending_.pop_front();
}

void SfTranscriber::onError(const std::string& msg) {
  std::lock_guard<std::mutex> lk(mu_);
  error_ = msg;
}

void SfTranscriber::poll(std::vector<Result>& out) {
  std::lock_guard<std::mutex> lk(mu_);
  out.insert(out.end(), pending_.begin(), pending_.end());
  pending_.clear();
}

}  // namespace detail

// --- factory + authorisation -------------------------------------------------

std::unique_ptr<Transcriber> Create(Backend want, std::string& err) {
  if (want == Backend::kAuto || want == Backend::kSpeechAnalyzer) {
#if KV2_WITH_SPEECH_ANALYZER
    if (SpeechAnalyzerRuntimeAvailable()) {
      std::string sub;
      std::unique_ptr<Transcriber> t = CreateSpeechAnalyzer(sub);
      if (t) return t;
      if (want == Backend::kSpeechAnalyzer) {
        err = "SpeechAnalyzer: " + sub;
        return nullptr;
      }
    } else if (want == Backend::kSpeechAnalyzer) {
      err = "SpeechAnalyzer needs macOS 26 or newer";
      return nullptr;
    }
#else
    if (want == Backend::kSpeechAnalyzer) {
      err = "this binary was built against an SDK older than macOS 26, so the "
            "SpeechAnalyzer backend is not compiled in";
      return nullptr;
    }
#endif
  }

  return std::unique_ptr<Transcriber>(new detail::SfTranscriber());
}

bool RequestAuthorization(double timeout_s, std::string& err) {
  __block SFSpeechRecognizerAuthorizationStatus status =
      [SFSpeechRecognizer authorizationStatus];
  if (status == SFSpeechRecognizerAuthorizationStatusAuthorized) return true;

  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  [SFSpeechRecognizer requestAuthorization:^(
                          SFSpeechRecognizerAuthorizationStatus s) {
    status = s;
    dispatch_semaphore_signal(sem);
  }];
  const dispatch_time_t deadline =
      dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout_s * NSEC_PER_SEC));
  if (dispatch_semaphore_wait(sem, deadline) != 0) {
    err = "timed out waiting for the speech-recognition permission dialog";
    return false;
  }

  switch (status) {
    case SFSpeechRecognizerAuthorizationStatusAuthorized:
      return true;
    case SFSpeechRecognizerAuthorizationStatusDenied:
      err = "speech recognition was denied. These are plain CLI binaries, so "
            "macOS attributes the request to the parent terminal: grant it "
            "under System Settings > Privacy & Security > Speech Recognition.";
      return false;
    case SFSpeechRecognizerAuthorizationStatusRestricted:
      err = "speech recognition is restricted on this device";
      return false;
    default:
      err = "speech recognition is not authorised";
      return false;
  }
}

}  // namespace transcribe
