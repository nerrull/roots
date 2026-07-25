// Backend-independent parts of the transcriber: naming and transcript
// accumulation. No Apple frameworks here, so this compiles and tests anywhere.

#include "transcriber.h"

namespace transcribe {

const char* BackendName(Backend b) {
  switch (b) {
    case Backend::kAuto: return "auto";
    case Backend::kSpeechAnalyzer: return "SpeechAnalyzer";
    case Backend::kSpeechRecognizer: return "SFSpeechRecognizer";
  }
  return "?";
}

void TranscriptLog::add(const Result& r) {
  if (r.is_final) {
    // A final result supersedes the partial it was being refined from --
    // keeping both would duplicate the utterance in the log.
    partial_.clear();
    if (r.text.empty()) return;
    finals_.push_back(r.text);
    if (finals_.size() > max_finals_) {
      finals_.erase(finals_.begin(),
                    finals_.begin() + (finals_.size() - max_finals_));
    }
  } else {
    // Volatile results replace rather than append: each one is a new guess at
    // the *same* stretch of speech, not a continuation of it.
    partial_ = r.text;
  }
}

void TranscriptLog::clear() {
  finals_.clear();
  partial_.clear();
}

std::string TranscriptLog::full() const {
  std::string out;
  for (const std::string& s : finals_) {
    if (!out.empty()) out += ' ';
    out += s;
  }
  if (!partial_.empty()) {
    if (!out.empty()) out += ' ';
    out += partial_;
  }
  return out;
}

}  // namespace transcribe
