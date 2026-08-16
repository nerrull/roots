// Backend-independent parts of the transcriber: naming and transcript
// accumulation. No Apple frameworks here, so this compiles and tests anywhere.

#include "transcriber.h"

#include <algorithm>
#include <cctype>

namespace transcribe {

const char* BackendName(Backend b) {
  switch (b) {
    case Backend::kAuto: return "auto";
    case Backend::kSpeechAnalyzer: return "SpeechAnalyzer";
    case Backend::kSpeechRecognizer: return "SFSpeechRecognizer";
  }
  return "?";
}

std::vector<std::string> SplitLocales(const std::string& locales) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i <= locales.size()) {
    const size_t comma = locales.find(',', i);
    const size_t end = comma == std::string::npos ? locales.size() : comma;
    size_t b = i, e = end;
    while (b < e && std::isspace(static_cast<unsigned char>(locales[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(locales[e - 1]))) --e;
    if (e > b) out.push_back(locales.substr(b, e - b));
    if (comma == std::string::npos) break;
    i = comma + 1;
  }
  return out;
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

void SpeechCapture::reset(int sample_rate, double keep_seconds) {
  buf_.clear();
  marks_.clear();
  total_ = 0;
  sample_rate_ = sample_rate > 0 ? sample_rate : 16000;
  const double n = keep_seconds > 0 ? keep_seconds * sample_rate_ : 0;
  keep_ = static_cast<size_t>(n < 0 ? 0 : n);
}

void SpeechCapture::append(const float* samples, int n) {
  if (!samples || n <= 0 || keep_ == 0) return;
  buf_.insert(buf_.end(), samples, samples + n);
  total_ += static_cast<uint64_t>(n);
  if (buf_.size() > keep_) {
    buf_.erase(buf_.begin(), buf_.begin() + (buf_.size() - keep_));
  }
  // A mark whose audio has slid out of the window names sound that no longer
  // exists, so it goes with it.
  const uint64_t oldest = total_ - buf_.size();
  size_t drop = 0;
  while (drop < marks_.size() && marks_[drop].first < oldest) ++drop;
  if (drop > 0) marks_.erase(marks_.begin(), marks_.begin() + drop);
}

void SpeechCapture::markFinal(const std::string& text) {
  if (text.empty()) return;
  marks_.emplace_back(total_, text);
}

void SpeechCapture::takeLast(double seconds, std::vector<float>& clip,
                             std::string& text) const {
  clip.clear();
  text.clear();
  if (buf_.empty() || seconds <= 0) return;

  const double want_f = seconds * sample_rate_;
  const size_t want = std::min<size_t>(
      buf_.size(), want_f <= 0 ? 0 : static_cast<size_t>(want_f));
  if (want == 0) return;
  clip.assign(buf_.end() - want, buf_.end());

  // Marks lag their audio, so a clip's opening words may have been marked just
  // before it starts. Erring towards including a mark keeps the reference
  // transcript complete; erring the other way silently truncates the text the
  // model conditions on.
  const uint64_t from = total_ - want;
  for (const std::pair<uint64_t, std::string>& m : marks_) {
    if (m.first < from) continue;
    if (!text.empty()) text += ' ';
    text += m.second;
  }
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
