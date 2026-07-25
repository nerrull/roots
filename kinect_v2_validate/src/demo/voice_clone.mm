// Voice-cloning client: WAV I/O, reference capture, and the HTTP call to
// tools/voice_server.py. See voice_clone.h for why the model lives out of
// process.

#include "voice_clone.h"

#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace voice {

const char* StateName(State s) {
  switch (s) {
    case State::kIdle: return "idle";
    case State::kGenerating: return "generating";
    case State::kPlaying: return "playing";
    case State::kError: return "error";
  }
  return "?";
}

// --- WAV ---------------------------------------------------------------------

namespace {

void PutU32(std::vector<uint8_t>& v, uint32_t x) {
  v.push_back(uint8_t(x));
  v.push_back(uint8_t(x >> 8));
  v.push_back(uint8_t(x >> 16));
  v.push_back(uint8_t(x >> 24));
}
void PutU16(std::vector<uint8_t>& v, uint16_t x) {
  v.push_back(uint8_t(x));
  v.push_back(uint8_t(x >> 8));
}
void PutTag(std::vector<uint8_t>& v, const char* t) {
  for (int i = 0; i < 4; ++i) v.push_back(uint8_t(t[i]));
}

uint32_t GetU32(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}
uint16_t GetU16(const uint8_t* p) {
  return uint16_t(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
}

}  // namespace

bool WriteWav16(const std::string& path, const std::vector<float>& samples,
                int sample_rate, std::string& err) {
  if (sample_rate <= 0) {
    err = "invalid sample rate";
    return false;
  }
  const uint32_t n = uint32_t(samples.size());
  const uint32_t data_bytes = n * 2;

  std::vector<uint8_t> out;
  out.reserve(44 + data_bytes);
  PutTag(out, "RIFF");
  PutU32(out, 36 + data_bytes);
  PutTag(out, "WAVE");
  PutTag(out, "fmt ");
  PutU32(out, 16);                       // PCM fmt chunk size
  PutU16(out, 1);                        // PCM
  PutU16(out, 1);                        // mono
  PutU32(out, uint32_t(sample_rate));
  PutU32(out, uint32_t(sample_rate) * 2);  // byte rate
  PutU16(out, 2);                        // block align
  PutU16(out, 16);                       // bits
  PutTag(out, "data");
  PutU32(out, data_bytes);

  for (float s : samples) {
    // Clamp before scaling: a beam sample can exceed 1.0 when the limiter is
    // off, and wrapping that into int16 is the loudest possible bug.
    const float c = std::min(std::max(s, -1.f), 1.f);
    const int32_t v = int32_t(std::lround(c * 32767.f));
    const uint16_t u = uint16_t(int16_t(std::min(std::max(v, -32768), 32767)));
    PutU16(out, u);
  }

  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    err = "could not open '" + path + "' for writing";
    return false;
  }
  const size_t wrote = std::fwrite(out.data(), 1, out.size(), f);
  std::fclose(f);
  if (wrote != out.size()) {
    err = "short write to '" + path + "'";
    return false;
  }
  return true;
}

bool ReadWavMono(const std::string& path, std::vector<float>& samples,
                 int* sample_rate, std::string& err) {
  samples.clear();
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    err = "could not open '" + path + "'";
    return false;
  }
  std::fseek(f, 0, SEEK_END);
  const long len = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (len < 44) {
    std::fclose(f);
    err = "'" + path + "' is too short to be a WAV";
    return false;
  }
  std::vector<uint8_t> buf(static_cast<size_t>(len), 0);
  const size_t got = std::fread(buf.data(), 1, buf.size(), f);
  std::fclose(f);
  if (got != buf.size()) {
    err = "short read from '" + path + "'";
    return false;
  }
  if (std::memcmp(buf.data(), "RIFF", 4) != 0 ||
      std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
    err = "'" + path + "' is not a RIFF/WAVE file";
    return false;
  }

  // Walk the chunk list rather than assuming a 44-byte header: anything that
  // has been through a resampler tends to carry a LIST or fact chunk first.
  uint16_t fmt = 0, channels = 1, bits = 16;
  uint32_t rate = 0;
  size_t pos = 12;
  const uint8_t* data = nullptr;
  uint32_t data_len = 0;
  while (pos + 8 <= buf.size()) {
    const char* tag = reinterpret_cast<const char*>(&buf[pos]);
    const uint32_t sz = GetU32(&buf[pos + 4]);
    const size_t body = pos + 8;
    if (body + sz > buf.size()) break;
    if (std::memcmp(tag, "fmt ", 4) == 0 && sz >= 16) {
      fmt = GetU16(&buf[body]);
      channels = GetU16(&buf[body + 2]);
      rate = GetU32(&buf[body + 4]);
      bits = GetU16(&buf[body + 14]);
    } else if (std::memcmp(tag, "data", 4) == 0) {
      data = &buf[body];
      data_len = sz;
    }
    pos = body + sz + (sz & 1);  // chunks are word-aligned
  }
  if (!data || channels == 0 || rate == 0) {
    err = "'" + path + "' has no usable fmt/data chunks";
    return false;
  }

  // fmt 1 = integer PCM, 3 = IEEE float. Anything else is compressed.
  const int ch = channels;
  if (fmt == 1 && bits == 16) {
    const uint32_t frames = data_len / uint32_t(2 * ch);
    samples.resize(frames);
    for (uint32_t i = 0; i < frames; ++i) {
      // Downmix by averaging; the server emits mono, but a hand-supplied
      // reference clip may not.
      float acc = 0.f;
      for (int c = 0; c < ch; ++c) {
        acc += float(int16_t(GetU16(data + (size_t(i) * ch + c) * 2))) / 32768.f;
      }
      samples[i] = acc / float(ch);
    }
  } else if (fmt == 3 && bits == 32) {
    const uint32_t frames = data_len / uint32_t(4 * ch);
    samples.resize(frames);
    for (uint32_t i = 0; i < frames; ++i) {
      float acc = 0.f;
      for (int c = 0; c < ch; ++c) {
        float v;
        std::memcpy(&v, data + (size_t(i) * ch + c) * 4, 4);
        acc += v;
      }
      samples[i] = acc / float(ch);
    }
  } else {
    char msg[128];
    std::snprintf(msg, sizeof(msg),
                  "unsupported WAV encoding (format %u, %u bits)", fmt, bits);
    err = msg;
    return false;
  }

  if (sample_rate) *sample_rate = int(rate);
  return true;
}

// --- reference clip ----------------------------------------------------------

void ReferenceRecorder::begin(double seconds, int sample_rate) {
  sample_rate_ = sample_rate > 0 ? sample_rate : 16000;
  target_s_ = std::max(seconds, 0.5);
  target_samples_ = size_t(target_s_ * sample_rate_);
  buf_.clear();
  buf_.reserve(target_samples_);
  recording_ = true;
}

void ReferenceRecorder::cancel() {
  recording_ = false;
  buf_.clear();
}

bool ReferenceRecorder::feed(const float* samples, int n) {
  if (!recording_ || !samples || n <= 0) return !recording_ && !buf_.empty();
  const size_t room = target_samples_ > buf_.size()
                          ? target_samples_ - buf_.size()
                          : 0;
  const size_t take = std::min(room, size_t(n));
  buf_.insert(buf_.end(), samples, samples + take);
  if (buf_.size() >= target_samples_) {
    recording_ = false;
    return true;
  }
  return false;
}

float ReferenceRecorder::peak() const {
  float p = 0.f;
  for (float s : buf_) p = std::max(p, std::fabs(s));
  return p;
}

bool ReferenceRecorder::save(const std::string& path, std::string& err) const {
  if (buf_.empty()) {
    err = "no reference audio captured";
    return false;
  }
  return WriteWav16(path, buf_, sample_rate_, err);
}

// --- client ------------------------------------------------------------------

struct VoiceCloner::Impl {
  NSURLSession* session = nil;
  AVAudioPlayer* player = nil;
};

VoiceCloner::VoiceCloner() : impl_(new Impl()) {
  NSURLSessionConfiguration* cfg =
      [NSURLSessionConfiguration ephemeralSessionConfiguration];
  // Generous: a cold server loads model weights on the first request, and the
  // failure mode of a short timeout is an error message for a request that was
  // in fact about to succeed.
  cfg.timeoutIntervalForRequest = 180.0;
  cfg.timeoutIntervalForResource = 300.0;
  impl_->session = [NSURLSession sessionWithConfiguration:cfg];
}

VoiceCloner::~VoiceCloner() {
  stopPlayback();
  [impl_->session invalidateAndCancel];
}

void VoiceCloner::setEndpoint(const std::string& base_url) {
  base_url_ = base_url;
  // The old server's identity no longer describes the new endpoint.
  std::lock_guard<std::mutex> lk(mu_);
  info_ = ServerInfo{};
}

ServerInfo VoiceCloner::info() const {
  std::lock_guard<std::mutex> lk(mu_);
  return info_;
}

std::string VoiceCloner::errorText() const {
  std::lock_guard<std::mutex> lk(mu_);
  return error_text_;
}

std::string VoiceCloner::lastWavPath() const {
  std::lock_guard<std::mutex> lk(mu_);
  return last_wav_;
}

void VoiceCloner::probeAsync() {
  NSString* url = [NSString stringWithFormat:@"%s/health",
                                             base_url_.c_str()];
  NSURLRequest* req = [NSURLRequest
      requestWithURL:[NSURL URLWithString:url]
         cachePolicy:NSURLRequestReloadIgnoringLocalCacheData
     timeoutInterval:2.0];

  VoiceCloner* self_ptr = this;
  NSURLSessionDataTask* task = [impl_->session
      dataTaskWithRequest:req
        completionHandler:^(NSData* data, NSURLResponse* resp, NSError* err) {
          ServerInfo out;
          if (err) {
            out.detail = err.localizedDescription.UTF8String;
          } else if (data) {
            NSDictionary* j =
                [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
            if ([j isKindOfClass:[NSDictionary class]]) {
              out.reachable = true;
              id m = j[@"model"];
              if ([m isKindOfClass:[NSString class]]) {
                out.model = [m UTF8String];
              }
              id loaded = j[@"model_loaded"];
              out.model_loaded = [loaded respondsToSelector:@selector(boolValue)]
                                     ? [loaded boolValue]
                                     : false;
            } else {
              out.detail = "server replied with something that is not JSON";
            }
          }
          std::lock_guard<std::mutex> lk(self_ptr->mu_);
          self_ptr->info_ = out;
        }];
  [task resume];
}

bool VoiceCloner::speak(const SpeakParams& p) {
  if (p.text.empty()) return false;
  const State s = state_.load();
  if (s == State::kGenerating) return false;

  NSMutableDictionary* body = [NSMutableDictionary dictionary];
  body[@"text"] = [NSString stringWithUTF8String:p.text.c_str()];
  if (!p.reference_wav.empty()) {
    body[@"reference_wav"] =
        [NSString stringWithUTF8String:p.reference_wav.c_str()];
  }
  if (!p.reference_text.empty()) {
    body[@"reference_text"] =
        [NSString stringWithUTF8String:p.reference_text.c_str()];
  }
  if (!p.style.empty()) {
    body[@"style"] = [NSString stringWithUTF8String:p.style.c_str()];
  }
  body[@"exaggeration"] = @(p.exaggeration);
  body[@"cfg"] = @(p.cfg);
  body[@"temperature"] = @(p.temperature);

  NSError* jerr = nil;
  NSData* json = [NSJSONSerialization dataWithJSONObject:body
                                                 options:0
                                                   error:&jerr];
  if (!json) {
    std::lock_guard<std::mutex> lk(mu_);
    error_text_ = "could not encode the request";
    state_.store(State::kError);
    return false;
  }

  NSString* url = [NSString stringWithFormat:@"%s/speak", base_url_.c_str()];
  NSMutableURLRequest* req =
      [NSMutableURLRequest requestWithURL:[NSURL URLWithString:url]];
  req.HTTPMethod = @"POST";
  [req setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
  req.HTTPBody = json;

  state_.store(State::kGenerating);
  {
    std::lock_guard<std::mutex> lk(mu_);
    error_text_.clear();
  }

  const double t0 = CFAbsoluteTimeGetCurrent();
  VoiceCloner* self_ptr = this;
  NSURLSessionDataTask* task = [impl_->session
      dataTaskWithRequest:req
        completionHandler:^(NSData* data, NSURLResponse* resp, NSError* err) {
          const double elapsed = CFAbsoluteTimeGetCurrent() - t0;
          self_ptr->last_generate_s_.store(elapsed);

          auto fail = [self_ptr](const std::string& msg) {
            std::lock_guard<std::mutex> lk(self_ptr->mu_);
            self_ptr->error_text_ = msg;
            self_ptr->state_.store(State::kError);
          };

          if (err) {
            fail(std::string("request failed: ") +
                 err.localizedDescription.UTF8String +
                 "  (is tools/voice_server.py running?)");
            return;
          }
          NSHTTPURLResponse* http = (NSHTTPURLResponse*)resp;
          if (http.statusCode != 200) {
            // The server sends its own diagnostics as a JSON body; surfacing
            // that beats "HTTP 500".
            std::string detail;
            if (data.length) {
              NSDictionary* j = [NSJSONSerialization JSONObjectWithData:data
                                                                options:0
                                                                  error:nil];
              id e = [j isKindOfClass:[NSDictionary class]] ? j[@"error"] : nil;
              if ([e isKindOfClass:[NSString class]]) detail = [e UTF8String];
            }
            char msg[256];
            std::snprintf(msg, sizeof(msg), "server returned HTTP %ld%s%s",
                          (long)http.statusCode, detail.empty() ? "" : ": ",
                          detail.c_str());
            fail(msg);
            return;
          }
          if (!data.length) {
            fail("server returned an empty response");
            return;
          }

          // Keep the WAV on disk: it is the artefact worth auditioning, and
          // AVAudioPlayer is happiest with a file URL.
          NSString* path = [NSString
              stringWithFormat:@"%@/kv2_voice_%.0f.wav", NSTemporaryDirectory(),
                               CFAbsoluteTimeGetCurrent() * 1000.0];
          if (![data writeToFile:path atomically:YES]) {
            fail("could not write the generated audio to disk");
            return;
          }

          NSError* perr = nil;
          AVAudioPlayer* player = [[AVAudioPlayer alloc]
              initWithContentsOfURL:[NSURL fileURLWithPath:path]
                              error:&perr];
          if (!player) {
            fail(std::string("generated audio would not decode: ") +
                 (perr ? perr.localizedDescription.UTF8String : "unknown"));
            return;
          }
          self_ptr->last_audio_s_.store(player.duration);
          {
            std::lock_guard<std::mutex> lk(self_ptr->mu_);
            self_ptr->last_wav_ = path.UTF8String;
            self_ptr->impl_->player = player;
          }
          [player prepareToPlay];
          [player play];
          self_ptr->state_.store(State::kPlaying);
        }];
  [task resume];
  return true;
}

void VoiceCloner::stopPlayback() {
  std::lock_guard<std::mutex> lk(mu_);
  if (impl_->player) {
    [impl_->player stop];
    impl_->player = nil;
  }
  if (state_.load() == State::kPlaying) state_.store(State::kIdle);
}

bool VoiceCloner::playFile(const std::string& path, std::string& err) {
  NSString* p = [NSString stringWithUTF8String:path.c_str()];
  NSError* perr = nil;
  AVAudioPlayer* player = [[AVAudioPlayer alloc]
      initWithContentsOfURL:[NSURL fileURLWithPath:p]
                      error:&perr];
  if (!player) {
    err = perr ? perr.localizedDescription.UTF8String : "could not open file";
    return false;
  }
  std::lock_guard<std::mutex> lk(mu_);
  impl_->player = player;
  [player prepareToPlay];
  [player play];
  return true;
}

}  // namespace voice
