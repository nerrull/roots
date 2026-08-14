// Adapts Wwise's variable-size audio buffers to the fixed block size that the
// Mutable Instruments DSP cores require.
//
// The MI modules were written for a hardware audio loop that hands them a
// constant number of frames every time (24 for Rings, 12 for Plaits, 32 for
// Clouds, 16 for Elements). Wwise hands a plug-in whatever uValidFrames
// happens to be, which is usually 512 but is not guaranteed to be a multiple
// of anything. This adapter absorbs the mismatch by buffering one block of
// input and replaying the previously rendered block on the way out, which
// costs exactly kBlock samples of latency and nothing else.

#ifndef MI_BLOCK_ADAPTER_H_
#define MI_BLOCK_ADAPTER_H_

#include <cstddef>
#include <cstring>

namespace mi {

// kBlock: frames per MI render call. kOutChannels: how many output streams the
// module produces (Rings and Elements produce 2 -- 'out' and 'aux').
template <size_t kBlock, size_t kOutChannels>
class BlockAdapter {
 public:
  BlockAdapter() { Reset(); }

  void Reset() {
    pos_ = 0;
    memset(in_, 0, sizeof(in_));
    memset(out_, 0, sizeof(out_));
  }

  // Feeds one input sample and retrieves one sample per output channel.
  //
  // render is invoked as render(const float* in, float* const* out, size_t n)
  // once every kBlock calls. The samples it writes are read back out over the
  // *following* kBlock calls, which is where the latency comes from.
  template <typename Renderer>
  inline void Tick(float in, float* out, Renderer&& render) {
    in_[pos_] = in;
    for (size_t c = 0; c < kOutChannels; ++c) {
      out[c] = out_[c][pos_];
    }
    if (++pos_ == kBlock) {
      pos_ = 0;
      float* channels[kOutChannels];
      for (size_t c = 0; c < kOutChannels; ++c) {
        channels[c] = out_[c];
      }
      render(in_, channels, kBlock);
    }
  }

  static const size_t kLatencySamples = kBlock;

 private:
  float in_[kBlock];
  float out_[kOutChannels][kBlock];
  size_t pos_;
};

}  // namespace mi

#endif  // MI_BLOCK_ADAPTER_H_
