// A block of memory obtained from Wwise, shaped so stmlib::BufferAllocator can
// carve it up.
//
// The MI modules never call malloc: on hardware they are handed a fixed region
// of SRAM/SDRAM and sub-allocate from it with stmlib::BufferAllocator. That
// maps cleanly onto Wwise's requirement that plug-ins allocate only through
// IAkPluginMemAlloc -- we take one allocation per plug-in instance and let the
// module divide it up exactly as it does on the module.

#ifndef MI_ARENA_H_
#define MI_ARENA_H_

#include <AK/SoundEngine/Common/IAkPlugin.h>
#include <AK/Tools/Common/AkAssert.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mi {

class Arena {
 public:
  Arena() : data_(nullptr), size_(0) {}
  ~Arena() { AKASSERT(data_ == nullptr); }

  AKRESULT Init(AK::IAkPluginMemAlloc* in_pAllocator, size_t in_uBytes) {
    data_ = (uint8_t*)AK_PLUGIN_ALLOC(in_pAllocator, in_uBytes);
    if (!data_) {
      return AK_InsufficientMemory;
    }
    size_ = in_uBytes;
    Clear();
    return AK_Success;
  }

  void Term(AK::IAkPluginMemAlloc* in_pAllocator) {
    if (data_) {
      AK_PLUGIN_FREE(in_pAllocator, data_);
      data_ = nullptr;
      size_ = 0;
    }
  }

  // The MI cores assume their buffers start zeroed; on hardware that is just
  // how the RAM comes up after a cold boot.
  void Clear() {
    if (data_) {
      memset(data_, 0, size_);
    }
  }

  inline uint8_t* data() const { return data_; }
  inline size_t size() const { return size_; }

 private:
  uint8_t* data_;
  size_t size_;
};

}  // namespace mi

#endif  // MI_ARENA_H_
