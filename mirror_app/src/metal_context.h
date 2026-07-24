// MetalContext — the single Metal device/queue the whole app shares.
//
// Deliberately the *system default* device (MTLCreateSystemDefaultDevice), which
// is also the device MLX allocates its arrays on. Sharing one device is what lets
// the mirror path stay zero-copy: an mx::array's backing MTLBuffer can be wrapped
// as an MTLTexture without a round-trip. Every scene and the compositor take a
// reference to this one context.
//
// ObjC++ only (holds id<MTL...> handles) — include from .mm translation units.
#pragma once

#ifndef __OBJC__
#error "metal_context.h is ObjC++ only; include it from a .mm file"
#endif

#import <Metal/Metal.h>
#include <string>
#include <vector>

class MetalContext {
public:
    MetalContext();

    id<MTLDevice>       device() const { return device_; }
    id<MTLCommandQueue> queue()  const { return queue_; }

    // Compile an MSL source file into a library at runtime (the reactor_cpp
    // pattern: shaders are the single source of truth on disk, recompiled each
    // launch so edits need no rebuild). Logs and returns nil on failure.
    id<MTLLibrary> newLibraryFromFile(const std::string& path) const;

    // Compile MSL from an in-memory source string (used by the MLP/feature port,
    // whose MSL is reused verbatim from the Python kernel strings).
    id<MTLLibrary> newLibraryFromSource(const std::string& src) const;

    // Compile MSL assembled from several files concatenated in order. The
    // runtime source-string compiler has no #include search path, so shared
    // headers (e.g. root_shared.h) are prepended as plain text this way rather
    // than #included. Missing files log and yield nil.
    id<MTLLibrary> newLibraryFromFiles(const std::vector<std::string>& paths) const;

private:
    id<MTLDevice>       device_ = nil;
    id<MTLCommandQueue> queue_  = nil;
};
