#include "metal_context.h"
#import <Foundation/Foundation.h>
#include <cstdio>
#include <fstream>
#include <sstream>

MetalContext::MetalContext() {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) { fprintf(stderr, "MetalContext: no Metal device\n"); return; }
    queue_ = [device_ newCommandQueue];
}

id<MTLLibrary> MetalContext::newLibraryFromFile(const std::string& path) const {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "MetalContext: cannot open MSL '%s'\n", path.c_str()); return nil; }
    std::stringstream ss; ss << f.rdbuf();
    return newLibraryFromSource(ss.str());
}

id<MTLLibrary> MetalContext::newLibraryFromSource(const std::string& src) const {
    NSError* err = nil;
    id<MTLLibrary> lib =
        [device_ newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                              options:nil error:&err];
    if (!lib) { NSLog(@"MetalContext: MSL compile failed: %@", err); }
    return lib;
}
