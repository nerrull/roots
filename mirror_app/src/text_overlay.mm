#include "text_overlay.h"

#include "metal_context.h"
#include "text_sdf.h"

#import <CoreGraphics/CoreGraphics.h>
#import <CoreText/CoreText.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace mirror {
namespace {

// One laid-out line, kept alive until it has been drawn.
struct Line {
    CTLineRef line = nullptr;
    double width = 0, ascent = 0, descent = 0, leading = 0;
};

std::vector<std::string> SplitLines(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == '\n') {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    if (out.empty()) out.push_back("");
    return out;
}

CTLineRef MakeLine(const std::string& text, CTFontRef font, float tracking_em,
                   float font_size) {
    NSMutableDictionary* attrs = [NSMutableDictionary dictionary];
    attrs[(__bridge NSString*)kCTFontAttributeName] = (__bridge id)font;
    // CTKern is in points, and tracking reads more naturally as a fraction of
    // the em -- at a fixed em fraction the spacing looks the same at every
    // raster height, which a point value would not.
    if (tracking_em != 0.f) {
        attrs[(__bridge NSString*)kCTKernAttributeName] =
            @(double(tracking_em * font_size));
    }
    NSAttributedString* as =
        [[NSAttributedString alloc] initWithString:@(text.c_str())
                                        attributes:attrs];
    return CTLineCreateWithAttributedString((__bridge CFAttributedStringRef)as);
}

}  // namespace

TextOverlay::TextOverlay(const MetalContext& ctx) : ctx_(ctx) {}

void TextOverlay::update(const TextParams& p) {
    const int raster = std::max(p.raster_px, 16);
    if (p.text == built_text_ && p.font == built_font_ &&
        p.tracking == built_tracking_ && raster == built_raster_ && tex_ != nil) {
        return;
    }
    built_text_ = p.text;
    built_font_ = p.font;
    built_tracking_ = p.tracking;
    built_raster_ = raster;
    tex_ = nil;
    fw_ = fh_ = 0;

    if (p.text.empty()) return;

    const CGFloat font_size = raster;
    CTFontRef font = CTFontCreateWithName(
        (__bridge CFStringRef)@(p.font.c_str()), font_size, nullptr);
    if (!font) {
        // An unknown font name is a typo in the UI, not a reason to have no
        // text: fall back rather than leaving the overlay silently empty.
        font = CTFontCreateWithName(CFSTR("Helvetica-Bold"), font_size, nullptr);
    }
    if (!font) return;

    std::vector<std::string> strs = SplitLines(p.text);
    std::vector<Line> lines;
    lines.reserve(strs.size());
    double max_w = 0;
    for (const std::string& s : strs) {
        Line L;
        L.line = MakeLine(s, font, p.tracking, float(font_size));
        if (!L.line) continue;
        L.width = CTLineGetTypographicBounds(L.line, &L.ascent, &L.descent,
                                             &L.leading);
        max_w = std::max(max_w, L.width);
        lines.push_back(L);
    }

    // Line height from the font's metrics rather than per-line bounds, so lines
    // of differing ascender content still sit on an even baseline grid.
    const double line_h = CTFontGetAscent(font) + CTFontGetDescent(font) +
                          CTFontGetLeading(font);
    const double text_h = line_h * double(lines.size());

    // The field needs room around the glyphs for the distance to ramp out; the
    // spread scales with the raster height so the ramp is the same fraction of
    // a stroke at every quality setting.
    //
    // 0.15 rather than something tighter because the encoded band is what the
    // shader's soft edge has to fit inside: at a bold weight this is roughly one
    // stem width, which is as far as blurring an edge stays meaningful anyway.
    // It costs only padding.
    const float spread = std::max(2.f, float(raster) * 0.15f);
    const int pad = int(std::ceil(spread)) + 2;

    const int w = int(std::ceil(max_w)) + 2 * pad;
    const int h = int(std::ceil(text_h)) + 2 * pad;
    if (w <= 0 || h <= 0 || lines.empty()) {
        for (Line& L : lines) if (L.line) CFRelease(L.line);
        CFRelease(font);
        return;
    }

    // Alpha-only context: the glyph coverage *is* the byte, which is exactly
    // what the distance transform wants, with no colour space in the way.
    std::vector<unsigned char> cov(size_t(w) * h, 0);
    CGContextRef cg = CGBitmapContextCreate(cov.data(), w, h, 8, w, nullptr,
                                            kCGImageAlphaOnly);
    if (cg) {
        CGContextSetShouldAntialias(cg, true);
        CGContextSetShouldSmoothFonts(cg, false);  // no subpixel: this is a mask
        CGContextSetAlpha(cg, 1.0);
        for (size_t i = 0; i < lines.size(); ++i) {
            const Line& L = lines[i];
            // CG's origin is bottom-left; lay the lines out top-down.
            const double baseline =
                double(h) - pad - line_h * double(i + 1) + CTFontGetDescent(font);
            CGContextSetTextPosition(cg, (double(w) - L.width) * 0.5, baseline);
            CTLineDraw(L.line, cg);
        }
        CGContextRelease(cg);
    }
    for (Line& L : lines) if (L.line) CFRelease(L.line);
    CFRelease(font);

    SdfImage sdf = BuildSDF(cov.data(), w, h, spread);
    if (sdf.px.empty()) return;

    MTLTextureDescriptor* d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                     width:w
                                    height:h
                                 mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    tex_ = [ctx_.device() newTextureWithDescriptor:d];
    if (!tex_) return;
    [tex_ replaceRegion:MTLRegionMake2D(0, 0, w, h)
            mipmapLevel:0
              withBytes:sdf.px.data()
            bytesPerRow:w];

    fw_ = w;
    fh_ = h;
    spread_ = sdf.spread;
    pad_ratio_ = text_h > 0 ? float(double(h) / text_h) : 1.f;
    aspect_ratio_ = float(double(w) / double(h));
}

TextUniforms TextOverlay::uniforms(const TextParams& p, float aspect,
                                   const TextRipple& r, double time) const {
    TextUniforms u;
    const bool live = p.on && tex_ != nil && p.strength > 0.f && p.reveal > 0.f;
    u.cnt.y = live ? 1.f : 0.f;
    if (!live) return u;

    // p.size is the half-height of the *text*; the bitmap is taller by the
    // padding, and the placement extent has to cover the bitmap.
    const float hy = p.size * pad_ratio_;
    const float hx = hy * aspect_ratio_;
    u.place = {p.cx, p.cy, hx, hy};

    // The dilate slider is in coord units, which is the only framing that stays
    // meaningful as `size` changes; the shader compares against the encoded
    // distance, so convert through the bitmap's pixels-per-coord-unit.
    const float px_per_coord = hy > 0 ? float(fh_) / (2.f * hy) : 0.f;
    const float dilate_encoded = p.dilate * px_per_coord / (2.f * spread_);

    u.tune  = {aspect, p.strength, std::max(p.softness, 0.f), dilate_encoded};
    u.tune2 = {p.warp, r.k, r.decay, r.core_r2};

    u.diss = {std::clamp(p.turbulence, 0.f, 1.f), std::max(p.turb_scale, 0.f),
              p.turb_speed, 0.f};
    // Wrapped: the drift is a noise offset, so an installation left running for
    // hours would otherwise walk it out to where float spacing is coarser than
    // the noise itself and the turbulence visibly freezes.
    u.cnt.z = float(std::fmod(time, 4096.0));
    u.cnt.w = std::clamp(p.reveal, 0.f, 1.f);

    const int n = std::min(r.n, 16);
    u.cnt.x = float(n);
    for (int i = 0; i < n; ++i) {
        u.src[i] = {r.src[i][0], r.src[i][1], r.src[i][2], r.src[i][3]};
    }
    return u;
}

}  // namespace mirror
