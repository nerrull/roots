// kinect_v2_demo — ImGui viewer for all three Kinect v2 stream families.
//
//   [ RGB 1920x1080 ]  [ depth 512x424 ]
//   [   4-channel mic-array scope       ]
//
// Layout is fixed (video left, depth right, audio underneath) with a controls
// strip; the poll rate of the colour and depth streams is adjustable at runtime.
//
// Notes on the choices here, since the point of this tool is stability:
//   * ImGui runs on GLFW + Metal (the repo convention), so we never share an
//     OpenGL context with libfreenect2's depth pipeline.
//   * Colour and depth use independent "latest wins" listeners -- see
//     kinect_source.h for why SyncMultiFrameListener is the wrong tool here.
//   * Each stream uploads into a rotating ring of Metal textures, so we never
//     overwrite a texture the GPU may still be sampling for the previous frame.
//   * A missing sensor or missing mic permission degrades to a warning in the
//     UI; it is never fatal.

#include "audio_capture.h"
#include "dsp.h"
#include "kinect_source.h"
#include "person_tracker.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_metal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

namespace {

constexpr int kTextureRing = 3;

// --- Metal image target ------------------------------------------------------

// A ring of same-sized textures. Uploading into slot N+1 while the GPU still
// reads slot N avoids a write-after-read hazard on shared-storage textures.
class GpuImage {
 public:
  void ensure(id<MTLDevice> device, int w, int h, MTLPixelFormat fmt) {
    if (w == w_ && h == h_ && fmt == fmt_ && texs_) return;
    w_ = w;
    h_ = h;
    fmt_ = fmt;
    texs_ = [NSMutableArray arrayWithCapacity:kTextureRing];
    MTLTextureDescriptor* d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:fmt
                                     width:(NSUInteger)w
                                    height:(NSUInteger)h
                                 mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    for (int i = 0; i < kTextureRing; ++i) {
      [texs_ addObject:[device newTextureWithDescriptor:d]];
    }
    idx_ = 0;
    current_ = texs_[0];
  }

  void upload(const void* bytes, int bytes_per_row) {
    if (!texs_) return;
    idx_ = (idx_ + 1) % kTextureRing;
    id<MTLTexture> tex = texs_[idx_];
    [tex replaceRegion:MTLRegionMake2D(0, 0, (NSUInteger)w_, (NSUInteger)h_)
           mipmapLevel:0
             withBytes:bytes
           bytesPerRow:(NSUInteger)bytes_per_row];
    current_ = tex;
  }

  bool valid() const { return current_ != nil; }
  int width() const { return w_; }
  int height() const { return h_; }
  ImTextureID texID() const {
    return (ImTextureID)(intptr_t)(__bridge void*)current_;
  }

 private:
  NSMutableArray* texs_ = nil;
  id<MTLTexture> current_ = nil;
  int idx_ = 0, w_ = 0, h_ = 0;
  MTLPixelFormat fmt_ = MTLPixelFormatInvalid;
};

// --- Depth colourisation -----------------------------------------------------

struct Rgb {
  float r, g, b;
};

// Piecewise-linear approximation of the Turbo colourmap. Perceptually far
// better than jet for reading depth gradients, and cheap enough to run on the
// CPU for 512x424 every frame.
Rgb Turbo(float t) {
  static const Rgb kStops[] = {
      {0.190f, 0.072f, 0.232f}, {0.276f, 0.435f, 0.996f},
      {0.098f, 0.831f, 0.812f}, {0.412f, 0.988f, 0.322f},
      {0.898f, 0.882f, 0.176f}, {0.984f, 0.502f, 0.098f},
      {0.729f, 0.113f, 0.019f}, {0.480f, 0.016f, 0.011f},
  };
  constexpr int kN = int(sizeof(kStops) / sizeof(kStops[0]));
  t = std::min(std::max(t, 0.f), 1.f) * (kN - 1);
  const int i = std::min(int(t), kN - 2);
  const float f = t - i;
  return {kStops[i].r + f * (kStops[i + 1].r - kStops[i].r),
          kStops[i].g + f * (kStops[i + 1].g - kStops[i].g),
          kStops[i].b + f * (kStops[i + 1].b - kStops[i].b)};
}

// Converts float-mm depth to RGBA8. Invalid samples (0, NaN, inf, out of the
// sensor's usable range) are painted a flat dark blue-grey so holes are
// visually distinct from "very near" or "very far".
void ColourizeDepth(const FrameSnapshot& f, float near_mm, float far_mm,
                    bool grayscale, std::vector<uint8_t>& rgba,
                    int* valid_px, float* min_mm, float* max_mm,
                    float* mean_mm) {
  const int n = f.width * f.height;
  rgba.resize(size_t(n) * 4);
  const float* src = reinterpret_cast<const float*>(f.data.data());
  const float span = (far_mm > near_mm) ? (far_mm - near_mm) : 1.f;

  int valid = 0;
  float lo = 0.f, hi = 0.f;
  double sum = 0.0;

  for (int i = 0; i < n; ++i) {
    const float mm = src[i];
    uint8_t* px = &rgba[size_t(i) * 4];
    if (!(mm > 0.f) || !std::isfinite(mm)) {
      px[0] = 24; px[1] = 26; px[2] = 34; px[3] = 255;
      continue;
    }
    if (valid == 0) { lo = hi = mm; }
    lo = std::min(lo, mm);
    hi = std::max(hi, mm);
    sum += mm;
    ++valid;

    const float t = std::min(std::max((mm - near_mm) / span, 0.f), 1.f);
    if (grayscale) {
      const uint8_t g = uint8_t((1.f - t) * 255.f);
      px[0] = px[1] = px[2] = g;
    } else {
      const Rgb c = Turbo(t);
      px[0] = uint8_t(c.r * 255.f);
      px[1] = uint8_t(c.g * 255.f);
      px[2] = uint8_t(c.b * 255.f);
    }
    px[3] = 255;
  }

  *valid_px = valid;
  *min_mm = lo;
  *max_mm = hi;
  *mean_mm = valid ? float(sum / valid) : 0.f;
}

// --- Small UI helpers --------------------------------------------------------

// Where an image ended up on screen, so callers can overlay on top of it.
struct ImagePlacement {
  bool drawn = false;
  ImVec2 origin{0, 0};  // screen position of the image's top-left
  float scale = 1.f;    // screen pixels per image pixel
};

// Draws `img` centred in the remaining content region, preserving aspect.
ImagePlacement DrawFittedImage(const GpuImage& img, const char* placeholder) {
  ImagePlacement out;
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  if (!img.valid() || avail.x <= 1 || avail.y <= 1) {
    const ImVec2 sz = ImGui::CalcTextSize(placeholder);
    ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + (avail.x - sz.x) * 0.5f,
                               ImGui::GetCursorPosY() + (avail.y - sz.y) * 0.5f));
    ImGui::TextDisabled("%s", placeholder);
    return out;
  }
  const float scale = std::min(avail.x / img.width(), avail.y / img.height());
  const ImVec2 draw(img.width() * scale, img.height() * scale);
  ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + (avail.x - draw.x) * 0.5f,
                             ImGui::GetCursorPosY() + (avail.y - draw.y) * 0.5f));
  out.origin = ImGui::GetCursorScreenPos();
  out.scale = scale;
  out.drawn = true;
  ImGui::Image(img.texID(), draw);
  return out;
}

// Horizontal gain-reduction meter: bar grows left-to-right as the compressor
// works harder.
void DrawGainReductionMeter(float gr_db, float range_db, ImVec2 size) {
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 p0 = ImGui::GetCursorScreenPos();
  const ImVec2 p1(p0.x + size.x, p0.y + size.y);
  dl->AddRectFilled(p0, p1, IM_COL32(24, 25, 32, 255), 2.f);
  const float t = std::min(std::max(-gr_db / std::max(range_db, 1.f), 0.f), 1.f);
  if (t > 0.f) {
    dl->AddRectFilled(p0, ImVec2(p0.x + size.x * t, p1.y),
                      IM_COL32(240, 170, 90, 255), 2.f);
  }
  char buf[32];
  std::snprintf(buf, sizeof(buf), "GR %.1f dB", gr_db);
  dl->AddText(ImVec2(p0.x + 5, p0.y + 1), IM_COL32(225, 228, 238, 255), buf);
  ImGui::Dummy(size);
}

// Half-polar plot of the SRP response over azimuth, with markers for the
// current beam direction and each locator's estimate.
void DrawDoaPlot(const dsp::SrpDoa& srp, bool have_response, float beam_deg,
                 bool mic_valid, float mic_deg, bool depth_valid,
                 float depth_deg, ImVec2 size) {
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 p0 = ImGui::GetCursorScreenPos();
  const ImVec2 p1(p0.x + size.x, p0.y + size.y);
  dl->AddRectFilled(p0, p1, IM_COL32(16, 17, 22, 255), 3.f);

  // Fan centred on the bottom edge: straight up is 0 deg, right is +90.
  const ImVec2 c(p0.x + size.x * 0.5f, p1.y - 6.f);
  const float radius = std::min(size.x * 0.5f - 6.f, size.y - 12.f);
  if (radius <= 4.f) {
    ImGui::Dummy(size);
    return;
  }

  auto dir = [&](float deg, float r) {
    const float a = deg * float(M_PI) / 180.f;
    return ImVec2(c.x + std::sin(a) * r, c.y - std::cos(a) * r);
  };

  // Graticule every 30 deg.
  for (int d = -90; d <= 90; d += 30) {
    dl->AddLine(c, dir(float(d), radius), IM_COL32(52, 56, 68, 255));
  }
  dl->AddLine(dir(-90.f, radius), dir(-90.f, radius), IM_COL32(52, 56, 68, 255));

  // Response surface.
  if (have_response && srp.angleCount() > 1) {
    const std::vector<float>& resp = srp.lastResponse();
    for (int i = 0; i + 1 < int(resp.size()); ++i) {
      const float r0 = 6.f + (radius - 6.f) * std::max(resp[size_t(i)], 0.f);
      const float r1 = 6.f + (radius - 6.f) * std::max(resp[size_t(i + 1)], 0.f);
      dl->AddLine(dir(srp.angleAt(i), r0), dir(srp.angleAt(i + 1), r1),
                  IM_COL32(90, 170, 255, 220), 1.5f);
    }
  }

  // Markers. Beam is what the audio is actually steered at.
  if (depth_valid) {
    dl->AddLine(c, dir(depth_deg, radius), IM_COL32(120, 240, 150, 255), 2.f);
  }
  if (mic_valid) {
    dl->AddLine(c, dir(mic_deg, radius), IM_COL32(255, 200, 110, 255), 2.f);
  }
  dl->AddLine(c, dir(beam_deg, radius), IM_COL32(255, 255, 255, 255), 2.5f);

  dl->AddText(ImVec2(p0.x + 5, p0.y + 2), IM_COL32(150, 155, 170, 255),
              "DOA  white=beam  green=depth  amber=mic");
  ImGui::Dummy(size);
}

// Min/max envelope plot: one vertical bar per pixel column spanning the
// loudest excursion in that column's samples. This is how audio scopes avoid
// aliasing when thousands of samples map onto a few hundred pixels.
void DrawWaveform(const std::vector<float>& s, ImVec2 size, float gain,
                  ImU32 colour, const char* label, float peak, float rms) {
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 p0 = ImGui::GetCursorScreenPos();
  const ImVec2 p1(p0.x + size.x, p0.y + size.y);

  dl->AddRectFilled(p0, p1, IM_COL32(16, 17, 22, 255), 3.f);
  const float mid = p0.y + size.y * 0.5f;
  dl->AddLine(ImVec2(p0.x, mid), ImVec2(p1.x, mid), IM_COL32(70, 74, 86, 255));

  const int cols = std::max(1, int(size.x));
  const int n = int(s.size());
  if (n > 0) {
    const float half = size.y * 0.5f - 1.f;
    for (int c = 0; c < cols; ++c) {
      const int a = int(int64_t(c) * n / cols);
      const int b = std::max(a + 1, int(int64_t(c + 1) * n / cols));
      float mn = s[a], mx = s[a];
      for (int i = a; i < b && i < n; ++i) {
        mn = std::min(mn, s[i]);
        mx = std::max(mx, s[i]);
      }
      const float y0 = mid - std::min(std::max(mx * gain, -1.f), 1.f) * half;
      const float y1 = mid - std::min(std::max(mn * gain, -1.f), 1.f) * half;
      const float x = p0.x + c + 0.5f;
      dl->AddLine(ImVec2(x, y0), ImVec2(x, std::max(y1, y0 + 1.f)), colour);
    }
  }

  char buf[128];
  std::snprintf(buf, sizeof(buf), "%s  pk %6.1f dB   rms %6.1f dB", label,
                peak > 0 ? 20.f * std::log10(peak) : -120.f,
                rms > 0 ? 20.f * std::log10(rms) : -120.f);
  dl->AddText(ImVec2(p0.x + 6, p0.y + 3), IM_COL32(190, 195, 210, 255), buf);
  ImGui::Dummy(size);
}

void glfw_error_callback(int error, const char* description) {
  std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

}  // namespace

int main(int argc, char** argv) {
  bool use_opengl = true;
  bool want_audio = true;
  KinectSource::UsbReset usb_reset = KinectSource::UsbReset::kReset;

  // Default: pin the Kinect v2 by its USB VID:PID and require the 4-mic array.
  // No name matching, and no falling back to the system default input.
  AudioSelector audio_sel;
  audio_sel.model_uid = kKinectV2ModelUid;
  audio_sel.min_input_channels = 4;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--cpu-pipeline") {
      use_opengl = false;
    } else if (a == "--no-audio") {
      want_audio = false;
    } else if (a == "--list-audio") {
      const std::vector<AudioDeviceInfo> devs = ListAudioInputDevices();
      std::printf("%zu audio input device(s):\n", devs.size());
      for (const AudioDeviceInfo& d : devs) {
        std::printf("\n  \"%s\"%s\n    %d ch @ %.0f Hz  %s\n"
                    "    uid   : %s\n    model : %s\n    maker : %s\n",
                    d.name.c_str(),
                    d.is_default_input ? "   [system default input]" : "",
                    d.input_channels, d.sample_rate, d.is_usb ? "USB" : "",
                    d.uid.c_str(), d.model_uid.c_str(),
                    d.manufacturer.c_str());
      }
      return 0;
    } else if (a == "--audio-uid" && i + 1 < argc) {
      // Exact, unambiguous: also disambiguates two identical sensors by serial.
      audio_sel = AudioSelector{};
      audio_sel.uid = argv[++i];
    } else if (a == "--audio-model" && i + 1 < argc) {
      audio_sel = AudioSelector{};
      audio_sel.model_uid = argv[++i];
    } else if (a == "--audio-name" && i + 1 < argc) {
      audio_sel = AudioSelector{};
      audio_sel.name = argv[++i];
    } else if (a == "--audio-allow-fallback") {
      audio_sel.allow_default_fallback = true;
    } else if (a == "--no-usb-reset") {
      usb_reset = KinectSource::UsbReset::kSkip;
    } else if (a == "-h" || a == "--help") {
      std::printf(
          "usage: %s [options]\n"
          "  --cpu-pipeline          decode depth on the CPU instead of OpenGL\n"
          "  --no-audio              skip mic-array capture\n"
          "  --list-audio            print all audio inputs with their UIDs\n"
          "  --no-usb-reset          skip the USB reset on open. The reset is\n"
          "                          on by default -- it clears a sensor left\n"
          "                          in a weird state, and audio still works\n"
          "                          because we wait for the mic array to\n"
          "                          re-attach afterwards\n"
          "\n"
          "Audio device selection (default: model UID \"%s\", the Kinect v2's\n"
          "USB VID:PID, requiring >=4 input channels). Each option replaces the\n"
          "default rather than adding to it:\n"
          "  --audio-uid  <uid>      exact device UID (see --list-audio)\n"
          "  --audio-model <substr>  substring of the model UID\n"
          "  --audio-name  <substr>  substring of the display name\n"
          "  --audio-allow-fallback  permit falling back to the system default\n"
          "                          input if the selector matches nothing\n"
          "                          (off by default: a silent fallback is how\n"
          "                          you end up recording the wrong mic)\n",
          argv[0], kKinectV2ModelUid);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s (try --help)\n", a.c_str());
      return 2;
    }
  }

  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) return 1;

  const float scale =
      ImGui_ImplGlfw_GetContentScaleForMonitor(glfwGetPrimaryMonitor());
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(int(1500 * scale), int(950 * scale),
                                        "Kinect v2 — streams", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return 1;
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  style.ScaleAllSizes(scale);
  style.FontScaleDpi = scale;

  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> queue = [device newCommandQueue];

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplMetal_Init(device);

  NSWindow* nswin = glfwGetCocoaWindow(window);
  CAMetalLayer* layer = [CAMetalLayer layer];
  layer.device = device;
  layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
  nswin.contentView.layer = layer;
  nswin.contentView.wantsLayer = YES;
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor new];

  // Sensor first, audio second -- deliberately in that order.
  //
  // Opening the sensor USB-resets it, which is what clears a sensor left in a
  // weird state by a previous run. The reset transiently detaches
  // AppleUSBAudio, so the mic array vanishes from CoreAudio and reappears once
  // the device re-enumerates (~0.1 s measured). Starting audio afterwards, with
  // a wait, therefore gets us both: a freshly reset sensor *and* working audio.
  // Doing it the other way round would have the reset kill an already-open
  // audio stream mid-flight.
  //
  // Opening after our GLFW window exists also keeps libfreenect2's OpenGL
  // pipeline (which creates a hidden GLFW window of its own) from interfering
  // with our window hints.
  KinectSource kinect;
  std::string kinect_err;
  const bool kinect_ok = kinect.open(use_opengl, usb_reset, kinect_err);
  if (!kinect_ok) std::fprintf(stderr, "kinect: %s\n", kinect_err.c_str());

  AudioCapture audio;
  std::string audio_err;
  bool audio_ok = false;
  if (want_audio) {
    // Wait for the audio interface to re-attach after the reset.
    audio_ok = audio.start(audio_sel, audio_err, /*wait_seconds=*/8.0);
    if (!audio_ok) std::fprintf(stderr, "audio: %s\n", audio_err.c_str());
  }

  // UI state.
  float color_hz = 30.f, depth_hz = 30.f;
  bool color_paused = false, depth_paused = false;
  bool unthrottled = false;
  float depth_near = 500.f, depth_far = 4500.f;
  bool depth_gray = false;
  float audio_gain = 8.f;
  float audio_window_ms = 250.f;
  bool show_all_channels = true;

  GpuImage color_img, depth_img;
  FrameSnapshot color_frame, depth_frame;
  std::vector<uint8_t> depth_rgba;
  int depth_valid = 0;
  float depth_min = 0, depth_max = 0, depth_mean = 0;
  double last_color_time = 0, last_depth_time = 0;

  // Hitch tracking: the whole point of "is it stable?".
  float worst_frame_ms = 0.f;
  int long_frames = 0;  // UI frames over 33 ms
  const double t_start = NowSeconds();

  std::vector<float> wave;

  // Audio-stall detection: the symptom of the sensor's audio interface being
  // detached is simply that framesCaptured() stops advancing.
  uint64_t last_audio_frames = 0;
  double last_audio_advance = NowSeconds();
  bool audio_stalled = false;

  // --- beam steering -------------------------------------------------------
  //
  // Two independent ways to find the talker, per the brief:
  //   kDepth -- nearest supported depth surface, i.e. whoever is closest.
  //   kMic   -- SRP over the 4 mics, i.e. whatever is loudest.
  // kDepthThenMic prefers depth (it works in silence) and falls back to the mic
  // estimate when the depth track is lost.
  enum class Steer { kManual, kDepth, kMic, kDepthThenMic };
  Steer steer_mode = Steer::kDepthThenMic;
  const char* kSteerNames[] = {"manual", "depth (closest person)",
                               "mic array (loudest)", "depth, mic fallback"};
  float manual_steer_deg = 0.f;

  PersonTracker tracker;
  PersonTracker::Params tracker_params;
  PersonTrack track;

  dsp::SrpDoa srp;
  dsp::DoaResult doa;
  float srp_aperture_cached = -1.f;
  float doa_smoothed_deg = 0.f;
  bool doa_have_smoothed = false;
  float doa_min_confidence = 0.35f;
  float doa_smoothing = 0.3f;
  double last_doa_time = 0;
  bool doa_has_response = false;
  // SRP is the only heavy CPU cost here, and 15 Hz is far faster than a person
  // moves; running it per UI frame would be pure waste.
  const double kDoaInterval = 1.0 / 15.0;
  const int kDoaSamples = 1536;
  const int kDoaSettle = 128;  // discard the band filters' startup transient
  std::vector<std::vector<float>> doa_bufs;

  // DSP panel state mirrors of the atomics, so the sliders have something to
  // bind to.
  bool dsp_enabled = true;
  float ui_input_gain_db = 12.f;
  float ui_highpass_hz = 110.f;
  float ui_comp_threshold_db = -34.f;
  float ui_comp_ratio = 4.f;
  float ui_comp_attack_ms = 5.f;
  float ui_comp_release_ms = 140.f;
  float ui_comp_knee_db = 8.f;
  float ui_comp_makeup_db = 8.f;
  bool ui_limiter = true;
  bool ui_beamform = true;
  float ui_aperture_m = 0.16f;
  bool show_beam = true;

  tracker.setIntrinsics(kinect.depthIntrinsics());

  while (!glfwWindowShouldClose(window)) {
    @autoreleasepool {
      glfwPollEvents();

      if (audio_ok) {
        const uint64_t af = audio.framesCaptured();
        if (af != last_audio_frames) {
          last_audio_frames = af;
          last_audio_advance = NowSeconds();
        }
        audio_stalled = (NowSeconds() - last_audio_advance) > 1.0;
      }

      // --- capture -----------------------------------------------------------
      kinect.setColorRate(unthrottled ? 0.f : color_hz);
      kinect.setDepthRate(unthrottled ? 0.f : depth_hz);
      kinect.setColorPaused(color_paused);
      kinect.setDepthPaused(depth_paused);

      if (kinect.pollColor(color_frame)) {
        const MTLPixelFormat fmt = (color_frame.format == libfreenect2::Frame::RGBX)
                                       ? MTLPixelFormatRGBA8Unorm
                                       : MTLPixelFormatBGRA8Unorm;
        color_img.ensure(device, color_frame.width, color_frame.height, fmt);
        color_img.upload(color_frame.data.data(),
                         color_frame.width * color_frame.bytes_per_pixel);
        last_color_time = NowSeconds();
      }
      if (kinect.pollDepth(depth_frame)) {
        ColourizeDepth(depth_frame, depth_near, depth_far, depth_gray,
                       depth_rgba, &depth_valid, &depth_min, &depth_max,
                       &depth_mean);
        depth_img.ensure(device, depth_frame.width, depth_frame.height,
                         MTLPixelFormatRGBA8Unorm);
        depth_img.upload(depth_rgba.data(), depth_frame.width * 4);
        last_depth_time = NowSeconds();

        // Locator 1: whoever is closest to the sensor.
        track = tracker.update(depth_frame, tracker_params, NowSeconds());
      }

      // --- locator 2: direction of the loudest source ------------------------
      if (audio_ok && audio.channels() >= 2 &&
          NowSeconds() - last_doa_time >= kDoaInterval) {
        last_doa_time = NowSeconds();
        const float aperture = ui_aperture_m;
        if (aperture != srp_aperture_cached) {
          srp.configure(dsp::UniformLinearArray(audio.channels(), aperture),
                        float(audio.sampleRate()), -90.f, 90.f, 3.f);
          srp_aperture_cached = aperture;
        }

        // Band-limit to the speech range, and below the array's spatial
        // aliasing limit (c/2d), or the response surface goes ambiguous.
        const dsp::BiquadCoeffs hp =
            dsp::HighpassCoeffs(float(audio.sampleRate()), 300.f);
        const dsp::BiquadCoeffs lp =
            dsp::LowpassCoeffs(float(audio.sampleRate()), 3000.f);

        doa_bufs.resize(size_t(audio.channels()));
        const float* ptrs[dsp::MicArray::kMaxMics];
        for (int c = 0; c < audio.channels(); ++c) {
          audio.snapshot(c, kDoaSamples, doa_bufs[size_t(c)]);
          dsp::Biquad f1, f2;
          for (float& s : doa_bufs[size_t(c)]) {
            s = f2.process(f1.process(s, hp), lp);
          }
          ptrs[c] = doa_bufs[size_t(c)].data() + kDoaSettle;
        }
        doa = srp.estimate(ptrs, audio.channels(), kDoaSamples - kDoaSettle);
        doa_has_response = doa.valid;

        if (doa.valid && doa.confidence >= doa_min_confidence) {
          const float a = std::min(std::max(doa_smoothing, 0.f), 1.f);
          doa_smoothed_deg =
              doa_have_smoothed
                  ? doa_smoothed_deg + a * (doa.azimuth_deg - doa_smoothed_deg)
                  : doa.azimuth_deg;
          doa_have_smoothed = true;
        }
      }

      // --- resolve the steering angle ---------------------------------------
      const bool depth_steer_ok = track.valid;
      const bool mic_steer_ok =
          doa_have_smoothed && doa.valid && doa.confidence >= doa_min_confidence;
      float beam_deg = manual_steer_deg;
      switch (steer_mode) {
        case Steer::kManual:
          beam_deg = manual_steer_deg;
          break;
        case Steer::kDepth:
          if (depth_steer_ok) beam_deg = track.azimuth_deg;
          break;
        case Steer::kMic:
          if (mic_steer_ok) beam_deg = doa_smoothed_deg;
          break;
        case Steer::kDepthThenMic:
          // Depth first: it still works when the room is silent.
          if (depth_steer_ok) {
            beam_deg = track.azimuth_deg;
          } else if (mic_steer_ok) {
            beam_deg = doa_smoothed_deg;
          }
          break;
      }
      beam_deg = std::min(std::max(beam_deg, -90.f), 90.f);

      if (audio_ok) {
        audio.controls.enabled.store(dsp_enabled);
        audio.controls.input_gain_db.store(ui_input_gain_db);
        audio.controls.highpass_hz.store(ui_highpass_hz);
        audio.controls.comp_threshold_db.store(ui_comp_threshold_db);
        audio.controls.comp_ratio.store(ui_comp_ratio);
        audio.controls.comp_attack_ms.store(ui_comp_attack_ms);
        audio.controls.comp_release_ms.store(ui_comp_release_ms);
        audio.controls.comp_knee_db.store(ui_comp_knee_db);
        audio.controls.comp_makeup_db.store(ui_comp_makeup_db);
        audio.controls.limiter.store(ui_limiter);
        audio.controls.beamform.store(ui_beamform);
        audio.controls.aperture_m.store(ui_aperture_m);
        audio.controls.steer_deg.store(beam_deg);
      }

      // --- frame setup -------------------------------------------------------
      int fb_w, fb_h;
      glfwGetFramebufferSize(window, &fb_w, &fb_h);
      layer.drawableSize = CGSizeMake(fb_w, fb_h);
      id<CAMetalDrawable> drawable = [layer nextDrawable];
      if (!drawable) continue;  // occluded / mid-resize: skip, don't stall

      id<MTLCommandBuffer> cmd = [queue commandBuffer];
      pass.colorAttachments[0].clearColor = MTLClearColorMake(0.06, 0.06, 0.08, 1.0);
      pass.colorAttachments[0].texture = drawable.texture;
      pass.colorAttachments[0].loadAction = MTLLoadActionClear;
      pass.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> enc =
          [cmd renderCommandEncoderWithDescriptor:pass];

      ImGui_ImplMetal_NewFrame(pass);
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      const float dt_ms = io.DeltaTime * 1000.f;
      worst_frame_ms = std::max(worst_frame_ms, dt_ms);
      if (dt_ms > 33.f) ++long_frames;

      // --- one fullscreen window holding the fixed layout --------------------
      const ImGuiViewport* vp = ImGui::GetMainViewport();
      ImGui::SetNextWindowPos(vp->WorkPos);
      ImGui::SetNextWindowSize(vp->WorkSize);
      ImGui::Begin("##root", nullptr,
                   ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                       ImGuiWindowFlags_NoBringToFrontOnFocus |
                       ImGuiWindowFlags_NoSavedSettings);

      // Controls strip.
      if (!kinect_ok) {
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 140, 120, 255));
        ImGui::TextWrapped("Kinect unavailable: %s", kinect_err.c_str());
        ImGui::PopStyleColor();
      } else {
        ImGui::Text("serial %s   firmware %s   depth pipeline: %s",
                    kinect.serial().c_str(), kinect.firmware().c_str(),
                    kinect.pipelineName().c_str());
      }
      ImGui::Text("UI %.1f fps (%.2f ms)   worst %.1f ms   frames >33ms: %d   "
                  "uptime %.0fs",
                  io.Framerate, dt_ms, worst_frame_ms, long_frames,
                  NowSeconds() - t_start);
      ImGui::SameLine();
      if (ImGui::SmallButton("reset")) {
        worst_frame_ms = 0.f;
        long_frames = 0;
      }

      ImGui::Separator();

      ImGui::PushItemWidth(220 * scale);
      ImGui::SliderFloat("video poll Hz", &color_hz, 1.f, 30.f, "%.1f");
      ImGui::SameLine();
      ImGui::Checkbox("pause##c", &color_paused);
      ImGui::SameLine(0, 24 * scale);
      ImGui::SliderFloat("depth poll Hz", &depth_hz, 1.f, 30.f, "%.1f");
      ImGui::SameLine();
      ImGui::Checkbox("pause##d", &depth_paused);
      ImGui::SameLine(0, 24 * scale);
      ImGui::Checkbox("unthrottled", &unthrottled);
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Poll every UI frame instead of at a fixed rate.\n"
            "The sensor always runs at its own native rate; these sliders only\n"
            "change how often the UI pulls the newest frame.");
      }

      ImGui::SliderFloat("depth near mm", &depth_near, 0.f, 3000.f, "%.0f");
      ImGui::SameLine(0, 24 * scale);
      ImGui::SliderFloat("depth far mm", &depth_far, 500.f, 8000.f, "%.0f");
      ImGui::SameLine(0, 24 * scale);
      ImGui::Checkbox("grayscale depth", &depth_gray);
      ImGui::PopItemWidth();

      ImGui::Separator();

      // --- layout: video | depth over waveform ------------------------------
      const float avail_h = ImGui::GetContentRegionAvail().y;
      // The DSP/steering column needs the room; the scopes stretch to match.
      const float audio_h = std::max(330.f * scale, avail_h * 0.44f);
      const float video_h = std::max(120.f * scale, avail_h - audio_h -
                                                        style.ItemSpacing.y);
      const float half_w =
          (ImGui::GetContentRegionAvail().x - style.ItemSpacing.x) * 0.5f;

      const StreamStats cs = kinect.colorStats();
      const StreamStats ds = kinect.depthStats();

      ImGui::BeginChild("##video", ImVec2(half_w, video_h), ImGuiChildFlags_Borders);
      ImGui::Text("VIDEO  %dx%d", color_frame.width, color_frame.height);
      ImGui::SameLine();
      ImGui::TextDisabled("| sensor %.1f Hz -> polled %.1f Hz | seq gaps %llu",
                          cs.delivered_hz, cs.polled_hz,
                          (unsigned long long)cs.seq_gaps);
      if (color_frame.valid) {
        ImGui::TextDisabled("exposure %.2f  gain %.2f  gamma %.2f  seq %u",
                            color_frame.exposure, color_frame.gain,
                            color_frame.gamma, color_frame.sequence);
        if (color_frame.exposure > 20.f) {
          ImGui::SameLine();
          ImGui::TextColored(ImVec4(1.f, 0.8f, 0.3f, 1.f), "(dim: 15 fps)");
          if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Under long auto-exposure the colour camera halves its rate to\n"
                "15 fps. Depth is unaffected and keeps running at 30 Hz.");
          }
        }
      }
      DrawFittedImage(color_img, kinect_ok ? "waiting for colour frames..."
                                          : "no sensor");
      ImGui::EndChild();

      ImGui::SameLine();

      ImGui::BeginChild("##depth", ImVec2(half_w, video_h), ImGuiChildFlags_Borders);
      ImGui::Text("DEPTH  %dx%d", depth_frame.width, depth_frame.height);
      ImGui::SameLine();
      ImGui::TextDisabled("| sensor %.1f Hz -> polled %.1f Hz | seq gaps %llu",
                          ds.delivered_hz, ds.polled_hz,
                          (unsigned long long)ds.seq_gaps);
      if (depth_frame.valid) {
        ImGui::TextDisabled("valid %d px  range [%.0f, %.0f] mm  mean %.0f mm",
                            depth_valid, depth_min, depth_max, depth_mean);
      }
      if (track.valid) {
        ImGui::TextColored(
            track.holding ? ImVec4(1.f, 0.85f, 0.4f, 1.f)
                          : ImVec4(0.5f, 1.f, 0.6f, 1.f),
            "person: %.2f m  az %+.1f deg  %d px%s", track.distance_mm * 0.001f,
            track.azimuth_deg, track.support_px,
            track.holding ? "  (holding)" : "");
      } else {
        ImGui::TextDisabled("person: none");
      }
      const ImagePlacement dp =
          DrawFittedImage(depth_img, kinect_ok ? "waiting for depth frames..."
                                               : "no sensor");
      // Crosshair on the tracked centroid, so the depth locator is verifiable
      // at a glance rather than only through the azimuth number.
      if (dp.drawn && track.valid && track.support_px > 0) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 c(dp.origin.x + float(track.centroid_u) * dp.scale,
                       dp.origin.y + float(track.centroid_v) * dp.scale);
        const ImU32 col = track.holding ? IM_COL32(255, 215, 100, 255)
                                        : IM_COL32(120, 255, 150, 255);
        const float r = 14.f * scale;
        dl->AddCircle(c, r, col, 0, 2.f);
        dl->AddLine(ImVec2(c.x - r * 1.6f, c.y), ImVec2(c.x - r * 0.5f, c.y), col, 2.f);
        dl->AddLine(ImVec2(c.x + r * 0.5f, c.y), ImVec2(c.x + r * 1.6f, c.y), col, 2.f);
        dl->AddLine(ImVec2(c.x, c.y - r * 1.6f), ImVec2(c.x, c.y - r * 0.5f), col, 2.f);
        dl->AddLine(ImVec2(c.x, c.y + r * 0.5f), ImVec2(c.x, c.y + r * 1.6f), col, 2.f);
      }
      ImGui::EndChild();

      // --- audio ------------------------------------------------------------
      ImGui::BeginChild("##audio", ImVec2(0, 0), ImGuiChildFlags_Borders);
      if (!want_audio) {
        ImGui::TextDisabled("audio disabled (--no-audio)");
      } else if (!audio_ok) {
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 140, 120, 255));
        ImGui::TextWrapped("Audio unavailable: %s", audio_err.c_str());
        ImGui::PopStyleColor();
      } else {
        const AudioDeviceInfo& ad = audio.device();
        // Show the bound device prominently: silently scoping the wrong
        // microphone is the failure mode worth making impossible to miss.
        const bool is_kinect =
            ad.model_uid.find(kKinectV2ModelUid) != std::string::npos;
        ImGui::Text("AUDIO");
        ImGui::SameLine();
        ImGui::TextColored(is_kinect ? ImVec4(0.6f, 1.f, 0.7f, 1.f)
                                     : ImVec4(1.f, 0.8f, 0.3f, 1.f),
                           "%s", ad.name.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("| %d ch @ %.0f Hz | %llu frames | cb errors %llu",
                            audio.channels(), audio.sampleRate(),
                            (unsigned long long)audio.framesCaptured(),
                            (unsigned long long)audio.callbackErrors());
        ImGui::TextDisabled("uid %s", ad.uid.c_str());
        if (!is_kinect) {
          ImGui::SameLine();
          ImGui::TextColored(ImVec4(1.f, 0.8f, 0.3f, 1.f),
                             "— NOT the Kinect mic array%s",
                             audio.usedFallback() ? " (default-input fallback)"
                                                  : "");
        }
        if (audio_stalled) {
          ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 140, 120, 255));
          ImGui::TextWrapped(
              "Audio stalled: no new frames for %.1fs (%llu render errors). "
              "Something detached the sensor's audio interface.",
              NowSeconds() - last_audio_advance,
              (unsigned long long)audio.callbackErrors());
          ImGui::PopStyleColor();
        }
        if (audio.looksLikeSilence()) {
          ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 200, 120, 255));
          ImGui::TextWrapped(
              "All channels are digital silence — this is usually a missing "
              "microphone permission for the parent terminal, not a dead "
              "sensor. System Settings > Privacy & Security > Microphone.");
          ImGui::PopStyleColor();
        }

        ImGui::PushItemWidth(160 * scale);
        ImGui::SliderFloat("scope zoom", &audio_gain, 1.f, 64.f, "%.1fx");
        ImGui::SameLine(0, 18 * scale);
        ImGui::SliderFloat("window ms", &audio_window_ms, 20.f, 2000.f, "%.0f");
        ImGui::SameLine(0, 18 * scale);
        ImGui::Checkbox("raw mics", &show_all_channels);
        ImGui::SameLine(0, 18 * scale);
        ImGui::Checkbox("beam out", &show_beam);
        ImGui::PopItemWidth();

        static const ImU32 kChColours[4] = {
            IM_COL32(120, 210, 255, 255), IM_COL32(150, 255, 170, 255),
            IM_COL32(255, 210, 120, 255), IM_COL32(255, 150, 200, 255)};

        const int win =
            std::max(16, int(audio_window_ms * 0.001 * audio.sampleRate()));

        // Scopes left, DSP + steering right.
        const float panel_w = std::min(360.f * scale,
                                       ImGui::GetContentRegionAvail().x * 0.42f);
        const float scopes_w =
            ImGui::GetContentRegionAvail().x - panel_w - style.ItemSpacing.x;

        ImGui::BeginChild("##scopes", ImVec2(scopes_w, 0));
        {
          const int raw_rows = show_all_channels ? audio.channels() : 0;
          const int rows = raw_rows + (show_beam ? 1 : 0);
          if (rows > 0) {
            const float total_h = ImGui::GetContentRegionAvail().y;
            // The beam is the output that matters, so give it a double-height row.
            const float units = float(raw_rows) + (show_beam ? 2.f : 0.f);
            const float unit_h =
                std::max(24.f * scale,
                         (total_h - style.ItemSpacing.y * float(rows - 1)) /
                             std::max(units, 1.f));
            const float w = ImGui::GetContentRegionAvail().x;

            if (show_beam) {
              audio.beamSnapshot(win, wave);
              float pk = 0.f, rms = 0.f;
              audio.beamLevels(win, &pk, &rms);
              char label[64];
              std::snprintf(label, sizeof(label), "BEAM %+.1f deg%s", beam_deg,
                            ui_beamform ? "" : " (bypassed: mic 1)");
              DrawWaveform(wave, ImVec2(w, unit_h * 2.f), audio_gain,
                           IM_COL32(255, 255, 255, 235), label, pk, rms);
            }
            for (int c = 0; c < raw_rows; ++c) {
              audio.snapshot(c, win, wave);
              float pk = 0.f, rms = 0.f;
              audio.levels(c, win, &pk, &rms);
              char label[16];
              std::snprintf(label, sizeof(label), "mic %d", c + 1);
              DrawWaveform(wave, ImVec2(w, unit_h), audio_gain,
                           kChColours[c % 4], label, pk, rms);
            }
          }
        }
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("##dsp", ImVec2(0, 0));
        {
          DrawGainReductionMeter(audio.gainReductionDb(), 24.f,
                                 ImVec2(ImGui::GetContentRegionAvail().x,
                                        16.f * scale));

          DrawDoaPlot(srp, doa_has_response, beam_deg, mic_steer_ok,
                      doa_smoothed_deg, depth_steer_ok, track.azimuth_deg,
                      ImVec2(ImGui::GetContentRegionAvail().x, 96.f * scale));

          ImGui::Text("steer: %s", kSteerNames[int(steer_mode)]);
          ImGui::PushItemWidth(-1);
          int mode_i = int(steer_mode);
          if (ImGui::Combo("##steermode", &mode_i,
                           "manual\0depth (closest)\0mic (loudest)\0"
                           "depth, mic fallback\0")) {
            steer_mode = Steer(mode_i);
          }
          ImGui::PopItemWidth();

          ImGui::PushItemWidth(120 * scale);
          if (steer_mode == Steer::kManual) {
            ImGui::SliderFloat("angle", &manual_steer_deg, -90.f, 90.f, "%+.0f");
          } else {
            ImGui::BeginDisabled();
            float shown_deg = beam_deg;
            ImGui::SliderFloat("angle", &shown_deg, -90.f, 90.f, "%+.1f");
            ImGui::EndDisabled();
          }

          ImGui::Text("depth %s %+.1f | mic %s %+.1f (c %.2f)",
                      depth_steer_ok ? "ok" : "--",
                      depth_steer_ok ? track.azimuth_deg : 0.f,
                      mic_steer_ok ? "ok" : "--",
                      doa_have_smoothed ? doa_smoothed_deg : 0.f,
                      doa.confidence);

          ImGui::Separator();
          ImGui::Checkbox("DSP", &dsp_enabled);
          ImGui::SameLine();
          ImGui::Checkbox("beamform", &ui_beamform);
          ImGui::SameLine();
          ImGui::Checkbox("limiter", &ui_limiter);

          ImGui::SliderFloat("in gain dB", &ui_input_gain_db, 0.f, 40.f, "%.1f");
          ImGui::SliderFloat("HPF Hz", &ui_highpass_hz, 20.f, 400.f, "%.0f");
          ImGui::SliderFloat("thresh dB", &ui_comp_threshold_db, -60.f, 0.f,
                             "%.1f");
          ImGui::SliderFloat("ratio", &ui_comp_ratio, 1.f, 20.f, "%.1f:1");
          ImGui::SliderFloat("attack ms", &ui_comp_attack_ms, 0.5f, 100.f,
                             "%.1f");
          ImGui::SliderFloat("release ms", &ui_comp_release_ms, 10.f, 1000.f,
                             "%.0f");
          ImGui::SliderFloat("knee dB", &ui_comp_knee_db, 0.f, 24.f, "%.1f");
          ImGui::SliderFloat("makeup dB", &ui_comp_makeup_db, 0.f, 30.f, "%.1f");

          ImGui::Separator();
          if (ImGui::SliderFloat("aperture m", &ui_aperture_m, 0.04f, 0.30f,
                                 "%.3f")) {
            // Re-derive the SRP grid on the next DOA pass.
            srp_aperture_cached = -1.f;
          }
          if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "End-to-end mic spacing. The Kinect v2's true geometry is not\n"
                "published, so this is an estimate -- calibrate it by standing\n"
                "at a known angle and matching 'mic' to 'depth' above.");
          }
          ImGui::SliderFloat("DOA conf min", &doa_min_confidence, 0.f, 1.f,
                             "%.2f");
          ImGui::Checkbox("mirror depth az", &tracker_params.mirror);
          if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Flip the sign of the depth-derived angle if the beam steers\n"
                "away from the person instead of toward them.");
          }
          ImGui::PopItemWidth();
          ImGui::TextDisabled("beam latency %.2f ms", audio.beamLatencyMs());
        }
        ImGui::EndChild();
      }
      ImGui::EndChild();

      // Stall warnings, once the streams have had a chance to start.
      const double now = NowSeconds();
      if (kinect_ok && now - t_start > 3.0) {
        if (!color_paused && last_color_time > 0 && now - last_color_time > 2.0) {
          ImGui::SetTooltip("colour stream stalled");
        }
        if (!depth_paused && last_depth_time > 0 && now - last_depth_time > 2.0) {
          ImGui::SetTooltip("depth stream stalled");
        }
      }

      ImGui::End();

      // --- render ------------------------------------------------------------
      ImGui::Render();
      ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), cmd, enc);
      [enc endEncoding];
      [cmd presentDrawable:drawable];
      [cmd commit];
    }
  }

  // Stop capture before tearing down the UI so no callback outlives its state.
  audio.stop();
  kinect.close();

  ImGui_ImplMetal_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
