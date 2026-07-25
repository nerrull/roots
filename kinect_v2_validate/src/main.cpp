// Kinect v2 stream validator (macOS / libfreenect2)
//
// Opens the first Kinect v2 sensor and validates the RGB, depth, and IR
// streams: confirms frames arrive, reports resolution / format / measured FPS
// / dropped-frame counts, and sanity-checks depth values. Runs headless and
// prints a summary you can eyeball to confirm each stream is healthy.
//
// NOTE: This validator covers video + depth + IR only, because libfreenect2 has
// no audio support. The 4-mic array IS available on macOS, though — the OS
// exposes it as a CoreAudio device ("Xbox NUI Sensor", 4ch @ 16 kHz). See
// kinect_v2_demo, which scopes all 4 channels alongside the video streams.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Per-stream running stats.
struct StreamStat {
  const char* name;
  uint64_t frames = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t bpp = 0;         // bytes per pixel reported by the frame
  uint64_t last_seq = 0;    // libfreenect2 sequence counter (for drop detection)
  uint64_t drops = 0;       // gaps in the sequence counter
  bool seq_seen = false;
  Clock::time_point first{};
  Clock::time_point last{};

  void record(libfreenect2::Frame* f) {
    auto now = Clock::now();
    if (frames == 0) first = now;
    last = now;
    ++frames;
    width = f->width;
    height = f->height;
    bpp = f->bytes_per_pixel;
    // f->sequence is a monotonically increasing device frame counter; gaps
    // mean the host dropped packets/frames.
    if (seq_seen && f->sequence > last_seq + 1) {
      drops += (f->sequence - last_seq - 1);
    }
    last_seq = f->sequence;
    seq_seen = true;
  }

  double fps() const {
    if (frames < 2) return 0.0;
    double dt = std::chrono::duration<double>(last - first).count();
    return dt > 0 ? (frames - 1) / dt : 0.0;
  }
};

// Scan a depth frame (float32 mm) for a valid range, ignoring 0 = no-return.
void depth_range(libfreenect2::Frame* f, float& lo, float& hi, double& mean,
                 uint64_t& valid) {
  lo = 1e9f;
  hi = 0.0f;
  double sum = 0.0;
  valid = 0;
  const float* px = reinterpret_cast<const float*>(f->data);
  const size_t n = static_cast<size_t>(f->width) * f->height;
  for (size_t i = 0; i < n; ++i) {
    float d = px[i];
    if (d > 0.0f && d < 1e5f) {
      lo = d < lo ? d : lo;
      hi = d > hi ? d : hi;
      sum += d;
      ++valid;
    }
  }
  if (valid == 0) {
    lo = hi = 0.0f;
    mean = 0.0;
  } else {
    mean = sum / valid;
  }
}

}  // namespace

int main(int argc, char** argv) {
  double run_seconds = 10.0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if ((a == "-t" || a == "--seconds") && i + 1 < argc) {
      run_seconds = std::atof(argv[++i]);
    } else if (a == "-h" || a == "--help") {
      std::printf("usage: %s [-t seconds]\n", argv[0]);
      return 0;
    }
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  libfreenect2::setGlobalLogger(
      libfreenect2::createConsoleLogger(libfreenect2::Logger::Info));

  libfreenect2::Freenect2 freenect2;
  const int n = freenect2.enumerateDevices();
  std::printf("\n=== Kinect v2 stream validator ===\n");
  std::printf("devices found: %d\n", n);
  if (n == 0) {
    std::fprintf(stderr,
                 "No Kinect v2 detected.\n"
                 "  - Check the USB3 adapter/power brick is connected.\n"
                 "  - The sensor needs a genuine USB3 port.\n");
    return 1;
  }

  const std::string serial = freenect2.getDefaultDeviceSerialNumber();
  std::printf("default serial: %s\n", serial.c_str());

  // Prefer a GPU pipeline (OpenGL) for depth processing; fall back to CPU.
  libfreenect2::PacketPipeline* pipeline = nullptr;
#if defined(LIBFREENECT2_WITH_OPENGL_SUPPORT)
  pipeline = new libfreenect2::OpenGLPacketPipeline();
  std::printf("pipeline: OpenGL\n");
#else
  pipeline = new libfreenect2::CpuPacketPipeline();
  std::printf("pipeline: CPU\n");
#endif

  libfreenect2::Freenect2Device* dev = freenect2.openDevice(serial, pipeline);
  if (!dev) {
    std::fprintf(stderr, "openDevice failed for %s\n", serial.c_str());
    return 1;
  }

  const int types = libfreenect2::Frame::Color | libfreenect2::Frame::Ir |
                    libfreenect2::Frame::Depth;
  libfreenect2::SyncMultiFrameListener listener(types);
  dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);

  if (!dev->start()) {
    std::fprintf(stderr, "device start failed\n");
    return 1;
  }

  std::printf("firmware: %s\n", dev->getFirmwareVersion().c_str());
  std::printf("device serial: %s\n", dev->getSerialNumber().c_str());
  std::printf("\ncapturing for %.1fs (Ctrl-C to stop early)...\n\n",
              run_seconds);

  StreamStat rgb{"RGB  "}, ir{"IR   "}, depth{"DEPTH"};
  float depth_lo = 0, depth_hi = 0;
  double depth_mean = 0;
  uint64_t depth_valid = 0;

  auto t0 = Clock::now();
  libfreenect2::FrameMap frames;
  bool timed_out = false;

  while (!g_stop.load() && seconds_since(t0) < run_seconds) {
    // 10s wait so a stalled stream is visible rather than hanging forever.
    if (!listener.waitForNewFrame(frames, 10 * 1000)) {
      std::fprintf(stderr, "timeout waiting for frames — stream stalled\n");
      timed_out = true;
      break;
    }
    rgb.record(frames[libfreenect2::Frame::Color]);
    ir.record(frames[libfreenect2::Frame::Ir]);
    libfreenect2::Frame* d = frames[libfreenect2::Frame::Depth];
    depth.record(d);
    depth_range(d, depth_lo, depth_hi, depth_mean, depth_valid);
    listener.release(frames);
  }

  dev->stop();
  dev->close();

  auto report = [](const StreamStat& s) {
    std::printf("  %s  %4ux%-4u  %ubpp  frames=%-5llu  %.1f fps  drops=%llu\n",
                s.name, s.width, s.height, s.bpp,
                static_cast<unsigned long long>(s.frames), s.fps(),
                static_cast<unsigned long long>(s.drops));
  };

  std::printf("\n=== summary ===\n");
  report(rgb);
  report(ir);
  report(depth);
  std::printf("  depth valid px (last frame): %llu  range=[%.0f, %.0f] mm  "
              "mean=%.0f mm\n",
              static_cast<unsigned long long>(depth_valid), depth_lo, depth_hi,
              depth_mean);

  // Verdict: each expected stream must have delivered frames.
  bool ok = rgb.frames > 0 && ir.frames > 0 && depth.frames > 0 &&
            depth_valid > 0 && !timed_out;
  std::printf("\nRESULT: %s\n", ok ? "PASS — all streams healthy"
                                    : "FAIL — see missing/empty streams above");
  std::printf("(audio not covered here: 4-ch mic array is a CoreAudio device — "
              "see kinect_v2_demo)\n\n");
  return ok ? 0 : 2;
}
