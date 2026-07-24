// pond_state — full C++/MLX port of mlx_fused_mlp/demo_panel.py's PondState.
//
// All the live-tunable "pond" parameters plus the render pipeline: ripple-source
// build, weight shaping (detail / gain-tilt / gauss↔uniform / contrast / reseed),
// colour mixing, and the mask-emergence (hydro-dip) transition. Pure C++/MLX; the
// ObjC++ MirrorScene owns a Pond and uploads its output to a Metal texture.
#pragma once

#include <mlx/mlx.h>
#include <optional>
#include <vector>

#include "mlp_forward.h"
#include "mirror_render.h"

namespace mirror {

namespace mx = mlx::core;

// Every ImGui-bound knob (defaults match PondState.__init__).
struct PondParams {
    // ripples
    float ring_freq = 3.0f;
    float decay = 1.8f;
    float speed = 1.2f;
    float ripple_offset = 0.0f;      // manual phase added to the time phase
    bool  color_travel = false;      // palette follows the orbit source
    float warp = 0.0f;               // refraction: ripple gradient warps colour coords
    int   drops = 4;
    bool  orbit_on = true;
    bool  core_rolloff = true;
    float core_radius = 0.12f;
    // weight shaping
    float detail = 3.0f;             // hidden-layer scale
    float gain_tilt = 0.0f;          // depth gain profile
    float uniform_mix = 0.0f;        // 0 = Gaussian, 1 = uniform weights
    float contrast = 6.0f;           // output-layer scale
    // tone / colour
    bool  srgb_fix = false;
    float gamma = 1.0f;
    float color_mix = 1.0f;          // 0 = greyscale, 1 = full RGB
    int   grey_channel = 0;          // 0/1/2 = R/G/B
    bool  amp_drives_color = false;
    float amp_gain = 1.5f;
    bool  swap_rb = false;
    // time
    bool  paused = false;
    float time_scale = 1.0f;
    // z latent
    float z = 0.0f;                  // phase; fed as z_amp*sin(z), z_amp*cos(z)
    float z_amp = 1.0f;
    float z_rate = 0.0f;             // auto-advance /s
    float z_step = 0.10f;
    // mask emergence transition
    float transition = 0.0f;
    bool  trans_auto = false;
    float trans_rate = 0.25f;
    float relief_h = 0.6f;
    float mask_ax = 0.55f;
    float mask_ay = 0.72f;
    float light_az = 0.7f;
    float light_elev = 0.8f;
    float ambient = 0.35f;
    float diff_amt = 0.9f;
    float spec_amt = 0.5f;
    float shininess = 24.0f;
    float bg_dim = 0.5f;
};

class Pond {
public:
    explicit Pond(int seed = 11);

    // Render the LOW-RES (lh, lw, 3) fp32 image in [0,1] for time t (ripple clock).
    mx::array render(int lh, int lw, double t, const PondParams& p);

    void reseed();                 // seed += 1, rebuild base weights
    int  seed() const { return seed_; }
    const MLPConfig& config() const { return cfg_; }

private:
    static mx::array make_weights(const MLPConfig& cfg, int seed, float scale = 1.0f);
    std::vector<float> layer_scales(const PondParams& p) const;
    const mx::array& shaped_base(const PondParams& p);
    const mx::array& weights(const PondParams& p);
    std::vector<RippleSource> sources(float asp, double t, const PondParams& p) const;
    mx::array apply_transition(const mx::array& img, const mx::array& coords,
                               int lh, int lw, float asp, const PondParams& p) const;

    MLPConfig cfg_;
    int seed_;
    mx::array wb_;                  // base (scale-1) weights, fp16

    // caches keyed on their inputs (mirrors PondState)
    mx::array wb_shaped_;
    std::optional<float> shaped_key_;         // uniform_mix (+ seed via reseed)
    mx::array w_;
    std::optional<std::array<float, 5>> w_key_;   // detail,tilt,umix,contrast,seed
};

}  // namespace mirror
