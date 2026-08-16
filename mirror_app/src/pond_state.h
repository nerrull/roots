// pond_state — full C++/MLX port of mlx_fused_mlp/demo_panel.py's PondState.
//
// All the live-tunable "pond" parameters plus the render pipeline: ripple-source
// build, weight shaping (detail / gain-tilt / gauss↔uniform / contrast / reseed),
// colour mixing, and the mask-emergence (hydro-dip) transition. Pure C++/MLX; the
// ObjC++ MirrorScene owns a Pond and uploads its output to a Metal texture.
#pragma once

#include <mlx/mlx.h>
#include <array>
#include <optional>
#include <vector>

#include "drop_spawner.h"
#include "mlp_forward.h"
#include "mirror_render.h"
#include "mirror_train.h"

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
    // Raindrops on / off. Everything about *how* they fall is in `spawn`; this
    // stays a plain bool so the rest of the app (and the presets) keep a single
    // switch for "is there rain".
    //
    // Off by default: ripples are a decorative field that dominates the MLP's
    // input features, which is wrong once the network is being *fitted* to
    // something -- during training they are signal the target does not contain.
    // Turn them on for the standalone pond look.
    bool  drops_on = false;
    DropSpawnParams spawn;
    bool  orbit_on = false;
    bool  core_rolloff = true;
    float core_radius = 0.12f;
    // --- hybrid sine/tanh --------------------------------------------------
    // sine_layers > 0 puts SIREN sine activations on that many leading hidden
    // layers, with tanh behind them. The two scales do different jobs and are
    // deliberately separate knobs:
    //   sine_w0     -- how many regions the field breaks into (composition)
    //   detail      -- how hard the boundaries between them are (articulation)
    // Measured: at sine_w0 = 5 the frame stays ~49% flat as detail goes 0.8 ->
    // 4.0 while mean gradient rises 7x, i.e. large open areas survive sharper
    // edges. By sine_w0 = 40 that decoupling is gone (flatness falls to 30%)
    // and the field is uniform texture with no background left.
    int   sine_layers = 0;           // 0 = the original all-tanh network
    float sine_w0 = 6.0f;

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
    // --- the fit region, and what it holds still ---------------------------
    //
    // A live fit is a person the network is reproducing surrounded by a field
    // nothing constrains. These let the two behave differently in one frame:
    // inside the region every input is frozen at what the fit was trained on,
    // outside it the latent is free to animate and the colour to drain away.
    // Without the region the whole canvas has to choose one or the other.
    FitRegion region;
    // The region's distance field, when region.use_field is set: fw*fh floats,
    // distance in coord units outward from the trained shape (see
    // DistanceOutside). Held here rather than pushed through a setter because
    // it changes every frame, exactly like every other field in this struct.
    std::vector<float> region_field;
    // Animate z outside the region. Inside, z is pinned to the value captured
    // at beginFit -- z is an MLP *input*, so moving it under a fitted network
    // is not a colour tweak, it is asking a different question of the same
    // weights and the face comes apart.
    bool  z_free_outside = false;
    // Drain colour outside the region: 0 keeps the frame as it is, 1 takes the
    // outside to greyscale. The transition rides the same weight as the latent,
    // so the two edges cannot disagree.
    float grey_outside = 0.0f;
    // Shift the MLP's input coordinates. The head-stabilised fit mode moves
    // this with the subject so the network always sees them in the same place;
    // colour travel adds to it. Applied to render and fit alike -- the two
    // disagreeing would mean training one function and drawing another.
    float coord_off_x = 0.0f, coord_off_y = 0.0f;
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

    // --- live fitting ------------------------------------------------------
    //
    // Once a target is set and training starts, render() draws the *trained*
    // weights instead of the shaped/scaled base ones -- detail/contrast/tilt
    // stop applying, because the network no longer derives from them. Stopping
    // does not revert: the fitted weights stay until reseed() or a new fit.
    //
    // `rgb` is h*w*3 floats in [0,1]. The fit resolution is whatever the target
    // was given at, independent of the display resolution.
    // `mask` is optional, h*w bytes, non-zero = train on this pixel. Empty
    // fits the whole frame. With a mask, the training pass runs on ONLY those
    // pixels (they are gathered into a compact batch), so cost scales with the
    // masked area -- and the network is left unconstrained everywhere else,
    // which is what keeps a background generative while a subject is fitted.
    void  beginFit(const std::vector<float>& rgb, int h, int w, const PondParams& p,
                   const std::vector<unsigned char>& mask = {});
    // Swap the target WITHOUT touching the weights or the optimiser state.
    // This is the live-feed path: beginFit() resets Adam, so calling it per
    // frame would discard the momentum every step and the fit would never
    // build up enough velocity to follow anything moving. Measured with the
    // reset in place, a continuously moving target tracked at worst-MSE 0.142;
    // keeping the state it is 0.00021 -- a 677x difference.
    void  updateFitTarget(const std::vector<float>& rgb, int h, int w,
                          const std::vector<unsigned char>& mask = {});
    // Pixels the last target selected (== fit grid area when unmasked).
    int   fitPixels() const { return trainer_.trainedPixels(); }
    // One optimiser step. Returns MSE before the update, or -1 if not fitting.
    float fitStep(float lr, const PondParams& p);
    void  stopFit()  { fitting_ = false; }
    void  clearFit() { fitting_ = false; trainer_ = MlpTrainer(); }
    bool  fitting()  const { return fitting_; }
    bool  fitted()   const { return trainer_.ready(); }
    int   fitSteps() const { return trainer_.steps(); }
    // The z phase the fit was begun at. render() pins the region's interior to
    // this, so `z` is free to animate everywhere else.
    float fitZ()     const { return fit_z_; }

    // Spawn a drop right now: an audio onset, a MIDI hit, a button. `strength`
    // is 0..1 (negative means "you choose"), `pan` -1..1 across the frame. Takes
    // effect on the next render(), which is where drops learn what time it is.
    void triggerDrop(float strength = -1.f, float pan = 0.f) {
        spawner_.trigger(strength, pan);
    }
    const DropSpawner& spawner() const { return spawner_; }

    // The ripple sources the last render() actually used.
    //
    // Cached rather than recomputed by the caller from the clock: the text
    // overlay refracts through this same field in the present pass, and a
    // second evaluation would be a frame's phase out of step with the image it
    // is drawn over -- close enough to look right when still and to shear
    // visibly the moment the ripples move.
    const std::vector<RippleSource>& lastSources() const { return last_src_; }

private:
    static mx::array make_weights(const MLPConfig& cfg, int seed, float scale = 1.0f);
    std::vector<float> layer_scales(const PondParams& p) const;
    const mx::array& shaped_base(const PondParams& p);
    const mx::array& weights(const PondParams& p);
    std::vector<RippleSource> sources(float asp, double t, const PondParams& p);
    mx::array apply_transition(const mx::array& img, const mx::array& coords,
                               int lh, int lw, float asp, const PondParams& p) const;

    MLPConfig cfg_;
    int seed_;
    DropSpawner spawner_;
    mx::array wb_;                  // base (scale-1) weights, fp16

    // caches keyed on their inputs (mirrors PondState)
    mx::array wb_shaped_;
    std::optional<float> shaped_key_;         // uniform_mix (+ seed via reseed)
    mx::array w_;
    std::optional<std::array<float, 7>> w_key_;   // detail,tilt,umix,contrast,seed,
                                                  // sine_layers,sine_w0

    // The coordinate grid depends only on the render size, so rebuilding it per
    // frame was 0.65 ms of pure waste (linspace + meshgrid + stack + cast over
    // 518k points). Keyed on (lh, lw) since asp is derived from them.
    mx::array coords_;
    std::optional<std::array<int, 2>> coords_key_;

    MlpTrainer trainer_;
    bool fitting_ = false;
    // The fit grid is its own size, so it needs its own cached coordinate
    // features -- rebuilding them per step was the single largest avoidable
    // cost in the render loop before it was cached.
    mx::array fit_feats_ = mx::zeros({1});
    std::optional<std::array<int, 2>> fit_feats_key_;
    // Features are gathered to match a masked target. Keyed on the pixel count
    // as well as the grid, so a mask that changes shape rebuilds them.
    int fit_feats_px_ = -1;
    // ...and on the input shift and the latent, which are what the features
    // are *of*. Without these in the key, moving either would leave training
    // running against the features built for the old ones -- silently, since
    // nothing else notices.
    std::array<float, 3> fit_feats_in_{0.f, 0.f, 0.f};
    // The latent the fit was begun at, held for as long as the fit lives.
    float fit_z_ = 0.f;
    std::vector<RippleSource> last_src_;
    void rebuildFitFeatures(const PondParams& p);
    const mx::array& coord_grid(int lh, int lw, float asp);

    // Changing the split changes both the kernel (a template parameter) and
    // the base weights (sine layers are SIREN-initialised), so it is tracked
    // and triggers a rebuild rather than being read per frame.
    void setSplit(int split);
};

}  // namespace mirror
