// ui_params — every control in the app, by name, in one place.
//
// Three features want the same thing and would otherwise each invent it:
//
//   MIDI          a knob has to reach a specific parameter, which means the
//                 parameter needs a name that survives being written to disk.
//   presets       saving "the settings" means enumerating them, which an
//                 immediate-mode UI does not otherwise let you do.
//   the panel     grouping controls into sections is exactly the same tree the
//                 other two need.
//
// So the ImGui calls are wrapped: each control declares itself as it is drawn,
// under the section it is drawn in, and that declaration is the registry. There
// is no second list to keep in step with the panel, which is the failure mode a
// hand-maintained parameter table always ends in.
//
// Values arrive from MIDI and from loaded presets *by name*, into a pending
// table, and are written into the real variable at declare time. Nothing holds
// a pointer across frames -- the pointers here belong to scene objects and
// globals, and a stale one would be a crash rather than a wrong value.
//
// ## Declaring is not drawing
//
// The registry is only complete if every control declares itself *every* frame,
// including the ones inside a collapsed header, on an inactive scene's page, or
// behind a disabled feature's tick box. Otherwise a loaded preset reaches only
// what happens to be on screen -- and, worse, a save writes the last value each
// undrawn control was seen with, so load-then-save quietly rewrites a preset
// with stale numbers.
//
// So the panel body always runs, top to bottom, and visibility gates only the
// ImGui call. `BeginGroup` and `BeginHidden` return nothing and never skip a
// body: pass the visibility in, do not branch around it.
//
//   ui::BeginGroup("rain", true, P.drops_on);   // drawn only when it rains,
//   ...                                        // declared always
//   ui::EndGroup();
//
// ## Banks
//
// A parameter belongs to exactly one bank, which decides which file it is saved
// to and which is the only classification in the system -- see kBankRules in
// the .cpp, which is the master table, and SettingsDoc(), which prints the live
// registry grouped by it. A parameter whose section matches no rule lands in
// Unassigned and is reported, loudly, in both the panel and the document. That
// is deliberate: the failure mode to design against is not a setting in the
// wrong bank, it is a setting nobody ever decided about.
#pragma once

#include "imgui.h"

#include <string>
#include <vector>

namespace ui {

// --- banks ------------------------------------------------------------------
// Where a parameter is saved, and therefore what it means.
//
//   Machine   the room, not the piece: sensor, screen, calibration, MIDI map.
//             One file, loaded at startup, never carried between venues.
//   Show      the running order -- what plays and when.
//   Look      composition that outlives any one scene: text, transition.
//   Mirror    the ripple scene. Many presets; this is the one that gets dialled
//             in per performance.
//   Roots     the root scene.
enum class Bank { Unassigned = 0, Machine, Show, Look, Mirror, Roots, Count };

const char* BankName(Bank b);
// Directory a bank's presets live in, and the extension they carry. Machine is
// a single file rather than a directory; MachinePath() gives it.
std::string BankDir(Bank b);
const char* BankExt(Bank b);
std::string MachinePath();
std::vector<std::string> ListBank(Bank b);

// Save/load one bank. Load applies by name at the next declaration -- which,
// given the above, is this frame for everything.
bool SaveBank(Bank b, const std::string& path, std::string& err);
bool LoadBank(Bank b, const std::string& path, std::string& err);

// How many parameters a bank holds, and how many of those are retired.
int BankCount(Bank b, int* retired_out = nullptr);

// --- sections ---------------------------------------------------------------
// Push a section around a group of controls; the parameter's full name is the
// section path plus the widget label. Names are what presets and MIDI maps are
// written in terms of, so renaming a section invalidates saved bindings -- the
// loader reports unknown names rather than dropping them silently.
//
// A section inherits its parent's bank, retirement and visibility. The root
// section's bank comes from kBankRules; SetBank overrides it for anything
// declared from here down, which is how a section that is mostly rig config
// with a few artistic dials in it gets split.
void PushSection(const char* name);
void PopSection();
void SetBank(Bank b);
Bank CurrentBank();

struct Section {
    explicit Section(const char* name) { PushSection(name); }
    ~Section() { PopSection(); }
};

// A collapsing header that is also a section. The body is *always* executed;
// `visible` and the header's own open/closed state gate drawing only.
void BeginGroup(const char* name, bool default_open = false, bool visible = true);
void EndGroup();

// A collapsing header that is *not* a section: it gates drawing without adding
// a level to the parameter paths underneath it. Most of the panel is built this
// way -- a section called "z" whose header reads "z latent" -- and turning
// those into sections would rename every key in every saved preset.
void BeginHeader(const char* label, bool default_open = false, bool visible = true);
void EndHeader();

// Visibility on its own, no header and no path level. For the scene pages and
// for anything gated on a tick box drawn elsewhere.
void BeginGate(bool visible);
void EndGate();

// Everything declared until the matching EndRetired is kept working -- old
// presets still load it, saves still write it -- but is not drawn unless the
// operator asks to see retired controls. This is how a parameter stops
// cluttering the panel without breaking a preset that mentions it.
// `label` is a header like BeginHeader's, and adds no path level either: a
// control does not change its name by being retired, or the presets that
// mention it would stop finding it.
void BeginRetired(const char* label);
void EndRetired();
void SetShowRetired(bool on);
bool ShowRetired();

// Whether a control here would be drawn. For hand-drawn widgets (combos, radio
// groups, readouts) that sit alongside declared ones.
bool Visible();

// --- controls ---------------------------------------------------------------
// Same signatures as the ImGui calls they replace, plus registration and a
// right-click menu for binding. Each returns true when the user changed it.
bool SliderFloat(const char* label, float* v, float lo, float hi,
                 const char* fmt = "%.3f", ImGuiSliderFlags flags = 0);
bool SliderInt(const char* label, int* v, int lo, int hi, const char* fmt = "%d");
bool Checkbox(const char* label, bool* v);
bool DragFloat(const char* label, float* v, float speed, float lo, float hi,
               const char* fmt = "%.3f");
bool ColorEdit3(const char* label, float* rgb);

// A control that should be saved and MIDI-bindable but drawn by hand (a combo,
// a radio group). Declares the value without drawing anything.
void DeclareInt(const char* label, int* v, int lo, int hi);
void DeclareFloat(const char* label, float* v, float lo, float hi);

// --- MIDI -------------------------------------------------------------------
// A control-change message: routed to whatever is bound to it, or consumed by
// the pending learn if one is armed.
void ApplyCC(int channel, int cc, int value);
// Arm learn: the next CC received binds to `path`. Empty cancels.
void SetLearnTarget(const std::string& path);
const std::string& LearnTarget();
// How many bindings exist, and a readable list for the panel.
int  BindingCount();
void ForEachBinding(void* user,
                    void (*fn)(void* user, const char* path, int channel, int cc));
void ClearBinding(const std::string& path);
void ClearAllBindings();

// Rolling log of the last few messages, so a controller that is sending
// something other than what you expect can be seen doing it.
struct RecentCC { int channel, cc, value; };
const std::vector<RecentCC>& RecentMessages();
void NoteCC(int channel, int cc, int value);

// --- presets ----------------------------------------------------------------
// Every declared parameter, in every bank, plus the MIDI map, as one flat
// key=value file. Unknown keys are skipped and absent ones keep their current
// value, so a preset written before a control existed still loads afterwards.
// The per-bank files are the same format with a subset of the keys, so a whole
// dump can still be loaded over a banked setup.
bool SavePreset(const std::string& path, std::string& err);
bool LoadPreset(const std::string& path, std::string& err);
std::string PresetDir();
std::vector<std::string> ListPresets();

// Names seen in the last loaded file that no control claimed. Shown in the
// panel rather than discarded: it is the difference between "that preset is
// old" and "that preset did nothing".
const std::vector<std::string>& UnclaimedKeys();

// --- the master document ----------------------------------------------------
// The live registry as markdown: every parameter, its bank, its range and its
// current value, plus the unassigned ones called out at the top. Generated from
// the declarations rather than maintained beside them, which is the only way it
// stays true. One frame of the panel is enough to produce a complete one, since
// declaring no longer depends on what is on screen.
std::string SettingsDoc();
bool WriteSettingsDoc(const std::string& path, std::string& err);

// Number of parameters declared in the frame just drawn.
int DeclaredCount();
// Parameters that landed in no bank, from the frame just drawn.
const std::vector<std::string>& UnassignedParams();
// Call once per frame, before the panel is built.
void BeginFrame();

}  // namespace ui
