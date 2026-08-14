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
#pragma once

#include "imgui.h"

#include <string>
#include <vector>

namespace ui {

// --- sections ---------------------------------------------------------------
// Push a section around a group of controls; the parameter's full name is the
// section path plus the widget label. Names are what presets and MIDI maps are
// written in terms of, so renaming a section invalidates saved bindings -- the
// loader reports unknown names rather than dropping them silently.
void PushSection(const char* name);
void PopSection();

struct Section {
    explicit Section(const char* name) { PushSection(name); }
    ~Section() { PopSection(); }
};

// A collapsing header that is also a section. Returns whether it is open.
bool BeginGroup(const char* name, bool default_open = false);
void EndGroup();

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
// Every declared parameter, plus the MIDI map, as one flat key=value file.
// Unknown keys are skipped and absent ones keep their current value, so a
// preset written before a control existed still loads afterwards.
bool SavePreset(const std::string& path, std::string& err);
bool LoadPreset(const std::string& path, std::string& err);
std::string PresetDir();
std::vector<std::string> ListPresets();

// Names seen in the last loaded file that no control claimed. Shown in the
// panel rather than discarded: it is the difference between "that preset is
// old" and "that preset did nothing".
const std::vector<std::string>& UnclaimedKeys();

// Number of parameters declared in the frame just drawn.
int DeclaredCount();
// Call once per frame, before the panel is built.
void BeginFrame();

}  // namespace ui
