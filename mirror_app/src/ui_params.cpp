#include "ui_params.h"

#include <algorithm>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <map>
#include <sstream>
#include <sys/stat.h>

namespace ui {

namespace {

enum class Kind { Float, Int, Bool, Color3 };

struct Entry {
    Kind  kind;
    float lo = 0.f, hi = 1.f;
    // The value, kept alongside the pointer so a preset can be written without
    // the panel being open. A control inside a collapsed header is not drawn,
    // so its pointer is not offered that frame -- but its value is still part
    // of the settings and must still be saved.
    float f[3] = {0.f, 0.f, 0.f};
    int   i = 0;
    bool  b = false;
};

struct State {
    std::vector<std::string> stack;
    std::map<std::string, Entry> params;          // everything ever declared
    std::map<std::string, std::pair<int, int>> bind;   // path -> (channel, cc)
    std::map<std::string, float> pending01;      // path -> normalised value
    std::map<std::string, std::string> loaded;   // path -> literal, from a preset
    std::vector<std::string> unclaimed;
    std::vector<RecentCC> recent;
    std::string learn;
    int declared_this_frame = 0;
};

State& S() {
    static State s;
    return s;
}

std::string PathFor(const char* label) {
    std::string p;
    for (const std::string& s : S().stack) { p += s; p += '/'; }
    p += label;
    return p;
}

// Strip an ImGui label of its "##id" suffix so the saved name is the visible
// one -- otherwise a cosmetic id change silently invalidates saved presets.
//
// A '/' in the label is rewritten, since that character separates sections: a
// slider called "z rate /s" would otherwise register as a parameter "s" inside
// a section "z rate ", which is not wrong so much as unreadable in a preset
// file and in the bindings list.
std::string CleanLabel(const char* label) {
    std::string s = label;
    const size_t h = s.find("##");
    if (h != std::string::npos) s = s.substr(0, h);
    for (char& c : s) if (c == '/') c = '-';
    return s;
}

void Trim(std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    const size_t b = s.find_last_not_of(" \t\r\n");
    s = (a == std::string::npos) ? std::string() : s.substr(a, b - a + 1);
}

// The right-click menu every control carries. Kept here so binding a knob is
// the same gesture everywhere rather than a feature of a few special controls.
void ContextMenu(const std::string& path) {
    if (!ImGui::BeginPopupContextItem(path.c_str())) return;
    ImGui::TextDisabled("%s", path.c_str());
    ImGui::Separator();
    auto it = S().bind.find(path);
    if (it != S().bind.end()) {
        ImGui::Text("bound to ch %d cc %d", it->second.first + 1, it->second.second);
        if (ImGui::MenuItem("clear binding")) S().bind.erase(path);
    } else if (S().learn == path) {
        ImGui::TextColored(ImVec4(1.f, 0.85f, 0.4f, 1.f), "move a control...");
        if (ImGui::MenuItem("cancel")) S().learn.clear();
    } else {
        if (ImGui::MenuItem("MIDI learn")) S().learn = path;
    }
    ImGui::EndPopup();
}

// A bound control shows it, so a panel with a controller attached can be read
// at a glance rather than by right-clicking everything.
void BindingMark(const std::string& path) {
    auto it = S().bind.find(path);
    const bool learning = (S().learn == path);
    if (it == S().bind.end() && !learning) return;
    ImGui::SameLine();
    if (learning) ImGui::TextColored(ImVec4(1.f, 0.85f, 0.4f, 1.f), "[learn]");
    else          ImGui::TextDisabled("[cc%d]", it->second.second);
}

// Pull any value that arrived by name -- MIDI or a loaded preset -- into the
// variable, at the moment the control is declared.
template <class T>
bool TakePending(const std::string& path, Entry& e, T&& setter) {
    bool changed = false;
    auto lp = S().loaded.find(path);
    if (lp != S().loaded.end()) {
        setter(lp->second, /*normalised=*/false);
        S().loaded.erase(lp);
        changed = true;
    }
    auto pp = S().pending01.find(path);
    if (pp != S().pending01.end()) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.9g", pp->second);
        setter(std::string(buf), /*normalised=*/true);
        S().pending01.erase(pp);
        changed = true;
    }
    (void)e;
    return changed;
}

}  // namespace

void PushSection(const char* name) { S().stack.push_back(name); }
void PopSection() { if (!S().stack.empty()) S().stack.pop_back(); }

bool BeginGroup(const char* name, bool default_open) {
    const bool open = ImGui::CollapsingHeader(
        name, default_open ? ImGuiTreeNodeFlags_DefaultOpen : 0);
    PushSection(name);
    return open;
}
void EndGroup() { PopSection(); }

int DeclaredCount() { return S().declared_this_frame; }

void BeginFrame() { S().declared_this_frame = 0; }

bool SliderFloat(const char* label, float* v, float lo, float hi,
                 const char* fmt, ImGuiSliderFlags flags) {
    const std::string path = PathFor(CleanLabel(label).c_str());
    Entry& e = S().params[path];
    e.kind = Kind::Float; e.lo = lo; e.hi = hi;
    bool changed = TakePending(path, e, [&](const std::string& s, bool norm) {
        const float x = (float)atof(s.c_str());
        *v = norm ? lo + (hi - lo) * x : x;
    });
    ++S().declared_this_frame;
    if (ImGui::SliderFloat(label, v, lo, hi, fmt, flags)) changed = true;
    ContextMenu(path);
    BindingMark(path);
    e.f[0] = *v;
    return changed;
}

bool SliderInt(const char* label, int* v, int lo, int hi, const char* fmt) {
    const std::string path = PathFor(CleanLabel(label).c_str());
    Entry& e = S().params[path];
    e.kind = Kind::Int; e.lo = (float)lo; e.hi = (float)hi;
    bool changed = TakePending(path, e, [&](const std::string& s, bool norm) {
        if (norm) *v = lo + (int)((hi - lo) * (float)atof(s.c_str()) + 0.5f);
        else      *v = atoi(s.c_str());
    });
    ++S().declared_this_frame;
    if (ImGui::SliderInt(label, v, lo, hi, fmt)) changed = true;
    ContextMenu(path);
    BindingMark(path);
    e.i = *v;
    return changed;
}

bool Checkbox(const char* label, bool* v) {
    const std::string path = PathFor(CleanLabel(label).c_str());
    Entry& e = S().params[path];
    e.kind = Kind::Bool; e.lo = 0.f; e.hi = 1.f;
    bool changed = TakePending(path, e, [&](const std::string& s, bool norm) {
        // A knob past halfway is on. A button sending 0/127 works the same way,
        // which is what most controllers send for a toggle.
        *v = norm ? (atof(s.c_str()) >= 0.5) : (atoi(s.c_str()) != 0);
    });
    ++S().declared_this_frame;
    if (ImGui::Checkbox(label, v)) changed = true;
    ContextMenu(path);
    BindingMark(path);
    e.b = *v;
    return changed;
}

bool DragFloat(const char* label, float* v, float speed, float lo, float hi,
               const char* fmt) {
    const std::string path = PathFor(CleanLabel(label).c_str());
    Entry& e = S().params[path];
    e.kind = Kind::Float; e.lo = lo; e.hi = hi;
    bool changed = TakePending(path, e, [&](const std::string& s, bool norm) {
        const float x = (float)atof(s.c_str());
        *v = norm ? lo + (hi - lo) * x : x;
    });
    ++S().declared_this_frame;
    if (ImGui::DragFloat(label, v, speed, lo, hi, fmt)) changed = true;
    ContextMenu(path);
    BindingMark(path);
    e.f[0] = *v;
    return changed;
}

bool ColorEdit3(const char* label, float* rgb) {
    const std::string path = PathFor(CleanLabel(label).c_str());
    Entry& e = S().params[path];
    e.kind = Kind::Color3;
    // Not MIDI-bindable: one CC cannot say three numbers, and quietly binding
    // it to the red channel would be worse than not offering it.
    auto lp = S().loaded.find(path);
    bool changed = false;
    if (lp != S().loaded.end()) {
        std::istringstream ss(lp->second);
        ss >> rgb[0] >> rgb[1] >> rgb[2];
        S().loaded.erase(lp);
        changed = true;
    }
    ++S().declared_this_frame;
    if (ImGui::ColorEdit3(label, rgb)) changed = true;
    e.f[0] = rgb[0]; e.f[1] = rgb[1]; e.f[2] = rgb[2];
    return changed;
}

void DeclareInt(const char* label, int* v, int lo, int hi) {
    const std::string path = PathFor(CleanLabel(label).c_str());
    Entry& e = S().params[path];
    e.kind = Kind::Int; e.lo = (float)lo; e.hi = (float)hi;
    TakePending(path, e, [&](const std::string& s, bool norm) {
        if (norm) *v = lo + (int)((hi - lo) * (float)atof(s.c_str()) + 0.5f);
        else      *v = atoi(s.c_str());
    });
    ++S().declared_this_frame;
    e.i = *v;
}

void DeclareFloat(const char* label, float* v, float lo, float hi) {
    const std::string path = PathFor(CleanLabel(label).c_str());
    Entry& e = S().params[path];
    e.kind = Kind::Float; e.lo = lo; e.hi = hi;
    TakePending(path, e, [&](const std::string& s, bool norm) {
        const float x = (float)atof(s.c_str());
        *v = norm ? lo + (hi - lo) * x : x;
    });
    ++S().declared_this_frame;
    e.f[0] = *v;
}

// --- MIDI -------------------------------------------------------------------

void NoteCC(int channel, int cc, int value) {
    auto& r = S().recent;
    r.insert(r.begin(), RecentCC{channel, cc, value});
    if (r.size() > 8) r.resize(8);
}

const std::vector<RecentCC>& RecentMessages() { return S().recent; }

void ApplyCC(int channel, int cc, int value) {
    NoteCC(channel, cc, value);
    if (!S().learn.empty()) {
        // Learn takes the message rather than acting on it: the knob is
        // somewhere arbitrary when it is bound, and jumping the parameter there
        // is a surprise nobody wants at the moment they are wiring things up.
        S().bind[S().learn] = {channel, cc};
        S().learn.clear();
        return;
    }
    for (const auto& kv : S().bind) {
        if (kv.second.first == channel && kv.second.second == cc)
            S().pending01[kv.first] = float(value) / 127.f;
    }
}

void SetLearnTarget(const std::string& path) { S().learn = path; }
const std::string& LearnTarget() { return S().learn; }
int BindingCount() { return (int)S().bind.size(); }

void ForEachBinding(void* user,
                    void (*fn)(void* user, const char* path, int ch, int cc)) {
    for (const auto& kv : S().bind)
        fn(user, kv.first.c_str(), kv.second.first, kv.second.second);
}

void ClearBinding(const std::string& path) { S().bind.erase(path); }
void ClearAllBindings() { S().bind.clear(); }

// --- presets ----------------------------------------------------------------

std::string PresetDir() {
    return std::string(MIRROR_APP_SRC_DIR) + "/../presets";
}

std::vector<std::string> ListPresets() {
    std::vector<std::string> out;
    DIR* d = opendir(PresetDir().c_str());
    if (!d) return out;
    while (struct dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > 4 && n.compare(n.size() - 4, 4, ".set") == 0)
            out.push_back(n.substr(0, n.size() - 4));
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

bool SavePreset(const std::string& path, std::string& err) {
    mkdir(PresetDir().c_str(), 0755);
    std::ofstream f(path);
    if (!f.is_open()) { err = "could not write " + path; return false; }
    f << "# mirror_app settings\n";
    for (const auto& kv : S().params) {
        const Entry& e = kv.second;
        f << "p " << kv.first << " = ";
        switch (e.kind) {
            case Kind::Float:  f << e.f[0]; break;
            case Kind::Int:    f << e.i; break;
            case Kind::Bool:   f << (e.b ? 1 : 0); break;
            case Kind::Color3: f << e.f[0] << ' ' << e.f[1] << ' ' << e.f[2]; break;
        }
        f << '\n';
    }
    for (const auto& kv : S().bind)
        f << "m " << kv.first << " = " << kv.second.first << ' ' << kv.second.second << '\n';
    return true;
}

bool LoadPreset(const std::string& path, std::string& err) {
    std::ifstream f(path);
    if (!f.is_open()) { err = "could not read " + path; return false; }
    S().loaded.clear();
    S().unclaimed.clear();
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos || line.size() < 3) continue;
        const char tag = line[0];
        std::string key = line.substr(2, eq - 2), val = line.substr(eq + 1);
        Trim(key); Trim(val);
        if (key.empty()) continue;
        if (tag == 'p') {
            // Values are applied when the control next declares itself, which
            // is this frame for anything visible and the frame a collapsed
            // section is opened for anything else.
            S().loaded[key] = val;
            if (!S().params.count(key)) S().unclaimed.push_back(key);
        } else if (tag == 'm') {
            std::istringstream ss(val);
            int ch = 0, cc = 0;
            ss >> ch >> cc;
            S().bind[key] = {ch, cc};
        }
    }
    return true;
}

const std::vector<std::string>& UnclaimedKeys() { return S().unclaimed; }

}  // namespace ui
