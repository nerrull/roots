#include "ui_params.h"

#include <algorithm>
#include <cstdio>
#include <ctime>
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
    // the panel being open. A control inside a collapsed header is still
    // declared -- that is the point of the split -- but nothing else guarantees
    // the cache is fresh, so it is rewritten at every declaration.
    float f[3] = {0.f, 0.f, 0.f};
    int   i = 0;
    bool  b = false;
    Bank  bank = Bank::Unassigned;
    bool  retired = false;
    // Whether this path was declared in the frame just drawn. A parameter that
    // stops being declared -- a section deleted, a scene page removed -- would
    // otherwise sit in the registry forever, saved out of a stale cache.
    bool  live = false;
};

// A section on the stack. Bank, retirement and visibility are stored resolved
// rather than as deltas: a child inherits by copying, so reading the state is
// one lookup at the top instead of a walk.
struct Frame {
    std::string name;
    Bank bank = Bank::Unassigned;
    bool retired = false;
    bool visible = true;
};

struct State {
    std::vector<Frame> stack;
    std::map<std::string, Entry> params;          // everything ever declared
    std::map<std::string, std::pair<int, int>> bind;   // path -> (channel, cc)
    std::map<std::string, float> pending01;      // path -> normalised value
    std::map<std::string, std::string> loaded;   // path -> literal, from a preset
    std::vector<std::string> unclaimed;
    std::vector<std::string> unassigned;
    std::vector<RecentCC> recent;
    std::string learn;
    int declared_this_frame = 0;
    bool show_retired = false;
};

State& S() {
    static State s;
    return s;
}

// --- the master table --------------------------------------------------------
// Which bank a top-level section belongs to. This is the one place the question
// "where does this setting go" is answered; everything else -- the files, the
// panel's grouping, the generated document -- reads it. A section that is not
// here declares into Unassigned and is reported rather than guessed at.
struct BankRule { const char* section; Bank bank; };

const BankRule kBankRules[] = {
    // The room. Calibration and hardware: true of this venue, not of the piece.
    {"screen",        Bank::Machine},
    {"camera mask",   Bank::Machine},
    {"sensor",        Bank::Machine},
    // Face tracking straddles the line -- acquisition is rig, the identity fit
    // is artistic. The section declares Machine and calls SetBank(Look) around
    // the part that travels.
    {"face tracking", Bank::Machine},

    {"show",          Bank::Show},

    {"text",          Bank::Look},
    {"transition",    Bank::Look},

    {"mirror",        Bank::Mirror},
    {"roots",         Bank::Roots},
};

Bank BankForSection(const std::string& name) {
    for (const BankRule& r : kBankRules)
        if (name == r.section) return r.bank;
    return Bank::Unassigned;
}

// An unnamed frame contributes nothing to the path: it exists to carry
// visibility (a collapsing header, a tick box) without renaming what is inside
// it. Renaming is the one thing this registry must never do by accident, since
// the name is the whole contract with the preset files and the MIDI map.
std::string PathFor(const char* label) {
    std::string p;
    for (const Frame& f : S().stack) {
        if (f.name.empty()) continue;
        p += f.name; p += '/';
    }
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

// Whether the control being declared right now should also be drawn. Retired
// controls are declared like any other and simply not shown.
bool DrawHere() {
    if (S().stack.empty()) return true;
    const Frame& f = S().stack.back();
    if (!f.visible) return false;
    if (f.retired && !S().show_retired) return false;
    return true;
}

// Bookkeeping every declaration does, whether or not it draws: claim the entry,
// stamp it with the bank and retirement of the section it is in, and count it.
Entry& Declare(const std::string& path, Kind k, float lo, float hi) {
    Entry& e = S().params[path];
    e.kind = k; e.lo = lo; e.hi = hi;
    e.live = true;
    if (!S().stack.empty()) {
        e.bank = S().stack.back().bank;
        e.retired = S().stack.back().retired;
    }
    if (e.bank == Bank::Unassigned) S().unassigned.push_back(path);
    ++S().declared_this_frame;
    return e;
}

// The right-click menu every control carries. Kept here so binding a knob is
// the same gesture everywhere rather than a feature of a few special controls.
void ContextMenu(const std::string& path) {
    if (!ImGui::BeginPopupContextItem(path.c_str())) return;
    ImGui::TextDisabled("%s", path.c_str());
    auto pe = S().params.find(path);
    if (pe != S().params.end())
        ImGui::TextDisabled("bank: %s", BankName(pe->second.bank));
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
    const bool retired = !S().stack.empty() && S().stack.back().retired;
    if (retired) { ImGui::SameLine(); ImGui::TextDisabled("[retired]"); }
    if (it == S().bind.end() && !learning) return;
    ImGui::SameLine();
    if (learning) ImGui::TextColored(ImVec4(1.f, 0.85f, 0.4f, 1.f), "[learn]");
    else          ImGui::TextDisabled("[cc%d]", it->second.second);
}

// Pull any value that arrived by name -- MIDI or a loaded preset -- into the
// variable, at the moment the control is declared.
template <class T>
bool TakePending(const std::string& path, T&& setter) {
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
    return changed;
}

void WriteValue(std::ostream& o, const Entry& e) {
    switch (e.kind) {
        case Kind::Float:  o << e.f[0]; break;
        case Kind::Int:    o << e.i; break;
        case Kind::Bool:   o << (e.b ? 1 : 0); break;
        case Kind::Color3: o << e.f[0] << ' ' << e.f[1] << ' ' << e.f[2]; break;
    }
}

const char* KindName(Kind k) {
    switch (k) {
        case Kind::Float:  return "float";
        case Kind::Int:    return "int";
        case Kind::Bool:   return "bool";
        case Kind::Color3: return "rgb";
    }
    return "?";
}

bool MkDirP(const std::string& path) {
    if (mkdir(path.c_str(), 0755) == 0) return true;
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Write one bank's parameters (Bank::Count means all of them) plus, for Machine
// and for a whole dump, the MIDI map -- the controller is part of the rig, so a
// mirror preset carrying bindings would rewire the desk when it loaded.
bool WriteFile(const std::string& path, Bank only, std::string& err) {
    std::ofstream f(path);
    if (!f.is_open()) { err = "could not write " + path; return false; }
    f << "# mirror_app settings";
    if (only != Bank::Count) f << " -- bank: " << BankName(only);
    f << "\n";
    for (const auto& kv : S().params) {
        const Entry& e = kv.second;
        if (only != Bank::Count && e.bank != only) continue;
        f << "p " << kv.first << " = ";
        WriteValue(f, e);
        f << '\n';
    }
    if (only == Bank::Count || only == Bank::Machine) {
        for (const auto& kv : S().bind)
            f << "m " << kv.first << " = " << kv.second.first << ' '
              << kv.second.second << '\n';
    }
    return true;
}

bool ReadFile(const std::string& path, std::string& err) {
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
            // Applied when the control next declares itself, which -- since
            // declaring no longer depends on what is on screen -- is the next
            // frame, for every control in the app.
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

}  // namespace

// --- banks ------------------------------------------------------------------

const char* BankName(Bank b) {
    switch (b) {
        case Bank::Unassigned: return "unassigned";
        case Bank::Machine:    return "machine";
        case Bank::Show:       return "show";
        case Bank::Look:       return "look";
        case Bank::Mirror:     return "mirror";
        case Bank::Roots:      return "roots";
        case Bank::Count:      return "all";
    }
    return "?";
}

const char* BankExt(Bank b) {
    switch (b) {
        case Bank::Machine: return ".machine";
        case Bank::Show:    return ".show";
        case Bank::Look:    return ".look";
        case Bank::Mirror:  return ".mirror";
        case Bank::Roots:   return ".roots";
        default:            return ".set";
    }
}

std::string PresetDir() {
    return std::string(MIRROR_APP_SRC_DIR) + "/../presets";
}

std::string BankDir(Bank b) { return PresetDir() + "/" + BankName(b); }

// One file, not a directory: there is only ever one machine, and offering to
// pick between machine configurations is offering to load the wrong one.
std::string MachinePath() { return PresetDir() + "/machine" + BankExt(Bank::Machine); }

std::vector<std::string> ListBank(Bank b) {
    std::vector<std::string> out;
    const std::string dir = BankDir(b);
    const std::string ext = BankExt(b);
    DIR* d = opendir(dir.c_str());
    if (!d) return out;
    while (struct dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > ext.size() &&
            n.compare(n.size() - ext.size(), ext.size(), ext) == 0)
            out.push_back(n.substr(0, n.size() - ext.size()));
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

bool SaveBank(Bank b, const std::string& path, std::string& err) {
    MkDirP(PresetDir());
    if (b != Bank::Machine) MkDirP(BankDir(b));
    return WriteFile(path, b, err);
}

bool LoadBank(Bank, const std::string& path, std::string& err) {
    // The bank a key belongs to is decided by the code that declares it, not by
    // the file it arrived in, so loading is the same operation either way.
    return ReadFile(path, err);
}

int BankCount(Bank b, int* retired_out) {
    int n = 0, r = 0;
    for (const auto& kv : S().params) {
        if (!kv.second.live || kv.second.bank != b) continue;
        ++n;
        if (kv.second.retired) ++r;
    }
    if (retired_out) *retired_out = r;
    return n;
}

// --- sections ---------------------------------------------------------------

void PushSection(const char* name) {
    Frame f;
    f.name = name;
    if (S().stack.empty()) {
        f.bank = BankForSection(f.name);
    } else {
        const Frame& p = S().stack.back();
        f.bank = p.bank; f.retired = p.retired; f.visible = p.visible;
    }
    S().stack.push_back(std::move(f));
}

void PopSection() { if (!S().stack.empty()) S().stack.pop_back(); }

void SetBank(Bank b) { if (!S().stack.empty()) S().stack.back().bank = b; }
Bank CurrentBank() {
    return S().stack.empty() ? Bank::Unassigned : S().stack.back().bank;
}

namespace {

// The one implementation behind BeginGroup/BeginHeader/BeginRetired: draw the
// header if the enclosing section is drawing, then push a frame whose only job
// is to carry the result. The body after it always runs -- that is the whole
// point -- so nothing here ever tells a caller to skip anything.
void PushHeaderFrame(const char* label, const char* path_name,
                     bool default_open, bool visible, bool retired) {
    const bool parent_draws = DrawHere();
    const bool shown = parent_draws && visible && (!retired || S().show_retired);
    bool open = false;
    if (shown) {
        std::string h = label;
        if (retired) h += "  (retired)";
        open = ImGui::CollapsingHeader(
            h.c_str(), default_open ? ImGuiTreeNodeFlags_DefaultOpen : 0);
    }
    PushSection(path_name);
    S().stack.back().visible = shown && open;
    if (retired) S().stack.back().retired = true;
}

}  // namespace

void BeginGroup(const char* name, bool default_open, bool visible) {
    PushHeaderFrame(name, name, default_open, visible, /*retired=*/false);
}
void EndGroup() { PopSection(); }

void BeginHeader(const char* label, bool default_open, bool visible) {
    PushHeaderFrame(label, "", default_open, visible, /*retired=*/false);
}
void EndHeader() { PopSection(); }

void BeginGate(bool visible) {
    const bool parent_draws = DrawHere();
    PushSection("");
    S().stack.back().visible = parent_draws && visible;
}
void EndGate() { PopSection(); }

void BeginRetired(const char* label) {
    PushHeaderFrame(label, "", /*default_open=*/false, /*visible=*/true,
                    /*retired=*/true);
}
void EndRetired() { PopSection(); }

void SetShowRetired(bool on) { S().show_retired = on; }
bool ShowRetired() { return S().show_retired; }

bool Visible() { return DrawHere(); }

int DeclaredCount() { return S().declared_this_frame; }
const std::vector<std::string>& UnassignedParams() { return S().unassigned; }

void BeginFrame() {
    S().declared_this_frame = 0;
    S().unassigned.clear();
    for (auto& kv : S().params) kv.second.live = false;
    S().stack.clear();
}

// --- controls ---------------------------------------------------------------

bool SliderFloat(const char* label, float* v, float lo, float hi,
                 const char* fmt, ImGuiSliderFlags flags) {
    const std::string path = PathFor(CleanLabel(label).c_str());
    Entry& e = Declare(path, Kind::Float, lo, hi);
    bool changed = TakePending(path, [&](const std::string& s, bool norm) {
        const float x = (float)atof(s.c_str());
        *v = norm ? lo + (hi - lo) * x : x;
    });
    if (DrawHere()) {
        if (ImGui::SliderFloat(label, v, lo, hi, fmt, flags)) changed = true;
        ContextMenu(path);
        BindingMark(path);
    }
    e.f[0] = *v;
    return changed;
}

bool SliderInt(const char* label, int* v, int lo, int hi, const char* fmt) {
    const std::string path = PathFor(CleanLabel(label).c_str());
    Entry& e = Declare(path, Kind::Int, (float)lo, (float)hi);
    bool changed = TakePending(path, [&](const std::string& s, bool norm) {
        if (norm) *v = lo + (int)((hi - lo) * (float)atof(s.c_str()) + 0.5f);
        else      *v = atoi(s.c_str());
    });
    if (DrawHere()) {
        if (ImGui::SliderInt(label, v, lo, hi, fmt)) changed = true;
        ContextMenu(path);
        BindingMark(path);
    }
    e.i = *v;
    return changed;
}

bool Checkbox(const char* label, bool* v) {
    const std::string path = PathFor(CleanLabel(label).c_str());
    Entry& e = Declare(path, Kind::Bool, 0.f, 1.f);
    bool changed = TakePending(path, [&](const std::string& s, bool norm) {
        // A knob past halfway is on. A button sending 0/127 works the same way,
        // which is what most controllers send for a toggle.
        *v = norm ? (atof(s.c_str()) >= 0.5) : (atoi(s.c_str()) != 0);
    });
    if (DrawHere()) {
        if (ImGui::Checkbox(label, v)) changed = true;
        ContextMenu(path);
        BindingMark(path);
    }
    e.b = *v;
    return changed;
}

bool DragFloat(const char* label, float* v, float speed, float lo, float hi,
               const char* fmt) {
    const std::string path = PathFor(CleanLabel(label).c_str());
    Entry& e = Declare(path, Kind::Float, lo, hi);
    bool changed = TakePending(path, [&](const std::string& s, bool norm) {
        const float x = (float)atof(s.c_str());
        *v = norm ? lo + (hi - lo) * x : x;
    });
    if (DrawHere()) {
        if (ImGui::DragFloat(label, v, speed, lo, hi, fmt)) changed = true;
        ContextMenu(path);
        BindingMark(path);
    }
    e.f[0] = *v;
    return changed;
}

bool ColorEdit3(const char* label, float* rgb) {
    const std::string path = PathFor(CleanLabel(label).c_str());
    Entry& e = Declare(path, Kind::Color3, 0.f, 1.f);
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
    if (DrawHere()) {
        if (ImGui::ColorEdit3(label, rgb)) changed = true;
        ContextMenu(path);
    }
    e.f[0] = rgb[0]; e.f[1] = rgb[1]; e.f[2] = rgb[2];
    return changed;
}

void DeclareInt(const char* label, int* v, int lo, int hi) {
    const std::string path = PathFor(CleanLabel(label).c_str());
    Entry& e = Declare(path, Kind::Int, (float)lo, (float)hi);
    TakePending(path, [&](const std::string& s, bool norm) {
        if (norm) *v = lo + (int)((hi - lo) * (float)atof(s.c_str()) + 0.5f);
        else      *v = atoi(s.c_str());
    });
    e.i = *v;
}

void DeclareFloat(const char* label, float* v, float lo, float hi) {
    const std::string path = PathFor(CleanLabel(label).c_str());
    Entry& e = Declare(path, Kind::Float, lo, hi);
    TakePending(path, [&](const std::string& s, bool norm) {
        const float x = (float)atof(s.c_str());
        *v = norm ? lo + (hi - lo) * x : x;
    });
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
    MkDirP(PresetDir());
    return WriteFile(path, Bank::Count, err);
}

bool LoadPreset(const std::string& path, std::string& err) {
    return ReadFile(path, err);
}

const std::vector<std::string>& UnclaimedKeys() { return S().unclaimed; }

// --- the master document ----------------------------------------------------

std::string SettingsDoc() {
    std::ostringstream o;
    o << "# mirror_app settings\n\n"
      << "Generated from the live registry -- do not edit by hand. Every row is a\n"
      << "control that declared itself in the frame this was written from, which is\n"
      << "all of them: declaring does not depend on what the panel has open.\n\n"
      << "Regenerate from the panel: **settings -> write SETTINGS.md**.\n\n";

    o << "## Where a setting goes\n\n"
      << "| bank | file | what belongs here |\n|---|---|---|\n"
      << "| `machine` | `presets/machine.machine` | the room, not the piece: sensor,"
      << " screen, camera mask, MIDI map. Loaded at startup, never carried to another"
      << " venue. |\n"
      << "| `show` | `presets/show/*.show` | the running order -- what plays and when. |\n"
      << "| `look` | `presets/look/*.look` | composition that outlives one scene: text"
      << " overlay, transition. |\n"
      << "| `mirror` | `presets/mirror/*.mirror` | the ripple scene. Many presets;"
      << " this is the one dialled in per performance. |\n"
      << "| `roots` | `presets/roots/*.roots` | the root scene. |\n\n"
      << "The mapping from a top-level panel section to a bank is `kBankRules` in\n"
      << "`src/ui_params.cpp`. Add a section, add a rule -- otherwise its parameters\n"
      << "land in `unassigned` and are listed below until somebody decides.\n\n";

    if (!S().unassigned.empty()) {
        o << "## Unassigned (" << S().unassigned.size() << ")\n\n"
          << "These belong to no bank, so they are saved only in a whole-registry\n"
          << "dump and load into nothing banked. Add a rule for their section.\n\n";
        for (const std::string& p : S().unassigned) o << "- `" << p << "`\n";
        o << "\n";
    }

    int retired_total = 0;
    for (int b = (int)Bank::Machine; b < (int)Bank::Count; ++b) {
        const Bank bank = (Bank)b;
        int retired = 0;
        const int n = BankCount(bank, &retired);
        if (n == 0) continue;
        retired_total += retired;
        o << "## " << BankName(bank) << " (" << n << " parameters";
        if (retired) o << ", " << retired << " retired";
        o << ")\n\n| parameter | type | range | value |\n|---|---|---|---|\n";
        for (const auto& kv : S().params) {
            const Entry& e = kv.second;
            if (!e.live || e.bank != bank) continue;
            o << "| `" << kv.first << "`";
            if (e.retired) o << " _(retired)_";
            o << " | " << KindName(e.kind) << " | ";
            if (e.kind == Kind::Bool || e.kind == Kind::Color3) o << "--";
            else o << e.lo << " .. " << e.hi;
            o << " | ";
            WriteValue(o, e);
            o << " |\n";
        }
        o << "\n";
    }

    if (retired_total) {
        o << "## Retired\n\n"
          << retired_total << " parameter(s) are marked retired: still loaded, still\n"
          << "saved, not drawn unless **show retired** is ticked. That is how a control\n"
          << "leaves the panel without breaking a preset that mentions it. To delete one\n"
          << "for good, remove the declaration and the key from every preset file.\n\n";
    }
    return o.str();
}

bool WriteSettingsDoc(const std::string& path, std::string& err) {
    std::ofstream f(path);
    if (!f.is_open()) { err = "could not write " + path; return false; }
    f << SettingsDoc();
    return true;
}

}  // namespace ui
