#include "show_timeline.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>

namespace show {

// --- the graph ---------------------------------------------------------------
//
// The piece, as a table. Read it as prose: idle waits for someone; fitting
// either captures them or gives up when they leave; the transition ends itself;
// roots run until the room empties.
//
// Edge order is priority order. In `fitting` the fit is checked before the
// absence, so a frame where the fit lands as the person steps back reads as a
// capture -- which is the generous interpretation, and the right one.
namespace {

const Edge kIdleEdges[] = {
    {Event::FacePresent, Phase::Fitting, "face_hold", 1.5f},
};
const Edge kFittingEdges[] = {
    {Event::FitConverged, Phase::Transition, "fit_hold", 1.5f},
    {Event::FaceAbsent, Phase::Idle, "absent_hold", 2.5f},
};
const Edge kTransitionEdges[] = {
    // No debounce by default: the scene reports done once and means it.
    {Event::SceneDone, Phase::Roots, "done_hold", 0.f},
};
const Edge kRootsEdges[] = {
    {Event::FaceAbsent, Phase::Idle, "absent_hold", 8.f},
};

const PhaseGraph kGraph[(int)Phase::Count] = {
    // edges,          n, timeout,       min,  max
    {kIdleEdges,       1, Phase::Idle,    8.f,  0.f},
    {kFittingEdges,    2, Phase::Idle,    2.f, 30.f},
    {kTransitionEdges, 1, Phase::Roots,   0.f, 12.f},
    {kRootsEdges,      1, Phase::Idle,   40.f,  0.f},
};

static_assert(sizeof(kFittingEdges) / sizeof(Edge) <= kMaxEdges,
              "kMaxEdges must cover the widest phase");

const char* kPhaseNames[(int)Phase::Count] = {"idle", "fitting", "transition",
                                              "roots"};
const char* kEventNames[(int)Event::Count] = {
    "phase_start", "face_present", "face_absent", "fit_converged", "fit_lost",
    "scene_done"};

float clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

bool IsSpace(char c) { return std::isspace((unsigned char)c) != 0; }

std::string Trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && IsSpace(s[a])) ++a;
    while (b > a && IsSpace(s[b - 1])) --b;
    return s.substr(a, b - a);
}

// Everything after an unquoted `#`. Quote-aware so a caption may contain one.
std::string StripComment(const std::string& line) {
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '\\' && quoted) { ++i; continue; }
        if (c == '"') quoted = !quoted;
        else if (c == '#' && !quoted) return line.substr(0, i);
    }
    return line;
}

// `\n` becomes a real newline, which is how a two-line caption is written --
// the overlay splits on it and centres the lines. Applied to quoted and bare
// values alike, so remembering to quote is never the difference between two
// lines and a literal backslash-n.
std::string Unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 >= s.size()) { out += s[i]; continue; }
        const char n = s[++i];
        switch (n) {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            default:  out += n;    break;
        }
    }
    return out;
}

// Splits a line into `key` and `value`. The colon is optional -- `min: 8s` and
// `min 8s` are the same line -- because punctuation that can be forgotten in
// the dark of an install will be.
//
// The key is the leading run of letters and underscores; everything after it
// (past an optional colon) is the value, trimmed, with surrounding quotes
// removed. Quoting only matters for a value with a leading or trailing space,
// or one containing a `#`.
//
// Returns false for a blank or comment-only line.
bool SplitKV(const std::string& raw, std::string& key, std::string& val) {
    const std::string line = Trim(StripComment(raw));
    if (line.empty()) return false;

    size_t i = 0;
    while (i < line.size() &&
           (std::isalpha((unsigned char)line[i]) || line[i] == '_'))
        ++i;
    key = line.substr(0, i);

    while (i < line.size() && IsSpace(line[i])) ++i;
    if (i < line.size() && line[i] == ':') ++i;
    while (i < line.size() && IsSpace(line[i])) ++i;

    std::string v = Trim(line.substr(i));
    if (v.size() >= 2 && v.front() == '"' && v.back() == '"')
        v = v.substr(1, v.size() - 2);
    val = Unescape(v);
    return true;
}

// "2", "2s", "250ms". Returns false on anything that is not a number, so a
// typo'd key lands as an error on its own line rather than as a silent 0.
bool ParseDur(const std::string& s, float& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    const float v = std::strtof(s.c_str(), &end);
    if (end == s.c_str()) return false;
    const std::string suffix(end);
    if (suffix.empty() || suffix == "s") { out = v; return true; }
    if (suffix == "ms") { out = v * 0.001f; return true; }
    return false;
}

}  // namespace

const PhaseGraph& Graph(Phase p) {
    const int i = (int)p;
    return kGraph[(i >= 0 && i < (int)Phase::Count) ? i : 0];
}

const char* PhaseName(Phase p) {
    const int i = (int)p;
    return (i >= 0 && i < (int)Phase::Count) ? kPhaseNames[i] : "?";
}

bool ParsePhase(const std::string& s, Phase& out) {
    for (int i = 0; i < (int)Phase::Count; ++i) {
        if (s == kPhaseNames[i]) { out = (Phase)i; return true; }
    }
    return false;
}

const char* EventName(Event e) {
    const int i = (int)e;
    return (i >= 0 && i < (int)Event::Count) ? kEventNames[i] : "?";
}

bool ParseEvent(const std::string& s, Event& out) {
    for (int i = 0; i < (int)Event::Count; ++i) {
        if (s == kEventNames[i]) { out = (Event)i; return true; }
    }
    return false;
}

ShowScript::ShowScript() {
    for (int i = 0; i < (int)Phase::Count; ++i) {
        const PhaseGraph& g = kGraph[i];
        PhaseScript& ps = phases[i];
        ps.min_time = g.min_time;
        ps.max_time = g.max_time;
        for (int e = 0; e < g.edge_count; ++e) ps.hold[e] = g.edges[e].hold;
    }
}

std::vector<std::string> PhaseKeys(Phase p) {
    std::vector<std::string> k = {"min", "max", "text"};
    const PhaseGraph& g = Graph(p);
    for (int i = 0; i < g.edge_count; ++i) k.push_back(g.edges[i].key);
    return k;
}

// --- parsing -----------------------------------------------------------------

bool ParseShow(const std::string& text, ShowScript& out, std::string& err) {
    out = ShowScript();
    err.clear();

    // Where a sub-key attaches. Attachment is by *most recent item*, never by
    // indentation: the sub-keys (`at`, `for`, `fade`, `when`) are disjoint from
    // the keys that open one, so where a line belongs is unambiguous from the
    // line itself. Indenting the file is then purely for reading, and
    // re-indenting it cannot change its meaning -- the property a
    // whitespace-significant format does not have, and the reason this is not
    // one.
    bool in_text = false;   // the last item opened was a text cue
    size_t cue_idx = 0;

    Phase cur = Phase::Idle;
    bool have_phase = false;
    int lineno = 0;
    size_t pos = 0;

    auto fail = [&](const std::string& msg) {
        err = "line " + std::to_string(lineno) + ": " + msg;
        return false;
    };

    std::string key, val;
    while (pos <= text.size()) {
        const size_t nl = text.find('\n', pos);
        const std::string line =
            text.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        pos = (nl == std::string::npos) ? text.size() + 1 : nl + 1;
        ++lineno;

        if (!SplitKV(line, key, val)) continue;
        if (key.empty()) return fail("expected a key, got '" + Trim(line) + "'");

        if (key == "show") { out.name = val; in_text = false; continue; }
        if (key == "phase") {
            if (!ParsePhase(val, cur))
                return fail("phase: unknown phase '" + val + "'");
            have_phase = true;
            in_text = false;
            continue;
        }
        if (!have_phase) return fail(key + ": before any phase");
        PhaseScript& ps = out[cur];

        // --- a text cue's own keys ------------------------------------------
        if (key == "at" || key == "after" || key == "for" || key == "fade" ||
            key == "when") {
            if (!in_text)
                return fail(key + ": belongs to a text cue, and none is open");
            TextCue& c = ps.cues[cue_idx];
            if (key == "when") {
                if (!ParseEvent(val, c.anchor))
                    return fail("when: unknown event '" + val + "'");
                continue;
            }
            float* dst = (key == "for") ? &c.dur
                                        : (key == "fade" ? &c.fade : &c.at);
            if (!ParseDur(val, *dst))
                return fail(key + ": '" + val + "' is not a duration");
            continue;
        }

        if (key == "text") {
            TextCue c;
            c.text = val;
            cue_idx = ps.cues.size();
            ps.cues.push_back(c);
            in_text = true;
            continue;
        }

        in_text = false;

        if (key == "min") {
            if (!ParseDur(val, ps.min_time))
                return fail("min: '" + val + "' is not a duration");
            continue;
        }
        if (key == "max") {
            if (!ParseDur(val, ps.max_time))
                return fail("max: '" + val + "' is not a duration");
            continue;
        }

        // --- an edge's debounce ---------------------------------------------
        const PhaseGraph& g = Graph(cur);
        bool matched = false;
        for (int i = 0; i < g.edge_count && !matched; ++i) {
            if (key != g.edges[i].key) continue;
            if (!ParseDur(val, ps.hold[i]))
                return fail(key + ": '" + val + "' is not a duration");
            matched = true;
        }
        if (matched) continue;

        // The graph is fixed, so the set of knobs a phase has is knowable --
        // and a misspelled one is worth naming the alternatives for rather
        // than merely rejecting.
        std::string known;
        for (const std::string& k : PhaseKeys(cur))
            known += (known.empty() ? "" : ", ") + k;
        return fail("unknown key '" + key + "' in phase " +
                    std::string(PhaseName(cur)) + " (try: " + known + ")");
    }
    return true;
}

bool LoadShow(const std::string& path, ShowScript& out, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    std::fclose(f);
    return ParseShow(text, out, err);
}

// --- runtime -----------------------------------------------------------------

Timeline::Timeline() : Timeline(ShowScript()) {}

Timeline::Timeline(const ShowScript& s) {
    script_ = s;
    restart();
}

void Timeline::setScript(const ShowScript& s) {
    script_ = s;
    // Same phase, fresh clock: a reload retimes what is on screen instead of
    // cutting back to the top of the piece.
    enter(phase_, "script reloaded");
}

void Timeline::restart() { enter(Phase::Idle, "restart"); }

void Timeline::goTo(Phase p) { enter(p, "forced"); }

void Timeline::go() {
    // Edge 0 is the forward path, so "go" means the same thing in every phase
    // without the operator having to know which event it is short-circuiting.
    const PhaseGraph& g = Graph(phase_);
    if (g.edge_count > 0) enter(g.edges[0].target, "cue: go");
}

void Timeline::enter(Phase p, const std::string& reason) {
    phase_ = p;
    t_ = 0.0;
    reason_ = reason;
    scene_done_ = false;
    ++entries_;

    for (int i = 0; i < kMaxEdges; ++i) held_[i] = 0.f;
    cue_state_.assign(script_[phase_].cues.size(), CueState());
    text_ = ActiveText();
}

bool Timeline::eventLevel(Event e) const {
    switch (e) {
        case Event::PhaseStart:   return true;
        case Event::FacePresent:  return sig_.face_present;
        case Event::FaceAbsent:   return !sig_.face_present;
        case Event::FitConverged: return sig_.fit_converged;
        case Event::FitLost:      return !sig_.fit_converged;
        case Event::SceneDone:    return scene_done_;
        default:                  return false;
    }
}

void Timeline::advance(double dt) {
    const float fdt = (float)std::max(0.0, dt);
    t_ += fdt;

    const PhaseScript& ps = script_[phase_];
    const PhaseGraph& g = Graph(phase_);

    for (int i = 0; i < g.edge_count; ++i) {
        const Edge& e = g.edges[i];
        const bool level = eventLevel(e.event);
        held_[i] = level ? held_[i] + fdt : 0.f;
        if (!level || held_[i] < ps.hold[i]) continue;

        // The floor gates the conditions the room produces on its own. It does
        // not gate a scene reporting itself finished: holding a completed
        // transition on screen to satisfy a minimum would just be a freeze.
        if (t_ < ps.min_time && e.event != Event::SceneDone) continue;

        enter(e.target,
              std::string(PhaseName(e.target)) + " on " + EventName(e.event));
        updateText();
        return;
    }

    // The ceiling, after the edges: an event that fires on the same frame the
    // timeout expires is the more specific answer.
    if (ps.max_time > 0.f && t_ >= ps.max_time) {
        enter(g.timeout, std::string(PhaseName(g.timeout)) + " on timeout");
        updateText();
        return;
    }

    updateText();
}

void Timeline::updateText() {
    const PhaseScript& ps = script_[phase_];
    ActiveText best;
    double best_t0 = -1.0;

    for (size_t i = 0; i < ps.cues.size(); ++i) {
        const TextCue& c = ps.cues[i];
        CueState& st = cue_state_[i];

        if (!st.started) {
            if (!eventLevel(c.anchor)) continue;
            st.started = true;
            // PhaseStart is true from the first frame, so an unanchored cue
            // starts exactly at `at` rather than on the first frame past it --
            // a long frame cannot shift a caption.
            st.t0 = (c.anchor == Event::PhaseStart) ? c.at : t_ + c.at;
        }
        if (st.finished) continue;

        const double e = t_ - st.t0;
        if (e < 0.0) continue;
        if (c.dur > 0.f && e >= c.dur) { st.finished = true; continue; }

        // A cue shorter than two fades gets a symmetric triangle instead of a
        // fade-in that never completes before the fade-out starts pulling it
        // back down.
        float fade = c.fade;
        if (c.dur > 0.f && fade * 2.f > c.dur) fade = c.dur * 0.5f;

        float reveal = 1.f;
        if (fade > 0.f) {
            reveal = clamp01((float)e / fade);
            if (c.dur > 0.f)
                reveal = std::min(reveal, clamp01((float)(c.dur - e) / fade));
        }

        // Overlapping cues: the most recently started one is on top, which is
        // what a script that schedules a second caption over a first means.
        if (st.t0 >= best_t0) {
            best_t0 = st.t0;
            best.on = true;
            best.text = c.text;
            best.reveal = reveal;
        }
    }
    text_ = best;
}

float Timeline::phaseProgress() const {
    const PhaseScript& ps = script_[phase_];
    if (ps.max_time <= 0.f) return 0.f;
    return clamp01((float)(t_ / ps.max_time));
}

// --- files -------------------------------------------------------------------

std::string ShowDir() { return std::string(MIRROR_APP_SRC_DIR) + "/../shows"; }

std::vector<std::string> ListShows() {
    std::vector<std::string> out;
    DIR* d = opendir(ShowDir().c_str());
    if (!d) return out;
    while (struct dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > 5 && n.compare(n.size() - 5, 5, ".show") == 0)
            out.push_back(n.substr(0, n.size() - 5));
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace show
