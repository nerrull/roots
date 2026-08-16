// show_timeline — the piece's running order: which of the four phases is up,
// when it ends, and what text is on screen while it runs.
//
// The installation is not a demo with a scene radio button. It idles as a
// mirror until someone walks up to it, fits their face, falls through the
// transition and grows roots out of them, and then lets go when they leave.
//
// ## The shape is code; the timing is data
//
// That sequence is the piece. It is not a thing to be configured: idle waits
// for someone, fitting captures them, the transition hands off, roots grow, and
// an empty room resets it. A format that let you rewire *that* would be a
// format in which the piece could be described wrongly, and every one of its
// edges would need a runtime error for the case where it was.
//
// So the graph below is a fixed table -- phases, the events that move between
// them, and which way each edge points. What an install actually needs to
// change is *when*: how long the idle floor is, how long a face must be gone
// before the piece lets go, when a caption lands. Those are durations, and
// durations are what the script file sets. Every one has a default here, so a
// script names only what it moves and an empty script still runs the piece.
//
// ## Time and events, composed
//
// A phase advances on either, and the two compose:
//
//   * **`min`** is a floor. The events the room produces on its own -- a face
//     arriving, a fit converging -- cannot advance the phase below it, so
//     someone walking past cannot retrigger the piece every few seconds.
//   * **`max`** is a ceiling with a target fixed in the graph: the escape hatch
//     for a fit that never converges, so a person in a hat cannot strand the
//     piece.
//   * Between them the events decide, each debounced by its own `hold` -- a
//     dropped tracker frame is not somebody leaving the room.
//
// Two things ignore the floor, both deliberately: an edge whose event is a
// scene reporting itself finished, and an operator cue. Holding a completed
// transition on screen to satisfy a minimum is a freeze, not a beat.
//
// ## What this module is not
//
// It holds no Metal, no scenes and no textures: it is a state machine over
// durations and booleans, which is what makes it testable headlessly. The host
// (main.mm) reads `phase()` to pick what to render, watches `entries()` to know
// a phase just started -- that is where TransitionScene::restart() goes -- and
// copies `text()` into TextParams. Signals go the other way, once a frame.
#pragma once

#include <string>
#include <vector>

namespace show {

// The running order. The values are stable: presets and MIDI mappings refer to
// them by name, but the UI indexes by number.
enum class Phase : int {
    Idle = 0,        // the neural mirror alone, waiting
    Fitting = 1,     // a face is present and the morphable fit is converging
    Transition = 2,  // the hydro-dip handoff
    Roots = 3,       // roots growing through the fitted face
    Count = 4,
};

const char* PhaseName(Phase p);
bool ParsePhase(const std::string& s, Phase& out);

// What an edge waits for.
//
// The `_present`/`_absent` pairs are edges *derived from a level*: the host sets
// a boolean every frame and this decides when it has been steady long enough to
// count. A tracker that drops one frame out of thirty must not read as somebody
// leaving the room, which is what `hold` is for.
enum class Event : int {
    PhaseStart = 0, // true from the moment the phase opens (text anchors only)
    FacePresent,    // a face has been tracked for `hold` seconds
    FaceAbsent,     // no face for `hold` seconds
    FitConverged,   // the fit has been settled for `hold` seconds
    FitLost,        // the fit stopped being settled
    SceneDone,      // the phase's own scene reported completion
    Count,
};

const char* EventName(Event e);
bool ParseEvent(const std::string& s, Event& out);

// --- the graph ---------------------------------------------------------------
// Fixed. `key` is the script key that tunes this edge's debounce, and it is the
// only part of an edge a script can touch.
struct Edge {
    Event event;
    Phase target;
    const char* key;
    float hold;          // default debounce, seconds
};

// A phase's outgoing edges, in priority order: they are checked in this order
// and the first match on a frame wins, so an ambiguous moment resolves the way
// the graph reads. Index 0 is the **forward path** through the piece -- the one
// an operator's "go" cue takes.
struct PhaseGraph {
    const Edge* edges;
    int edge_count;
    Phase timeout;       // where `max` sends it
    float min_time;      // default floor
    float max_time;      // default ceiling, 0 = none
};

const PhaseGraph& Graph(Phase p);
// Longest edge list of any phase, so PhaseScript can hold its holds inline.
constexpr int kMaxEdges = 4;

// --- the script --------------------------------------------------------------

// Text scheduled inside a phase. Either at a time from the phase's start, or
// anchored to an event with `at` as the delay after it fires.
//
// `fade` drives the reveal envelope rather than an opacity: the overlay's
// turbulent dissolve is its own 0->1 timeline, so text arrives by coming
// together out of the noise and leaves by breaking apart -- consistent with how
// everything else in the piece appears.
struct TextCue {
    std::string text;
    Event anchor = Event::PhaseStart;
    float at = 0.f;                 // delay after the anchor
    float dur = 0.f;                // 0 = until the phase ends
    float fade = 0.6f;              // in and out, seconds
};

// Everything a script may set, for one phase. Constructed from the graph's
// defaults, so an absent key means "as designed" rather than "zero".
struct PhaseScript {
    float min_time = 0.f;
    float max_time = 0.f;
    float hold[kMaxEdges] = {};      // parallel to Graph(p).edges
    std::vector<TextCue> cues;
};

struct ShowScript {
    ShowScript();                    // seeded from the graph
    std::string name;

    PhaseScript phases[(int)Phase::Count];
    const PhaseScript& operator[](Phase p) const { return phases[(int)p]; }
    PhaseScript& operator[](Phase p) { return phases[(int)p]; }
};

// The script file. Line-oriented; `#` to end of line is a comment, blank lines
// are ignored, and indentation is decorative -- a key attaches to the most
// recent item that accepts it, never to whatever it is nested under, so
// re-indenting a file cannot change its meaning. The colon is optional.
//
//   show: jardins racine
//
//   phase idle
//       min: 8s                    # floor
//       face_hold: 1.5s            # debounce on the edge to fitting
//       text: "JARDINS\nRACINE"
//           at: 3s
//           for: 7s
//           fade: 1.2s
//
//   phase fitting
//       min: 2s
//       max: 30s                   # the fit never took; the graph sends it home
//       fit_hold: 1.5s
//       absent_hold: 2.5s
//       text: "RESTEZ IMMOBILE"
//           when: fit_converged    # anchored to an event instead of a time
//           after: 0.5s
//           for: 4s
//
// Returns false with `err` set to "line N: ..." on the first bad line. The
// script is left partly built, so callers should keep the old one on failure
// rather than swapping in a fragment.
bool ParseShow(const std::string& text, ShowScript& out, std::string& err);
bool LoadShow(const std::string& path, ShowScript& out, std::string& err);

// The keys a script may set inside `phase p`, for error messages and the panel:
// "min", "max", "text", and this phase's edge keys.
std::vector<std::string> PhaseKeys(Phase p);

// --- runtime -----------------------------------------------------------------

// What the host should be showing this frame.
struct ActiveText {
    bool on = false;
    std::string text;
    float reveal = 0.f;   // 0..1, feeds TextParams::reveal
};

// Level signals, set by the host once a frame. Levels rather than edges because
// the host has them as booleans anyway and edge detection with debounce is
// exactly what this module should own.
struct Signals {
    bool face_present = false;
    bool fit_converged = false;
};

class Timeline {
public:
    Timeline();
    explicit Timeline(const ShowScript& s);

    // Swapping the script mid-run keeps the current phase and restarts its
    // clock -- a reload during a show should retime what is running, not cut to
    // the top of the piece.
    void setScript(const ShowScript& s);
    const ShowScript& script() const { return script_; }

    // Back to Idle with every clock cleared.
    void restart();

    void setSignals(const Signals& s) { sig_ = s; }
    // The phase's own scene finished (TransitionScene::done(), and anything
    // else that owns its duration). Latched until the phase changes, so a host
    // that reports it every frame after completion behaves the same as one that
    // reports it once.
    void sceneDone() { scene_done_ = true; }
    // The operator's "go": take the phase's forward edge now, whatever it was
    // waiting for and whatever its floor says. A MIDI CC, a key, a UI button.
    void go();

    // Forces a phase now, ignoring floors and ceilings. The operator override;
    // it counts as an entry, so the host restarts scenes for it.
    void goTo(Phase p);

    void advance(double dt);

    Phase phase() const { return phase_; }
    double phaseTime() const { return t_; }
    // Incremented on every phase entry, including the first and including
    // goTo(). The host keeps its own copy and compares: a change means "this
    // phase just started", which is where scene restarts belong.
    unsigned entries() const { return entries_; }

    // 0..1 through the phase's `max`, or 0 when it has none. For the UI.
    float phaseProgress() const;
    const ActiveText& text() const { return text_; }

    // Why the last transition happened, for the UI and for logs.
    const std::string& lastReason() const { return reason_; }

private:
    struct CueState {
        bool started = false;
        bool finished = false;
        double t0 = 0.0;    // phase time the cue started at
    };

    void enter(Phase p, const std::string& reason);
    void updateText();
    bool eventLevel(Event e) const;

    ShowScript script_;
    Phase phase_ = Phase::Idle;
    double t_ = 0.0;
    unsigned entries_ = 0;
    std::string reason_ = "start";

    Signals sig_;
    bool scene_done_ = false;

    float held_[kMaxEdges] = {};      // per-edge continuous-true accumulator
    std::vector<CueState> cue_state_;
    ActiveText text_;
};

// Where scripts live, alongside the root presets.
std::string ShowDir();
std::vector<std::string> ListShows();

}  // namespace show
