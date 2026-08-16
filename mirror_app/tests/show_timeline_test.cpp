// show_timeline_test — the running order, headlessly.
//
// The timeline's failure modes are all invisible until they happen in the room
// and are then unreproducible: a phase that advances one frame early because a
// floor was checked against the wrong clock, a debounce that a single dropped
// tracker frame defeats, a caption whose fade-out never starts because its
// duration was shorter than its fade. None of that shows in a screenshot and
// none of it is worth discovering with a person standing in front of the piece,
// so the whole state machine is driven here at a fixed step instead.
//
// The graph is code, so it is checked as code: the tests below assert the shape
// of the piece (idle -> fitting -> transition -> roots -> idle) rather than
// merely that some table was parsed.

#include "show_timeline.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

// Frames at 60 Hz, with the signals held.
void run(show::Timeline& tl, double seconds, show::Signals sig) {
    const double dt = 1.0 / 60.0;
    const int n = (int)std::lround(seconds / dt);
    for (int i = 0; i < n; ++i) {
        tl.setSignals(sig);
        tl.advance(dt);
    }
}

show::ShowScript parse(const char* text) {
    show::ShowScript s;
    std::string err;
    check(show::ParseShow(text, s, err), "script parses");
    if (!err.empty()) std::printf("  parse error: %s\n", err.c_str());
    return s;
}

}  // namespace

int main() {
    using namespace show;

    const Signals kNothing;
    Signals kFace;  kFace.face_present = true;
    Signals kFit;   kFit.face_present = true; kFit.fit_converged = true;

    // --- the graph is the piece ---------------------------------------------
    {
        // Every phase can be left, or the installation can strand.
        for (int i = 0; i < (int)Phase::Count; ++i) {
            const PhaseGraph& g = Graph((Phase)i);
            check(g.edge_count > 0 || g.max_time > 0.f,
                  "every phase has a way out");
            check(g.edge_count <= kMaxEdges, "edge count fits PhaseScript::hold");
            for (int e = 0; e < g.edge_count; ++e)
                check(g.edges[e].target != (Phase)i, "no edge to itself");
        }
        // The forward path, which is what the operator's "go" takes.
        check(Graph(Phase::Idle).edges[0].target == Phase::Fitting, "idle -> fitting");
        check(Graph(Phase::Fitting).edges[0].target == Phase::Transition,
              "fitting -> transition");
        check(Graph(Phase::Transition).edges[0].target == Phase::Roots,
              "transition -> roots");
        check(Graph(Phase::Roots).edges[0].target == Phase::Idle, "roots -> idle");
        // Fitting checks the fit before the absence: a fit landing as the
        // person steps back reads as a capture.
        check(Graph(Phase::Fitting).edges[0].event == Event::FitConverged &&
                  Graph(Phase::Fitting).edges[1].event == Event::FaceAbsent,
              "fitting prefers the capture");
        check(Graph(Phase::Fitting).timeout == Phase::Idle, "a failed fit goes home");
    }

    // --- an empty script is the designed piece ------------------------------
    {
        ShowScript d;   // no file at all
        for (int i = 0; i < (int)Phase::Count; ++i) {
            const Phase p = (Phase)i;
            check(near(d[p].min_time, Graph(p).min_time, 1e-6f), "min from the graph");
            check(near(d[p].max_time, Graph(p).max_time, 1e-6f), "max from the graph");
            for (int e = 0; e < Graph(p).edge_count; ++e)
                check(near(d[p].hold[e], Graph(p).edges[e].hold, 1e-6f),
                      "hold from the graph");
            check(d[p].cues.empty(), "no captions unless a script asks");
        }

        // And it runs end to end.
        Timeline tl(d);
        run(tl, 12.0, kFace);
        check(tl.phase() == Phase::Fitting, "a face moves it to fitting");
        run(tl, 4.0, kFit);
        check(tl.phase() == Phase::Transition, "a converged fit hands off");
        tl.sceneDone();
        run(tl, 0.1, kFit);
        check(tl.phase() == Phase::Roots, "the transition ends itself");
        run(tl, 60.0, kNothing);
        check(tl.phase() == Phase::Idle, "it resets when the room empties");
    }

    // --- the format ---------------------------------------------------------
    {
        ShowScript s = parse(R"(
show: t

phase idle
	min: 5s
	face_hold: 0.5s
	text: "HELLO"
		at: 1s
		for: 4s
		fade: 0.5s
)");
        check(s.name == "t", "show name");
        check(near(s[Phase::Idle].min_time, 5.f, 1e-6f), "min parsed");
        check(near(s[Phase::Idle].hold[0], 0.5f, 1e-6f), "edge debounce parsed");
        check(s[Phase::Idle].cues.size() == 1, "one cue");
        check(s[Phase::Idle].cues[0].text == "HELLO", "cue text");
        check(near(s[Phase::Idle].cues[0].dur, 4.f, 1e-6f), "cue duration");
        // Untouched phases keep the graph's defaults.
        check(near(s[Phase::Roots].min_time, Graph(Phase::Roots).min_time, 1e-6f),
              "an unnamed phase is unchanged");

        // The colon is optional and indentation is decorative: the same script
        // written flat, without colons, parses identically.
        ShowScript flat = parse(
            "show t\n"
            "phase idle\n"
            "min 5s\n"
            "face_hold 0.5s\n"
            "text HELLO\n"
            "at 1s\n"
            "for 4s\n"
            "fade 0.5s\n");
        check(near(flat[Phase::Idle].min_time, 5.f, 1e-6f), "colonless min");
        check(near(flat[Phase::Idle].hold[0], 0.5f, 1e-6f), "colonless hold");
        check(flat[Phase::Idle].cues.size() == 1 &&
                  flat[Phase::Idle].cues[0].text == "HELLO" &&
                  near(flat[Phase::Idle].cues[0].dur, 4.f, 1e-6f),
              "unindented cue keys still attach to their cue");

        // Comments, ms suffixes, escaped newlines, quoted values.
        ShowScript s2 = parse(R"(
# a comment
phase roots
	min: 250ms   # trailing comment
	text: "TWO\nLINES"
	text: BARE VALUE
)");
        check(near(s2[Phase::Roots].min_time, 0.25f, 1e-6f), "ms suffix");
        check(s2[Phase::Roots].cues[0].text == "TWO\nLINES", "escaped newline");
        check(s2[Phase::Roots].cues[1].text == "BARE VALUE", "unquoted value");

        // A second `text` closes the first, so its keys cannot leak.
        ShowScript s3 = parse(
            "phase roots\n"
            "  text: A\n"
            "    for: 2s\n"
            "  text: B\n"
            "    for: 5s\n");
        check(near(s3[Phase::Roots].cues[0].dur, 2.f, 1e-6f), "first cue keeps its own");
        check(near(s3[Phase::Roots].cues[1].dur, 5.f, 1e-6f), "second cue takes the rest");

        // Errors carry a line number and never throw.
        ShowScript bad;
        std::string err;
        check(!ParseShow("phase idle\n  min: nope\n", bad, err), "bad duration fails");
        check(err.rfind("line 2:", 0) == 0, "error carries the line number");
        check(!ParseShow("min: 5s\n", bad, err), "a key before any phase fails");
        check(!ParseShow("phase nowhere\n", bad, err), "unknown phase fails");
        check(!ParseShow("phase idle\n  fit_hold: 1s\n", bad, err),
              "another phase's edge key fails");
        check(err.find("face_hold") != std::string::npos,
              "and the error names what idle does have");
        check(!ParseShow("phase idle\n  for: 2s\n", bad, err),
              "a cue key with no cue open fails");
    }

    // --- floors: an event cannot fire before `min` --------------------------
    {
        Timeline tl(parse("phase idle\n  min: 5s\n  face_hold: 1s\n"));
        run(tl, 4.0, kFace);
        check(tl.phase() == Phase::Idle, "held below the floor despite the event");
        run(tl, 1.5, kFace);
        check(tl.phase() == Phase::Fitting, "fires once the floor is cleared");
        // The hold ran concurrently with the floor rather than after it: a face
        // present the whole time advances at `min`, not at min + hold.
        check(tl.phaseTime() < 1.0, "no double wait");
    }

    // --- debounce: a dropped tracker frame is not somebody leaving ----------
    {
        Timeline tl(parse("phase roots\n  min: 0s\n  absent_hold: 3s\n"));
        tl.goTo(Phase::Roots);
        run(tl, 2.0, kNothing);
        check(tl.phase() == Phase::Roots, "not yet");
        run(tl, 1.0 / 60.0 * 2, kFace);      // two frames of re-detection
        run(tl, 2.5, kNothing);
        check(tl.phase() == Phase::Roots, "the accumulator reset on re-detection");
        run(tl, 1.0, kNothing);
        check(tl.phase() == Phase::Idle, "leaves after a clean 3s absence");
    }

    // --- ceilings, and edge priority on the same frame ----------------------
    {
        Timeline tl(parse("phase fitting\n  min: 0s\n  max: 10s\n  fit_hold: 0s\n"));
        tl.goTo(Phase::Fitting);
        run(tl, 12.0, kFace);
        check(tl.phase() == Phase::Idle, "timeout sends a failed fit home");

        Timeline tl2(parse("phase fitting\n  min: 0s\n  max: 10s\n  fit_hold: 0s\n"));
        tl2.goTo(Phase::Fitting);
        run(tl2, 0.2, kFit);
        check(tl2.phase() == Phase::Transition, "the event beats the ceiling");
    }

    // --- scene_done, and that it bypasses the floor -------------------------
    {
        Timeline tl(parse("phase transition\n  min: 30s\n"));
        tl.goTo(Phase::Transition);
        run(tl, 1.0, kNothing);
        check(tl.phase() == Phase::Transition, "still running");
        tl.sceneDone();
        run(tl, 0.1, kNothing);
        check(tl.phase() == Phase::Roots,
              "a finished scene is not held back by the phase floor");
    }

    // --- the operator ------------------------------------------------------
    {
        Timeline tl(parse("phase idle\n  min: 600s\n"));
        run(tl, 1.0, kNothing);
        tl.go();
        check(tl.phase() == Phase::Fitting, "go takes the forward edge past the floor");
        tl.go();
        check(tl.phase() == Phase::Transition, "and means the same thing everywhere");

        const unsigned e0 = tl.entries();
        run(tl, 0.5, kNothing);
        check(tl.entries() == e0, "no entry without a transition");
        tl.goTo(Phase::Roots);
        check(tl.entries() == e0 + 1, "goTo counts as an entry");
        check(tl.phaseTime() == 0.0, "and resets the clock");
    }

    // --- text cues ----------------------------------------------------------
    {
        Timeline tl(parse("phase idle\n  text: ONE\n    at: 1s\n    for: 4s\n"
                          "    fade: 1s\n"));
        run(tl, 0.9, kNothing);
        check(!tl.text().on, "not before its time");
        run(tl, 0.6, kNothing);   // t = 1.5, half a second in
        check(tl.text().on && tl.text().text == "ONE", "on and correct");
        check(near(tl.text().reveal, 0.5f, 0.05f), "fading in");
        run(tl, 1.0, kNothing);   // t = 2.5, mid-hold
        check(near(tl.text().reveal, 1.f, 0.02f), "fully revealed");
        run(tl, 2.0, kNothing);   // t = 4.5, half a second from the end
        check(near(tl.text().reveal, 0.5f, 0.05f), "fading out");
        run(tl, 1.0, kNothing);   // t = 5.5, past the end
        check(!tl.text().on, "gone after its duration");
    }
    {
        // A cue shorter than two fades still peaks and comes back down instead
        // of being clipped by a fade-in that never finishes.
        Timeline tl(parse("phase idle\n  text: X\n    for: 1s\n    fade: 2s\n"));
        run(tl, 0.5, kNothing);
        check(near(tl.text().reveal, 1.f, 0.05f), "short cue peaks at its midpoint");
        run(tl, 0.4, kNothing);
        check(tl.text().reveal < 0.4f, "and falls away again");
    }
    {
        // Anchored to an event, with `after` as the delay past it.
        Timeline tl(parse("phase fitting\n  min: 60s\n"
                          "  text: CAPTURED\n    when: fit_converged\n"
                          "    after: 0.5s\n    for: 3s\n    fade: 0.1s\n"));
        tl.goTo(Phase::Fitting);
        run(tl, 2.0, kFace);
        check(!tl.text().on, "waits for its anchor");
        run(tl, 0.3, kFit);
        check(!tl.text().on, "and then for its delay");
        run(tl, 0.5, kFit);
        check(tl.text().on && tl.text().text == "CAPTURED", "anchored cue fires");
    }
    {
        // A phase change clears whatever was on screen.
        Timeline tl(parse("phase idle\n  min: 0s\n  face_hold: 0s\n  text: ONE\n"));
        run(tl, 0.1, kFace);
        check(tl.phase() == Phase::Fitting, "left immediately");
        check(!tl.text().on, "the caption did not survive the phase change");
    }

    // --- reload keeps the phase, restart does not ---------------------------
    {
        Timeline tl(parse("phase roots\n  min: 40s\n"));
        tl.goTo(Phase::Roots);
        run(tl, 2.0, kNothing);
        tl.setScript(parse("phase roots\n  min: 1s\n"));
        check(tl.phase() == Phase::Roots, "a reload retimes what is running");
        check(tl.phaseTime() == 0.0, "with a fresh clock");
        tl.restart();
        check(tl.phase() == Phase::Idle, "restart goes to the top of the piece");
    }

    if (failures == 0) std::printf("show_timeline_test: OK\n");
    else std::printf("show_timeline_test: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
