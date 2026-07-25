# GPU contention: voice synthesis vs. the mirror scene

Measured 25 July 2026 on **Apple M4, 10-core (4P/6E), 32 GB**, macOS 15.6.1.

The question: can `kinect_v2_validate`'s voice cloning
(`kinect_v2_validate/tools/voice_server.py`) run alongside `mirror_app` without
hitching the render loop?

**Answer: it hitches the neural mirror badly and the root scene not at all.**
Synthesise during the root scene, not the pond.

## Why there is contention at all

Both are MLX workloads on the *same* GPU:

- `mirror_app`'s neural mirror is `mlp_forward` — `mx::fast::metal_kernel` on the
  system default `MTLDevice` (see `mirror_app/src/mlp_forward.h`).
- The voice server is mlx-audio, on that same device, in another process.

Separate processes do not help: there is one GPU, and during synthesis two
saturating MLX compute workloads split it roughly evenly.

## Numbers

`mirror_app --bench 2 200` (neural mirror, 960×540) and `--rootbench 2 120`
(sphere-tracer, 960×540), with the voice server driven by a continuous loop of
`/speak` requests cloning from a 6.9 s reference.

| workload | idle GPU | during TTS | change |
| --- | --- | --- | --- |
| **neural mirror**, `chatterbox` | 26.16 ms (38 fps) | **40.38 ms (25 fps)** | **+54%** |
| neural mirror, `chatterbox` (repeat) | 26.16 ms | 40.04 ms (25 fps) | +53% |
| neural mirror, `chatterbox-turbo` | 26.16 ms | 36.48 ms (27 fps) | +39% |
| **root renderer** | 1.045 ms (957 fps) | 1.066 ms (938 fps) | +2% (noise) |
| CPlantBox `advance()` (CPU) | 0.278 ms | 0.299 ms | +8% (noise) |

Voice server during generation: **33% of one core, 3.2 GB RSS.** The work is on
the GPU, not the CPU — the CPU-side sim and the CoreAudio callback are untouched.

### Why the root scene is unaffected

It is ~1 ms of fragment work with ~15 ms of idle GPU per frame, so synthesis
fills the gaps. The pond has no gaps: it saturates the GPU for 26 ms, so
anything else is straight contention. This is about *headroom*, not about
compute-vs-raster.

## Consequences

1. **Synthesise during the root scene or the transition.** Free there.
2. **`chatterbox-turbo` roughly thirds the disruption** if overlap is
   unavoidable: less contended (36 vs 40 ms) *and* a shorter window (1.8 s vs
   3.8 s of generation).
3. **Pre-generate when the lines are known.** The server already returns WAVs;
   synthesising at startup and playing files costs nothing at runtime.

## Caveats

- Headless benches against a loaded GPU. Two real windowed processes add
  compositing cost on top of this.
- Transcription was **not** measured. `SFSpeechRecognizer` runs on the ANE/CPU
  rather than the GPU, so it should not appear in these numbers, but that is
  reasoning, not a measurement — it needs the sensor and a live mic to check.

## Reproducing

```sh
# baseline
./build/mirror_app/mirror_app --bench 2 200

# contended: start the server, wait for model_loaded, then loop /speak
kinect_v2_validate/.venv-voice/bin/python \
    kinect_v2_validate/tools/voice_server.py --model chatterbox &
until curl -s localhost:8765/health | grep -q '"model_loaded": true'; do sleep 4; done
( for i in $(seq 1 12); do curl -s localhost:8765/speak \
    -H 'content-type: application/json' \
    -d '{"text":"...","reference_wav":"/path/ref.wav"}' -o /dev/null; done ) &
./build/mirror_app/mirror_app --bench 2 200
```
