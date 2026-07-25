#!/usr/bin/env python3
"""Local zero-shot voice-cloning server for kinect_v2_demo.

Holds one MLX TTS model resident and synthesises text in a cloned voice on
demand. `src/demo/voice_clone.mm` is the client; nothing else here is
demo-specific, so it is equally usable with curl.

Why a server rather than a library call from C++: every zero-shot cloning model
worth using is a Python artefact reaching the GPU through MLX, which has no C++
inference path for them. Keeping the model in its own long-lived process also
means weight loading (seconds to tens of seconds) happens once, off the demo's
render loop, and the model can be swapped or restarted without a rebuild.

    ./tools/voice_server.py                          # Chatterbox, default port
    ./tools/voice_server.py --model chatterbox-turbo # faster, less expressive
    ./tools/voice_server.py --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16

    curl -s localhost:8765/health
    curl -s localhost:8765/speak -H 'content-type: application/json' \
         -d '{"text":"hello","reference_wav":"/tmp/ref.wav"}' -o out.wav

Model swapping is by design (see MODELS and _supported_kwargs): the parameter
names differ between families -- Chatterbox has `exaggeration`/`cfg_weight`,
Qwen3-TTS wants a `ref_text` transcript -- so rather than hard-coding one
model's vocabulary, the request carries neutral names and they are mapped onto
whatever the loaded model's `generate` actually accepts.

Binds to 127.0.0.1 only. Requests carry a filesystem path to a recording of
whoever is standing in front of the sensor and the response is a synthesis of
their voice; that is not traffic to put on a network interface by default.
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import os
import queue
import struct
import sys
import threading
import time
import traceback
import wave
from concurrent import futures
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Presets are shorthands, not a whitelist: --model also takes any HF repo id or
# local path that mlx-audio can load.
#
# RTF below is measured on an M4 (10-core, 32 GB), warm, cloning from a 6.9 s
# reference, generating ~3 s of speech. Under 1.0 is faster than real time.
MODELS = {
    # Emotion exaggeration control, ~5 s reference, MIT. The default: the most
    # expressive of these, and the exaggeration knob is the reason to want it.
    # Measured RTF 1.19-1.33 -- a ~3.8 s wait for a 3 s utterance.
    "chatterbox": "mlx-community/chatterbox-fp16",
    # Same family, 350M. Measured RTF 0.63-0.79, i.e. comfortably faster than
    # real time, at the cost of a noticeably flatter read. Use this one if the
    # interaction has to feel immediate.
    "chatterbox-turbo": "mlx-community/chatterbox-turbo-fp16",
    # 4-bit, for when memory matters more than fidelity. Unmeasured.
    "chatterbox-4bit": "mlx-community/chatterbox-4bit",
    # Needs `ref_text` (a transcript of the reference clip) alongside ref_audio.
    # Unmeasured here.
    "qwen3-tts": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
}

DEFAULT_MODEL = "chatterbox"


class Engine:
    """Owns the model, on one dedicated thread.

    The single worker is not just about serialising access -- MLX streams are
    **per-thread**, so evaluating a graph on a thread that did not create the
    stream fails outright with "There is no Stream(gpu, 0) in current thread."
    ThreadingHTTPServer hands every request a fresh thread, so generation cannot
    happen inline. Loading and every generate() therefore run on this one
    long-lived thread, and request handlers hand work to it and wait.

    Serialisation is the right policy anyway: MLX graphs are not reentrant, and
    concurrent requests on one GPU only trade latency for latency.
    """

    def __init__(self, model_id: str, lazy: bool = False):
        self.model_id = MODELS.get(model_id, model_id)
        self.model_alias = model_id
        self.model = None
        self.load_error: str | None = None
        self._gen_params: set[str] | None = None

        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._worker = threading.Thread(
            target=self._run, name="tts-worker", daemon=True
        )
        self._worker.start()

        if not lazy:
            # Block until the model is up, so startup reports load failures
            # rather than deferring them to the first request.
            self._submit(self._load).result()

    # -- worker plumbing --

    def _run(self):
        while True:
            fn, args, future = self._queue.get()
            if fn is None:
                return
            try:
                future.set_result(fn(*args))
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
                future.set_exception(exc)

    def _submit(self, fn, *args) -> "futures.Future":
        future = futures.Future()
        self._queue.put((fn, args, future))
        return future

    # -- model --

    def _load(self) -> bool:
        if self.model is not None:
            return True
        try:
            from mlx_audio.tts.utils import load_model

            t0 = time.time()
            print(f"loading {self.model_id} ...", file=sys.stderr, flush=True)
            model = load_model(self.model_id)
            self._gen_params = _supported_kwargs(model)
            self.model = model
            print(
                f"loaded in {time.time() - t0:.1f}s; generate() accepts: "
                f"{sorted(self._gen_params)}",
                file=sys.stderr,
                flush=True,
            )
            self.load_error = None
            self._warmup()
            return True
        except Exception as exc:  # noqa: BLE001 - reported to the client verbatim
            self.load_error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            return False

    def _warmup(self):
        """Generate once, into the void, before serving anyone.

        Loading weights is not the whole cost: the first generation also builds
        and compiles the MLX graph. Measured on an M4 with chatterbox-turbo,
        that makes request #1 take 12.9 s where #2 takes 1.9 s -- an RTF of 4.6
        against a steady-state 0.63. Paying it here means the first thing a user
        asks for is not seven times slower than everything after it.

        Best-effort: a warm-up that fails is not a reason to refuse to serve,
        since the real request may well succeed.
        """
        try:
            t0 = time.time()
            # Nothing but `text`: the optional parameters differ per model
            # (chatterbox takes `verbose`, chatterbox-turbo does not), and the
            # warm-up has no reason to exercise any of them.
            for _ in self.model.generate(text="Warming up."):
                pass
            print(
                f"warmed up in {time.time() - t0:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"warm-up failed (harmless): {exc}", file=sys.stderr, flush=True)

    def generate(self, req: dict) -> tuple[bytes, int, float]:
        """Returns (wav_bytes, sample_rate, seconds_of_audio).

        Validates on the calling thread so a bad request fails fast, then runs
        the model on the worker.
        """
        text = (req.get("text") or "").strip()
        if not text:
            raise ValueError("empty text")

        ref = req.get("reference_wav") or None
        if ref:
            ref = os.path.abspath(os.path.expanduser(ref))
            if not os.path.isfile(ref):
                raise FileNotFoundError(f"reference audio not found: {ref}")

        return self._submit(self._generate, req, text, ref).result()

    def _generate(self, req: dict, text: str, ref: str | None):
        if self.model is None and not self._load():
            raise RuntimeError(self.load_error or "model failed to load")

        # Neutral request vocabulary -> whatever this model actually named it.
        # Anything the model does not accept is dropped rather than passed
        # through as **kwargs, which some models reject outright.
        #
        # The families genuinely disagree about emotion. Chatterbox takes a
        # scalar (`exaggeration`); Qwen3-TTS takes a *sentence* (`instruct`,
        # e.g. "speak warmly and slowly"). There is no honest conversion
        # between them, so both are carried and each model picks up the one it
        # understands -- see the `style` field in the request.
        candidates = {
            "ref_audio": ref,
            "audio_prompt": ref,
            "exaggeration": _f(req, "exaggeration", 0.5),
            "cfg_weight": _f(req, "cfg", 0.5),
            "cfg_scale": _f(req, "cfg", 0.5),
            "temperature": _f(req, "temperature", 0.8),
            # Transcript of the reference clip. Qwen3-TTS needs it to clone;
            # Chatterbox has no such parameter and it is dropped for that model.
            "ref_text": req.get("reference_text") or None,
            "instruct": req.get("style") or None,
            "verbose": False,
        }
        kwargs = {
            k: v
            for k, v in candidates.items()
            if v is not None and (self._gen_params is None or k in self._gen_params)
        }
        # `audio_prompt` and `ref_audio` are aliases on Chatterbox; sending both
        # is harmless but sending neither is not, so make the omission loud.
        if ref and not any(k in kwargs for k in ("ref_audio", "audio_prompt")):
            raise RuntimeError(
                f"{self.model_id} exposes no reference-audio parameter, so it "
                "cannot clone a voice"
            )
        # Qwen3-TTS conditions on the reference *transcript* as well as the
        # audio. Without it the clone still renders, just worse -- a silent
        # quality regression, so say so rather than let it be mysterious.
        if (
            ref
            and self._gen_params
            and "ref_text" in self._gen_params
            and "ref_text" not in kwargs
        ):
            print(
                "warning: this model takes ref_text (a transcript of the "
                "reference clip) and none was supplied; the clone will be "
                "worse than it needs to be",
                file=sys.stderr,
                flush=True,
            )

        t0 = time.time()
        chunks = []
        sample_rate = 24000
        for result in self.model.generate(text=text, **kwargs):
            audio = getattr(result, "audio", None)
            if audio is None:
                continue
            sample_rate = getattr(result, "sample_rate", sample_rate) or sample_rate
            chunks.append(_to_float_list(audio))
        elapsed = time.time() - t0

        if not chunks:
            raise RuntimeError("model produced no audio")

        samples = [s for chunk in chunks for s in chunk]
        seconds = len(samples) / float(sample_rate)
        print(
            f"generated {seconds:.2f}s in {elapsed:.2f}s "
            f"(RTF {elapsed / max(seconds, 1e-6):.2f}) for {len(text)} chars",
            file=sys.stderr,
            flush=True,
        )
        return _wav_bytes(samples, sample_rate), sample_rate, seconds


def _supported_kwargs(model) -> set[str] | None:
    """Parameter names `model.generate` accepts, or None if it takes **kwargs
    and will tolerate anything."""
    try:
        sig = inspect.signature(model.generate)
    except (TypeError, ValueError):
        return None
    names = set()
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            # Still return the explicit names: passing an unknown kwarg through
            # **kwargs usually reaches a submodule that raises anyway.
            continue
        names.add(name)
    return names


def _f(req: dict, key: str, default: float) -> float:
    try:
        return float(req.get(key, default))
    except (TypeError, ValueError):
        return default


def _to_float_list(audio) -> list[float]:
    """Flattens whatever the model returned (mx.array, numpy, list) to floats."""
    try:
        import numpy as np

        arr = np.asarray(audio, dtype="float32").reshape(-1)
        return arr.tolist()
    except Exception:  # noqa: BLE001
        return [float(x) for x in audio]


def _wav_bytes(samples: list[float], sample_rate: int) -> bytes:
    """16-bit mono PCM. The client reads and plays this directly."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        pcm = bytearray()
        for s in samples:
            # Clamp: models occasionally overshoot [-1, 1], and wrapping that
            # into int16 turns a peak into full-scale noise.
            v = int(max(-1.0, min(1.0, float(s))) * 32767.0)
            pcm += struct.pack("<h", v)
        w.writeframes(bytes(pcm))
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None  # set on the server instance below

    def log_message(self, fmt, *args):  # quieter than the default
        pass

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] != "/health":
            self._json(404, {"error": "not found"})
            return
        eng = self.engine
        self._json(
            200,
            {
                "ok": True,
                "model": eng.model_id,
                "alias": eng.model_alias,
                "model_loaded": eng.model is not None,
                "load_error": eng.load_error,
                "presets": sorted(MODELS),
            },
        )

    def do_POST(self):
        if self.path.split("?")[0] != "/speak":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError) as exc:
            self._json(400, {"error": f"bad request body: {exc}"})
            return

        try:
            wav, rate, seconds = self.engine.generate(req)
        except (ValueError, FileNotFoundError) as exc:
            self._json(400, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.send_header("X-Sample-Rate", str(rate))
        self.send_header("X-Audio-Seconds", f"{seconds:.3f}")
        self.end_headers()
        self.wfile.write(wav)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"preset ({', '.join(sorted(MODELS))}) or any HF repo id / local path",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--lazy",
        action="store_true",
        help="load weights on the first request instead of at startup",
    )
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        for alias, repo in sorted(MODELS.items()):
            print(f"{alias:20s} {repo}")
        return 0

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"warning: binding to {args.host} exposes recorded voices and their "
            "clones beyond this machine",
            file=sys.stderr,
        )

    engine = Engine(args.model, lazy=args.lazy)
    if engine.model is None and not args.lazy:
        print(
            f"model did not load: {engine.load_error}\n"
            "serving anyway -- /health reports the error and /speak will retry.",
            file=sys.stderr,
        )

    handler = type("BoundHandler", (Handler,), {"engine": engine})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"voice server on http://{args.host}:{args.port}  model={engine.model_id}",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
