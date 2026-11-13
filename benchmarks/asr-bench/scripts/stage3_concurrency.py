#!/usr/bin/env python3
"""Stage-3 style concurrency harness: Kokoro TTS vs Parakeet STT.

Mirrors slug's stage3_bench.py: idle baselines, 1:1 strict concurrency, and
floods in both directions, so the deltas are comparable with slug's 2026-08
numbers. TTS = resident Kokoro service on :8093; STT = resident Parakeet
service whose URL is passed as the first argument.

usage: stage3_concurrency.py <stt_url> [audio_path]
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

TTS_URL = "http://127.0.0.1:8093/v1/audio/speech"
TEXT = "The voice service is running on r9slug."


def log(msg):
    line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)


def stt_once(path):
    boundary = "----asrbench"
    with open(path, "rb") as f:
        payload = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
    body += payload + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        STT_URL, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r:
        r.read()
    return time.perf_counter() - t0


def tts_once(stream=False):
    body = {"model": "kokoro", "input": TEXT, "voice": "af_bella"}
    if not stream:
        body["response_format"] = "wav"
    req = urllib.request.Request(
        TTS_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    ttfa = None
    total = 0
    with urllib.request.urlopen(req, timeout=180) as r:
        while True:
            chunk = r.read(4096)
            if not chunk:
                break
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            total += len(chunk)
    return time.perf_counter() - t0, ttfa, total


def bench(fn, n, label):
    lat = [fn() for _ in range(n)]
    log("%-30s n=%d  mean=%.3fs  min=%.3fs  max=%.3fs"
        % (label, n, statistics.mean(lat), min(lat), max(lat)))
    return statistics.mean(lat)


def concurrent(fn_a, fn_b):
    """Run one A and one B strictly simultaneously.

    Returns ((a_elapsed, a_return), (b_elapsed, b_return)).
    """
    out = {}

    def run(name, fn):
        t0 = time.perf_counter()
        ret = fn()
        out[name] = (time.perf_counter() - t0, ret)

    ta = threading.Thread(target=run, args=("a", fn_a))
    tb = threading.Thread(target=run, args=("b", fn_b))
    ta.start(); tb.start()
    ta.join(); tb.join()
    return out["a"], out["b"]


def flood_tts(n, fn_stt):
    """n concurrent TTS requests, with one STT request issued among them."""
    barrier = threading.Barrier(n + 1)
    results = {}

    def tts_worker():
        barrier.wait()
        t0 = time.perf_counter()
        tts_once()
        results.setdefault("tts", []).append(time.perf_counter() - t0)

    threads = [threading.Thread(target=tts_worker) for _ in range(n)]
    for t in threads:
        t.start()
    barrier.wait()
    results["stt"] = [fn_stt()]
    for t in threads:
        t.join()
    return results


def flood_stt(n, fn_tts):
    """n concurrent STT requests, with one TTS request issued among them."""
    barrier = threading.Barrier(n + 1)
    results = {}

    def stt_worker():
        barrier.wait()
        t0 = time.perf_counter()
        stt_once(AUDIO)
        results.setdefault("stt", []).append(time.perf_counter() - t0)

    threads = [threading.Thread(target=stt_worker) for _ in range(n)]
    for t in threads:
        t.start()
    barrier.wait()
    total, ttfa, _ = fn_tts()
    results["tts_total"] = total
    results["tts_ttfa"] = ttfa
    for t in threads:
        t.join()
    return results


STT_URL = sys.argv[1]
AUDIO = sys.argv[2] if len(sys.argv) > 2 else "/data/clips/000_1272-128104-0000.wav"

log("STT_URL=%s AUDIO=%s" % (STT_URL, AUDIO))

# ---- idle baselines ----
stt_idle = bench(lambda: stt_once(AUDIO), 5, "STT idle")
tts_idle_total = []
tts_idle_ttfa = []
for _ in range(5):
    total, ttfa, _ = tts_once(stream=True)
    tts_idle_total.append(total)
    tts_idle_ttfa.append(ttfa)
log("%-30s n=5  mean=%.3fs  min=%.3fs  max=%.3fs"
    % ("TTS idle total (stream)", statistics.mean(tts_idle_total),
       min(tts_idle_total), max(tts_idle_total)))
log("%-30s n=5  mean=%.3fs  min=%.3fs  max=%.3fs"
    % ("TTS idle TTFA", statistics.mean(tts_idle_ttfa),
       min(tts_idle_ttfa), max(tts_idle_ttfa)))

# ---- test 1: 1 STT + 1 TTS strictly simultaneous ----
(c_stt, _), (c_tts_elapsed, (c_tts_total, c_tts_ttfa)) = concurrent(
    lambda: stt_once(AUDIO), lambda: tts_once(stream=True)[:2])
log("%-30s mean=%.3fs (idle %.3fs, %+.1f%%)"
    % ("STT concurrent", c_stt, stt_idle, 100 * (c_stt / stt_idle - 1)))
log("%-30s mean=%.3fs (idle %.3fs, %+.1f%%)"
    % ("TTS concurrent total", c_tts_total, statistics.mean(tts_idle_total),
       100 * (c_tts_total / statistics.mean(tts_idle_total) - 1)))
log("%-30s mean=%.3fs (idle %.3fs, %+.1f%%)"
    % ("TTS concurrent TTFA", c_tts_ttfa, statistics.mean(tts_idle_ttfa),
       100 * (c_tts_ttfa / statistics.mean(tts_idle_ttfa) - 1)))

# ---- test 2: TTS flood x5, STT among them ----
f = flood_tts(5, lambda: stt_once(AUDIO))
log("%-30s mean=%.3fs (idle %.3fs, %+.1f%%)"
    % ("STT during TTS flood x5", statistics.mean(f["stt"]), stt_idle,
       100 * (statistics.mean(f["stt"]) / stt_idle - 1)))
log("%-30s mean=%.3fs" % ("TTS flood wall (per req)", statistics.mean(f["tts"])))

# ---- test 3: STT flood x5, TTS among them ----
g = flood_stt(5, lambda: tts_once(stream=True))
log("%-30s mean=%.3fs (idle %.3fs, %+.1f%%)"
    % ("TTS during STT flood x5 total", g["tts_total"], statistics.mean(tts_idle_total),
       100 * (g["tts_total"] / statistics.mean(tts_idle_total) - 1)))
log("%-30s mean=%.3fs (idle %.3fs, %+.1f%%)"
    % ("TTS during STT flood x5 TTFA", g["tts_ttfa"], statistics.mean(tts_idle_ttfa),
       100 * (g["tts_ttfa"] / statistics.mean(tts_idle_ttfa) - 1)))
log("%-30s mean=%.3fs" % ("STT flood wall (per req)", statistics.mean(g["stt"])))
