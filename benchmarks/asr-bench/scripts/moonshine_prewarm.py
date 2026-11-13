"""Bake Moonshine en models (tiny/base/medium streaming) into the image."""
from pathlib import Path
from moonshine_voice import get_model_for_language, ModelArch, Transcriber
import inspect

CACHE = Path("/models/ms")
CACHE.mkdir(parents=True, exist_ok=True)

for arch in (ModelArch.TINY, ModelArch.BASE, ModelArch.TINY_STREAMING,
             ModelArch.SMALL_STREAMING, ModelArch.MEDIUM_STREAMING):
    path, ret_arch = get_model_for_language("en", arch, cache_root=CACHE)
    print("cached", arch.name, "->", path)

print("SIG transcribe_without_streaming:", inspect.signature(Transcriber.transcribe_without_streaming))
print("SIG create_stream:", inspect.signature(Transcriber.create_stream))
print("prewarm OK")
