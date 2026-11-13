FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=6 HF_HOME=/models/hf
RUN pip install "onnx-asr[cpu,hub]==0.12.0"
COPY scripts/parakeet_prewarm.py /prewarm.py
RUN python /prewarm.py
COPY scripts/bench_parakeet.py /bench.py
CMD ["python", "/bench.py"]
