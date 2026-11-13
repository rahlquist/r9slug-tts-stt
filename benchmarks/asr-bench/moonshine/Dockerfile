FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=6
RUN pip install "moonshine-voice==0.1.6" "jiwer==4.0.0"
COPY scripts/moonshine_prewarm.py /prewarm.py
RUN python /prewarm.py
COPY scripts/bench_moonshine.py /bench.py
CMD ["python", "/bench.py"]
