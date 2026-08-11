FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxerces-c3.2 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /bench
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Smoke test: bit-reproducible reference run (expects: collision 102.59 m)
# docker run <img> python sim/benchmark_runner.py --map map2_crossing \
#   --mode mixed --algorithm dwa --seed 1 --max-time 200
CMD ["python", "sim/benchmark_runner.py", "--help"]
