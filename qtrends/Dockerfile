FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN printf '# qtrends\n' > README.md \
    && mkdir -p src/qtrends \
    && touch src/qtrends/__init__.py
RUN python -m pip install --upgrade pip \
    && python -m pip install ".[dev]"

COPY README.md ./
COPY src ./src
RUN python -m pip install --no-deps --force-reinstall .

COPY configs ./configs
COPY examples ./examples
COPY tests ./tests
COPY data ./data
COPY outputs ./outputs

ENTRYPOINT ["qtrends"]
CMD ["demo", "--config", "configs/sample.yaml", "--output-dir", "outputs/demo"]
