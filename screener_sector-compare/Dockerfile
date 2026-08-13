FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SCREENER_CONFIG_DIR=/app/config

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 screener

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir \
    'fastapi>=0.116' \
    'filelock>=3.13' \
    'hmmlearn>=0.3.2' \
    'jinja2>=3.1' \
    'numpy>=1.26,<3' \
    'pandas>=2.1,<3' \
    'pyarrow>=15' \
    'pyyaml>=6.0' \
    'scikit-learn>=1.4' \
    'typer>=0.12' \
    'uvicorn>=0.35' \
    'xgboost-cpu>=3.4' \
    'yfinance>=0.2.65'

COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY config ./config
COPY --chmod=755 scripts/docker-entrypoint.sh /usr/local/bin/screener-entrypoint
RUN mkdir -p /app/cache /app/outputs && chown -R screener:screener /app

USER root
ENTRYPOINT ["/usr/local/bin/screener-entrypoint"]
CMD ["screener", "--help"]

FROM runtime AS test

USER root
COPY tests ./tests
RUN pip install --no-cache-dir '.[dev]'
USER screener
ENTRYPOINT ["pytest"]
CMD ["-q"]
