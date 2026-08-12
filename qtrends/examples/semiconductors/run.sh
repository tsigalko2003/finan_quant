#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${example_dir}/../.." && pwd)"
cd "${repo_dir}"

config="examples/semiconductors/config.yaml"
export_target="data/example_nasdaq_semiconductors.csv"
manifest="data/universes/example_semiconductors.csv"
output_dir="outputs/example_semiconductors"

cleanup() {
  docker compose down --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker compose build

docker compose run --rm qtrends generate-sample \
  --output data/example_semiconductors_prices.csv \
  --periods 1250 \
  --seed 42

# Start from a clean example-only manifest so every run demonstrates both snapshots.
rm -f "${manifest}"
install -m 0644 examples/semiconductors/nasdaq_snapshot_1.csv "${export_target}"

echo
echo "Step 1: seed the Semiconductor universe from the initial Nasdaq-style export"
docker compose run --rm qtrends sync-universe \
  --config "${config}" \
  --snapshot-date 2020-01-02

install -m 0644 examples/semiconductors/nasdaq_snapshot_2.csv "${export_target}"

echo
echo "Step 2: apply the later export; EPSILON is incrementally added"
docker compose run --rm qtrends sync-universe \
  --config "${config}" \
  --snapshot-date 2021-07-01

echo
echo "Step 3: run breadth + PCA + HMM + XGBoost using the persisted manifest"
docker compose run --rm qtrends run \
  --config "${config}" \
  --output-dir "${output_dir}" \
  --no-universe-refresh

echo
echo "Step 4: validate the membership and generated artifacts"
docker compose run --rm --entrypoint python qtrends \
  examples/semiconductors/inspect_results.py
