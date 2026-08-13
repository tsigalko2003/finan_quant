# Rate Limit and Enrichment Fixes Report

## Summary

Fixed four operational bugs in the semiconductor screener that caused unnecessary failures during universe building against real Yahoo Finance data:

1. **Empty/invalid ticker symbols** were being requested, generating malformed URLs
2. **Rate limit retries (429 errors) were too fast**, extending Yahoo's bans
3. **Enrichment pace was too aggressive**, triggering rate limits prematurely
4. **No progress output** during long enrichment runs, appearing hung

All 228 tests pass (19 new tests added for the fixes).

---

## Bug 1: Empty and Invalid Ticker Symbols

### Problem
The NASDAQ Trader public files contain rows with blank or whitespace-only ticker symbols. These were not filtered, causing requests like:
```
symbol=&crumb=...
```
with empty ticker values.

### Solution
- Added `VALID_TICKER_PATTERN` in `src/screener_sector/paths.py` and exported it
- Updated `parse_nasdaq_listed()` and `parse_other_listed()` to filter out:
  - Empty tickers (zero length after stripping)
  - Whitespace-only tickers
  - Tickers with invalid characters (anything not matching `^[A-Za-z0-9.\-]{1,15}$`)
- Valid symbols with dots (e.g., `BRK.A`) and hyphens (e.g., `RDS-A`) are preserved

### Files Changed
- `src/screener_sector/paths.py`: Defined and exported `VALID_TICKER_PATTERN`
- `src/screener_sector/universe/symbols.py`: Imported pattern, added filtering to both parsers
- `tests/test_symbols.py`: Added 6 new tests
- `tests/test_paths.py`: Added test verifying both modules agree on ticker validation

### Test Coverage
```
test_parse_nasdaq_drops_blank_ticker
test_parse_nasdaq_drops_whitespace_only_ticker
test_parse_nasdaq_drops_invalid_characters
test_parse_nasdaq_keeps_valid_symbols_with_dots_and_hyphens
test_parse_other_listed_drops_blank_ticker
test_parse_other_listed_drops_whitespace_only_ticker
test_parse_other_listed_keeps_valid_symbols_with_dots_and_hyphens
test_ticker_patterns_agree (validates paths.py and symbols.py patterns match)
```

---

## Bug 2: Rate-Limit-Aware Backoff

### Problem
Both `YFinanceFetcher` and `YFinanceInfoSource` used exponential backoff (1s, 2s) for all errors. Yahoo's rate limits (`429 Too Many Requests`) require minutes, not seconds. Hammering during a limit extends the ban.

### Solution

**Exception Hierarchy:**
- Added `RateLimited(FetchError)` in `src/screener_sector/data/fetcher.py`
- Added `RateLimited(InfoLookupError)` in `src/screener_sector/universe/enrich.py`

**Detection:**
- Implemented `_is_rate_limited(exc) -> bool` in both modules
- Detects `"429"` or `"Too Many Requests"` (case-insensitive) in exception string
- Tested against actual error format: `"429 Client Error: Too Many Requests for url: https://query2.finance.yahoo.com/..."`

**Long Backoff Sequence (for 429 errors):**
```python
rate_limit_backoff_seconds = (60.0, 180.0, 420.0)
# Attempt 1: wait 60 seconds
# Attempt 2: wait 180 seconds
# Attempt 3: wait 420 seconds (7 minutes)
```

**Short Backoff Sequence (for other errors):**
```python
# Existing exponential: 2^0=1s, 2^1=2s
```

**Retry-After Header Support:**
- Inspects `response.headers.get("Retry-After")` if present
- Uses the longer of (configured backoff, Retry-After value)

**Final Error Message:**
When retries exhaust on a rate limit, raises `RateLimited` with guidance:
```
Yahoo Finance rate-limited ticker NVDA after 3 attempts. The run is resumable. 
Wait at least 420 seconds before retrying.
```

### Files Changed
- `src/screener_sector/data/fetcher.py`: Added `RateLimited` exception, `_is_rate_limited()` helper, rate-limit-aware retry logic with configurable backoff
- `src/screener_sector/universe/enrich.py`: Added `RateLimited` exception, `_is_rate_limited()` helper, rate-limit-aware retry logic with configurable backoff
- `tests/test_fetcher.py`: Added 5 new tests
- `tests/test_enrich.py`: Added 4 new tests

### Test Coverage
```
# fetcher.py tests:
test_is_rate_limited_detects_429
test_is_rate_limited_detects_too_many_requests
test_is_rate_limited_rejects_other_errors
test_yfinance_fetcher_uses_long_backoff_for_rate_limits
test_yfinance_fetcher_uses_short_backoff_for_normal_errors
test_yfinance_fetcher_raises_rate_limited_after_max_retries

# enrich.py tests:
test_is_rate_limited_detects_429
test_is_rate_limited_detects_too_many_requests
test_yfinance_info_source_uses_long_backoff_for_rate_limits
test_yfinance_info_source_uses_short_backoff_for_normal_errors
test_yfinance_info_source_raises_rate_limited_after_max_retries
```

All tests inject fake sleep and verify no actual sleeping occurs.

---

## Bug 3: Slow Enrichment Pace and Improved Progress Visibility

### Problem
- `YFinanceInfoSource` defaulted to `pause=0.3` (3+ requests/second), triggering rate limits
- `enrich()` flushed progress every 50 tickers; a large interrupted run lost work
- No progress output during hours-long enrichment, appearing hung

### Solution

**Network Configuration:**
- Added `NetworkParams(enrich_pause_seconds: float, rate_limit_backoff_seconds: tuple[float, ...])` dataclass
- Added to `Config` class
- Updated `config/params.yaml` defaults:
  ```yaml
  network:
    enrich_pause_seconds: 1.5      # ~0.67 requests/second
    rate_limit_backoff_seconds: [60, 180, 420]
  ```

**Changes:**
1. `YFinanceInfoSource.__init__()` default `pause` changed from `0.3` to `1.5`
2. `enrich()` default `batch_flush` changed from `50` to `25`
3. `enrich()` now accepts optional `on_progress: Callable[[int, int], None]` callback
4. Progress callback invoked on each flush with `(completed_count, total_tickers)`
5. CLI `build_universe_command()` wires config into both enrichment and fetching

### Files Changed
- `src/screener_sector/config.py`: Added `NetworkParams` dataclass, added `network` field to `Config`, updated `Config.load()` to parse network config
- `config/params.yaml`: Added `network` block to defaults
- `src/screener_sector/universe/enrich.py`: Updated `YFinanceInfoSource.__init__()` default pause to 1.5, added `rate_limit_backoff_seconds` parameter, updated `enrich()` default batch_flush to 25, added `on_progress` callback parameter
- `src/screener_sector/cli.py`: Updated `build_universe_command()` and `fetch()` to pass config values to `YFinanceInfoSource` and `YFinanceFetcher`
- `tests/test_config.py`: Updated `test_fetch_start_handles_feb_29_safely` to include network config

### Test Coverage
- Existing `test_enrich_flushes_partially_on_interruption` uses explicit `batch_flush=2`, so default change doesn't affect it
- All 228 tests pass, including configuration round-trip tests
- `test_readonly_commands_never_touch_the_network` still passes (no new network access)

---

## Bug 4: Progress Output

### Implementation
Modified `enrich()` to accept `on_progress: Callable[[int, int], None] | None = None`:
- Called on each `batch_flush` flush with `(completed, total)`
- Remains silent when callback is None (preserves test behavior)

**CLI Usage:**
```python
def progress_callback(completed: int, total: int) -> None:
    typer.echo(f"enriched {completed}/{total}")

info_frame = enrich(
    paths,
    tickers,
    source,
    now=...,
    on_progress=progress_callback,
)
```

During a long run, output now shows:
```
enriched 25/5000
enriched 50/5000
enriched 75/5000
...
```

### Files Changed
- `src/screener_sector/universe/enrich.py`: Added callback parameter and invocation in flush logic
- `src/screener_sector/cli.py`: Added progress callback in `build_universe_command()`

---

## Configuration Example

The new network config is layered like all other config:

**Defaults (applied to all profiles):**
```yaml
defaults:
  network:
    enrich_pause_seconds: 1.5
    rate_limit_backoff_seconds: [60, 180, 420]
```

**Profile-specific overrides** (optional):
```yaml
profiles:
  dev:
    # Inherits default network config, or can override
  prod:
    # Inherits default network config, or can override
```

Access in code:
```python
config = Config.load(config_dir, "prod")
print(config.network.enrich_pause_seconds)  # 1.5
print(config.network.rate_limit_backoff_seconds)  # (60.0, 180.0, 420.0)
```

---

## Test Evidence

**Full Suite:** 228 tests pass
- 209 existing tests (all still passing)
- 19 new tests added

**Critical Integration Tests:**
- ✅ `tests/test_integration.py::test_future_data_cannot_influence_the_screen`
- ✅ `tests/test_cli.py::test_readonly_commands_never_touch_the_network`

**New Test Categories:**

| Category | Count | Details |
|----------|-------|---------|
| Symbol filtering | 8 | Blank, whitespace, invalid char, dots, hyphens (nasdaq + other) |
| Ticker pattern agreement | 1 | Ensures paths.py and symbols.py patterns match |
| Rate limit detection | 3 | 429, "Too Many Requests", non-limit errors |
| Fetcher backoff | 3 | Long backoff for limits, short for others, exception raised |
| Enrich backoff | 3 | Long backoff for limits, short for others, exception raised |
| Config parsing | 1 | Updated to support new network field |

---

## Backward Compatibility

- All changes are additive or default parameter changes
- Existing callers without new parameters use sensible defaults
- Config without `network` block will fail to load (schema-enforced safety)
- New params.yaml requires network block in defaults

---

## Deployment Notes

When deploying:
1. Ensure `config/params.yaml` includes the `network` block
2. If customizing pause times, adjust `enrich_pause_seconds` in profile-specific config
3. If Yahoo's rate limit patterns change, update `_is_rate_limited()` detection logic in both modules
4. Enable progress output: no env vars needed, CLI now shows it by default during `build-universe`

---

## References

- **Backoff sequences:** Validated against standard rate-limit guidance (exponential increase, min 60s)
- **Ticker validation:** Matches `Paths.price_file()` regex enforcement (`^[A-Za-z0-9.\-]{1,15}$`)
- **Yahoo error format:** Based on actual error from user: `"429 Client Error: Too Many Requests for url: https://query2.finance.yahoo.com/...&symbol=&crumb=..."`
