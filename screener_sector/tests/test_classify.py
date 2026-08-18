from pathlib import Path

from screener_sector.universe.classify import (
    ThemeRules,
    enrichment_candidates,
    is_in_scope,
    match_themes,
)

CONFIG_DIR = Path("/app/config")


def rules():
    return ThemeRules.load(CONFIG_DIR)


def test_rules_load_from_yaml():
    r = rules()
    assert "Semiconductors" in r.industry_allow_list
    assert "semiconductor" in r.theme_keywords
    assert "SOXX" in r.seed_etfs


def test_matches_semiconductor_theme():
    themes = match_themes(
        "Applied Materials Inc.",
        "Provides wafer fabrication equipment for the semiconductor industry.",
        rules(),
    )
    assert "semiconductor" in themes


def test_matches_optical_theme():
    themes = match_themes(
        "Applied Optoelectronics",
        "Designs optical transceiver modules and laser diode products.",
        rules(),
    )
    assert "optical" in themes


def test_matches_ai_compute_theme():
    themes = match_themes(
        "NVIDIA Corporation",
        "Designs GPUs and accelerator platforms for data center AI workloads.",
        rules(),
    )
    assert "ai_compute" in themes


def test_matches_multiple_themes():
    themes = match_themes(
        "Broadcom Inc.",
        "Semiconductor supplier of ASIC accelerators and co-packaged optics.",
        rules(),
    )
    assert set(themes) >= {"semiconductor", "ai_compute", "optical"}


def test_no_match_for_unrelated_company():
    themes = match_themes(
        "Coca-Cola Company",
        "Manufactures and distributes non-alcoholic beverages worldwide.",
        rules(),
    )
    assert themes == ()


def test_keyword_matching_is_word_boundary_aware():
    themes = match_themes(
        "Ceda Holdings", "A general holding company with no chip exposure.", rules()
    )
    assert "design_tools" not in themes


def test_plural_keyword_still_matches():
    themes = match_themes(
        "Some Corp", "We build GPUs for rendering.", rules()
    )
    assert "ai_compute" in themes


def test_in_scope_via_industry_even_without_keywords():
    assert is_in_scope("Semiconductors", "Mystery Corp", "No description.", rules())


def test_in_scope_via_keywords_even_with_odd_industry():
    assert is_in_scope(
        "Specialty Business Services",
        "Photonics Co",
        "Builds silicon photonics engines.",
        rules(),
    )


def test_out_of_scope_when_neither_matches():
    assert not is_in_scope(
        "Beverages", "Coca-Cola", "Sells soft drinks.", rules()
    )


def test_enrichment_candidates_keeps_name_matches_and_seeds():
    """The prefilter that makes prod discovery survive Yahoo's rate limits."""
    import pandas as pd

    symbols = pd.DataFrame(
        {
            "ticker": ["AMAT", "COHR", "KO", "XYZ", "SOXX", "PHOT"],
            "name": [
                "Applied Materials Inc.",
                "Coherent Corp.",
                "Coca-Cola Co",
                "Generic Holdings Inc",
                "iShares Semiconductor ETF",
                "Bright Photonics Corp",
            ],
            "exchange": ["NASDAQ"] * 6,
            "etf": [False, False, False, False, True, False],
        }
    )
    candidates = enrichment_candidates(symbols, rules())

    assert "PHOT" in candidates          # kept on its name alone
    assert "COHR" in candidates          # name reveals nothing; kept via seed list
    assert "SOXX" in candidates          # benchmark, kept via seed_etfs
    assert "KO" not in candidates
    assert "XYZ" not in candidates
    assert candidates == sorted(set(candidates))


def test_enrichment_candidates_cuts_volume_substantially():
    """Motivation for the prefilter: 8000 profile requests get rate limited."""
    import pandas as pd

    rows = [{"ticker": f"T{i}", "name": f"Generic Business {i}"} for i in range(200)]
    for i in range(10):
        rows[i]["name"] = f"Acme Semiconductor {i}"
    symbols = pd.DataFrame(rows)
    symbols["exchange"] = "NASDAQ"
    symbols["etf"] = False

    candidates = enrichment_candidates(symbols, rules())
    assert len(candidates) < len(symbols) * 0.25


def test_seed_tickers_are_all_strings():
    """Guards the bare-ON YAML boolean trap."""
    assert all(isinstance(t, str) for t in rules().seed_tickers)
    assert "ON" in rules().seed_tickers
