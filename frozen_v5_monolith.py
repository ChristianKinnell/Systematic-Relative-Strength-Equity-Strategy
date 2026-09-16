"""
LONG-ONLY MOMENTUM MODEL:
Long-only momentum / relative-strength model.

  • Universe         Systematic Yahoo Finance screen (region=US, NYSE/NASDAQ,
                     market cap > MARKET_CAP_MIN) → US-domicile filter (drops
                     foreign ADRs the region filter lets through — Canadian/
                     UK/EU banks, foreign oil majors, TSM, ASML, etc. — since
                     region = US reflects listing venue, not company domicile)
                     → price/ADV/coverage/history filter → trend+RS+vol
                     confirmation gates.

                     SURVIVORSHIP BIAS: this screen runs against TODAY's
                     market caps/domiciles and is then applied retroactively
                     back to CFG['start']. Walk-forward/OOS testing does NOT
                     fix this — it still only ever tests today's surviving
                     large-caps, not the names that would actually have been
                     screened-in on each historical date (some of which have
                     since been acquired, delisted, or fallen out of the
                     cap band). Yahoo Finance has no point-in-time
                     constituent history to build a proper PIT universe
                     from, so this is disclosed rather than solved. Treat
                     results here as strategy research, not an institutional-
                     quality historical performance record.

  • Signal           Score = 0.15·M1m + 0.30·M3m + 0.30·M6m + 0.15·M12m + 0.10·RS,
                     each momentum leg skipping the most recent 5 trading days
                     (a 5-trading-day skip) to cut short-term reversal contamination,
                     vol-adjusted, then EWM-smoothed (halflife 5d). Alongside
                     the live score, a report-only diagnostics block computes
                     rank-IC ablations (M6 alone → full 4-horizon → +vol-adj
                     → +RS → +smoothing) and a rank-vs-winsorized-zscore
                     comparison against forward returns — informational only,
                     never fed back into the live score.

  • Factor health    Cross-sectional high-minus-low momentum tercile spread
                     (R_MOM,t), monitored via its trailing 3m return and
                     drawdown, mapped to a 100%/75%/50% exposure scale — but
                     this is diagnostic-only (report-only, [3c/8]), not wired
                     into live sizing; see Risk, below, for why it was reverted.

  • Trend gates      price>50DMA, 50DMA>200DMA, RS>0 vs S&P500, vol ceiling —
                     hard filters, not blended into the score.

  • HMM regime       GaussianHMM(3 states) on SPX/VIX/LQD daily changes,
                     EXPANDING-WINDOW refit (~quarterly, min 1y train) —
                     standardised per-window, not on the full sample. States
                     are labelled risk_on/neutral/risk_off by each state's
                     mean forward SPX return — a market-favorability ranking,
                     not a volatility-based risk bucket, so treat the names
                     as directional labels, not literal vol regimes. State
                     labels for each out-of-sample segment come from a causal
                     forward filter (P(S_t | X_1..X_t), see
                     hmm_filtered_states()), not model.predict()'s Viterbi
                     decode — Viterbi over a whole segment lets later days in
                     that segment influence the state assigned to earlier ones,
                     which is look-ahead even though the fit itself is already
                     expanding-window. A separate macro overlay (VIX vs its
                     50d average, 5d Treasury yield change, 5d LQD return)
                     can override the HMM label to risk_on/risk_off, and a
                     fast VIX-spike/credit-stress check can force "stress"
                     regardless of HMM/macro. Today only "stress" changes
                     gross exposure (50%) — risk_on/neutral/risk_off don't
                     currently scale exposure differently; see the report-only
                     regime ablation for whether damping risk_off would help.
                     Used only for gross-exposure control, not to tilt the
                     signal toward mean-reversion.

  • Position count   Top 7 by score among names already passing Confirmation
                     (which is where M3m>0/M6m>0 absolute-momentum actually
                     gets enforced — not here), further restricted to
                     score > score_threshold so a name has to clear both the
                     absolute-momentum gate AND a real positive cross-
                     sectional score (not just above-median, which with
                     hundreds of candidates is almost always true) — unfilled
                     slots are held as cash, never forced into a "least bad"
                     7th name. invested_fraction = n_qualified/top_n, so
                     conviction breadth scales gross exposure without a
                     separate model for it.

  • Sizing           Inverse-volatility weighting (default) — with 7 names and
                     a small book, a covariance/Kelly estimate from ~63 daily
                     observations adds estimation error a simple inverse-vol
                     scheme doesn't. A CVaR (Rockafellar-Uryasev LP, CVXPY) +
                     downside-Kelly engine is kept and fixed (correct R-U
                     confidence-level denominator) but is opt-in via
                     CFG['sizing_method']='cvar_kelly', for if/when capital or
                     position count grow enough to justify it. Inverse-vol
                     sizes RELATIVE weights between names; it doesn't cap
                     TOTAL portfolio risk. A single-target vol-targeting
                     variant (Exposure = min(1, vol_target/realised_vol))
                     exists in run_gate_ablation() as report-only, for A/B
                     testing against alternative targets. The CFG['vol_low']/
                     ['vol_high'] hysteresis-band variant (see Risk, below)
                     IS wired into live sizing (portfolio construction loop,
                     section 6) — promoted after the report-only ablation
                     (section 6j) showed the band, not just measured it.

  RISK:
  Portfolio-level and security-level risk are separate layers, both
  distinct from Alpha/Confirmation (which decide WHAT to hold) and from
  Execution/Validation (which happen after Risk has already acted).

  • Portfolio risk   Per-name cap (max_weight=15%) and sector cap
                     (sector_cap=45%) bound concentration; the vol ceiling
                     (Trend gates, above) rejects extremely volatile names
                     before they're ever eligible; regime/stress scaling cuts
                     gross exposure in adverse conditions. Momentum factor
                     health (above) is diagnostic-only (report-only, [3c/8])
                     — it was briefly wired live (50%/75% exposure cut) without
                     going through the test-first/promote-when-told gate used
                     for every other feature this session, then reverted:
                     found firing "crash" on 74.8% of all days including the
                     live-run date, because its internal drawdown trigger
                     shares the circuit breaker's original bug (all-time-peak
                     reference, never reset) AND the top-vs-bottom momentum
                     spread it's built on runs at ann. Sharpe -2.3 in this
                     construction — unresolved whether that's a genuine
                     finding about factor decay in this universe or a timing/
                     alignment bug (separate from alpha_final, which the
                     actual stock selection uses and isn't implicated). Do
                     not re-promote without investigating that spread and
                     applying the same peak-reset fix as the circuit breaker.
                     Below that sits a second, more
                     severe layer: a drawdown circuit breaker — dd_stop=-8%
                     is a genuine NAV_t/HWM_t-1 drawdown-from-peak trigger
                     (it used to silently mean a rolling ~7-day return
                     instead, despite the name — fixed so dd_stop means what
                     it says), freezing the book to 0% exposure for exactly
                     dd_cooldown=3 days, then resuming UNCONDITIONALLY —
                     resume used to require dd_peak recovering past
                     dd_resume=-2% measured against the ALL-TIME peak. Two
                     compounding bugs there: (1) at 0% exposure the book
                     earns ~0% return, so dd_peak can never improve on its
                     own — a hard deadlock; (2) even once resume was made
                     time-based, comparing the resumed book's drawdown
                     against a pre-crash all-time peak meant it almost
                     always re-triggered within one rebalance cycle anyway,
                     since a single week rarely recovers a >6% drawdown
                     against a now-unreachable old high — same practical
                     effect as the deadlock. Confirmed via backtest: the
                     model was stuck at ~0% exposure continuously from
                     2019-05-23 to the live dry-run date, over 7 years, the
                     first time this ever triggered. Fixed by resetting the
                     reference peak (dd_peak_ref) to the CURRENT equity on
                     every resume, so the NEXT freeze only fires on a NEW
                     decline from the post-resume level — the trigger check
                     also now only evaluates while NOT frozen and not on
                     the resume day itself, so a freshly-resumed book
                     always gets one real rebalance before its own
                     drawdown is re-assessed. So there are
                     two tiers: regime stress → 50% exposure, circuit
                     breaker → 0% exposure. A THIRD tier now sits
                     alongside those two: portfolio volatility targeting,
                     CFG['vol_low']=14%/['vol_high']=24% as a hysteresis
                     band (same shape as dd_stop/dd_resume) — once trailing
                     21d realised vol exceeds vol_high, exposure is scaled
                     by min(1, vol_high/realised_vol) until it drops back
                     below vol_low. This was report-only in section 6j; now
                     live in the portfolio construction loop (section 6) —
                     it was previously declared in CFG but never referenced
                     anywhere. Unfilled slots are held as cash (Position
                     count, above), so exposure also falls naturally when
                     qualifying names run out — a fourth, passive form of
                     risk reduction. Sector caps bound sector-LABEL
                     concentration, not correlation — two different sectors
                     can still move together, so a correlation/PC1/variance-
                     contribution check runs alongside market-factor beta and CVaR
                     (Validation, below) as monitoring, same as those — not
                     an automatic veto yet. See the report-only risk-
                     parameter and ATR-stop sensitivity blocks for whether
                     dd_stop/max_weight/sector_cap/atr_stop_mult are set
                     well; vol_low/vol_high are live, revisit if the DRY-run
                     shows an issue.

  • Security risk    ATR-based reference stop shown per held name in the
                     target-portfolio report — last_close − atr_stop_mult×
                     ATR(14) — for manual reference only. No brokerage
                     connection exists in this version, so nothing is placed
                     or monitored automatically; a separate ATR trailing-
                     stop is also simulated inside the report-only
                     run_gate_ablation() path for backtest/ablation purposes,
                     but does not touch the live target weights. The
                     optional CVaR/Kelly sizing engine (Sizing, above) also
                     carries an explicit CVaR_95% tail-loss constraint at
                     the portfolio-construction stage, independent of this
                     reference stop.

  EXECUTION:
  • Signal timing    T+1 enforced structurally in the loop: a day's decision
                     uses that day's close, and is applied starting only the
                     NEXT trading day's return — never the same close.

  • Costs            Fixed spread + square-root market-impact approximation
                     (10 bps spread PER trade — a full entry+exit round trip
                     is ~20bps of fixed cost before impact — impact ~ σ√(V/ADV)),
                     sized
                     off CFG['backtest_nav'] — a fixed reference NAV (not a
                     live-fetched account balance). Applied to the live/
                     backtest weight path (section 7) and, per candidate,
                     inside the walk-forward grid search (8f) as well, so
                     config selection there is net-of-cost rather than
                     ranking on gross returns and only pricing costs in
                     afterward.

  • Output           This is a research/signal script, not an order-
                     submission system — there is no brokerage integration
                     of any kind. It ends by printing a target portfolio
                     (weights, scores, momentum/RS/vol readouts, the
                     informational ATR stop reference above) and a
                     reconciliation against PORTFOLIO_UNIVERSE (current
                     holdings), for manual execution. Order placement,
                     adaptive limit offsets, partial-fill handling, and
                     TWAP/VWAP/participation scheduling are all out of
                     scope for this version.

  • Prices           Adjusted close (yfinance auto_adjust=True) throughout —
                     signals and realised P&L both use a price you could
                     actually execute at, not a synthetic (H+L+C)/3 average.

  • Fundamental veto Sanity/veto gate applied to the final target just
                     before sizing (see fundamental_veto()) — NOT an alpha
                     signal, never ranks or scores names. Checks solvency,
                     earnings deterioration, leverage, and cash runway for
                     unprofitable names; flags (does not exclude) binary/
                     event-driven names like clinical-stage biotech, since
                     momentum alone still drives selection. Not applied to
                     the historical backtest — yfinance has no point-in-time
                     fundamentals to backtest it against.

  VALIDATION:
  • Stability        Time-sliced Sharpe stability check across combinatorial
                     history slices, using the model's own already-causal
                     weights. This is explicitly NOT the Bailey et al. CPCV/
                     PBO (which needs multiple candidate configs) or a real
                     Deflated Sharpe Ratio (needs trial count + skew/kurtosis)
                     — those were previously reported under those names
                     without actually being them; that's been corrected.
                     Sharpe/Sortino/Calmar/CVaR, turnover and realised
                     slippage are all reported alongside this (sections 8-9
                     of the runtime pipeline below) — Validation judges the
                     finished strategy, after Risk has already shaped it and
                     Execution has already cost it. Portfolio Sharpe alone
                     doesn't prove the alpha score predicts returns, so
                     section 8b adds rank-IC by horizon (1w/1m/3m, mean/
                     std/IC-IR/hit-rate), a 5/10/20/40/60d signal-decay
                     curve, and forward-return quintile monotonicity —
                     alpha-specific evidence, not portfolio-level proxies
                     for it. Section 8c fills the one remaining parameter-
                     sensitivity gap (skip period; top_n/vol-ceiling/ATR/
                     momentum-weighting already covered in 3b/6b/6e/6i).
                     Section 8d is a STACKED module ablation (momentum ->
                     +RS -> +confirmation -> +regime -> +inverse-vol ->
                     +risk controls). Sections 8e/8f are walk-forward OOS:
                     8e tests the current fixed config across expanding
                     annual folds (temporal consistency, no re-selection);
                     8f is a genuine train->optimise->freeze->test grid
                     search, deliberately bounded to top_n/vol-ceiling/ATR
                     (the cheapest already-parameterised dimensions), ranking
                     and testing candidates on NET-of-cost returns (fixed
                     spread + sqrt market-impact approximation applied per
                     candidate's own turnover, not just to
                     the final reported live path) — full re-fitting of
                     momentum weights and HMM/regime params per fold is a
                     separate, substantially larger build, not attempted
                     here. Bootstrap confidence intervals and
                     genuine PBO/Deflated Sharpe remain explicitly not
                     built — the latter only becomes worthwhile once many
                     competing configs have actually been run (which 8f now
                     starts doing), per the review's own sequencing.
"""

import signal
import warnings
warnings.filterwarnings("ignore")

import itertools
import json
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy.linalg import lstsq
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from hmmlearn import hmm as hmmlib

# CONFIGURATION
CFG = dict(
    start            = "2018-01-01", 
    warmup_days      = 260,          
    top_n            = 7,            
    score_threshold  = 0.0,
    sizing_method    = "inverse_vol",
    rebal_band       = 0.08,
    dd_stop          = -0.08,        
    dd_resume        = -0.02,
    dd_cooldown      = 3,
    kelly_fraction   = 0.25,
    max_weight       = 0.15,         
    min_weight       = 0.10,         
    vol_low          = 0.14,         
    vol_high         = 0.24,
    cpcv_n           = 6,            
    cpcv_k           = 2,            
    embargo_days     = 5,            
    hmm_states       = 3,            
    hmm_restarts     = 8,            
    hmm_min_train    = 252,          
    hmm_refit_every  = 63,           
    cvar_confidence  = 0.95,         
    cvar_limit       = 0.030,        
    lambda_turnover  = 0.035,        
    lambda_alpha     = 0.10,         
    sector_cap       = 0.45,         
    stress_scale     = 0.50,
    atr_period       = 14,
    atr_stop_mult    = 2.5,
    backtest_nav     = 100_000,  
)

TNX_LIVE_FALLBACK = 4.507

def _retry(fn, attempts=3, base_delay=1.5):
    """Retry fn() with exponential backoff; re-raises the last exception if all attempts fail."""
    last_exc = RuntimeError("_retry called with attempts < 1")
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_exc

def get_live_tnx():
    try:
        tnx_live = _retry(lambda: yf.download("^TNX", period="1d", progress=False), attempts=3, base_delay=2.0)
        if not tnx_live.empty:
            return float(tnx_live["Close"].iloc[-1])
    except Exception:
        pass
    return TNX_LIVE_FALLBACK

# CURRENT HOLDINGS:
PORTFOLIO_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_state.json")

def _load_portfolio_universe():
    try:
        with open(PORTFOLIO_STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

PORTFOLIO_UNIVERSE = _load_portfolio_universe()

# OPTIONAL WATCHLIST RESTRICTION:
UNIVERSE_WATCHLIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "universe_watchlist.json")

def _load_universe_watchlist():
    try:
        with open(UNIVERSE_WATCHLIST_PATH, "r") as f:
            return set(json.load(f))
    except Exception:
        return set()

UNIVERSE_WATCHLIST = _load_universe_watchlist()
MARKET_CAP_MIN = 25_000_000_000
UNIVERSE_META_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         ".universe_meta_cache.json")
UNIVERSE_META_TTL_DAYS = 180
def _load_universe_meta_cache():
    try:
        with open(UNIVERSE_META_CACHE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_universe_meta_cache(cache):
    try:
        with open(UNIVERSE_META_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"Warning: could not persist universe metadata cache ({e})")

def build_universe():
    try:
        query = yf.EquityQuery('and', [
            yf.EquityQuery('eq', ['region', 'us']),
            yf.EquityQuery('is-in', ['exchange', 'NMS', 'NYQ']),
            yf.EquityQuery('gt', ['intradaymarketcap', MARKET_CAP_MIN]),
        ])
        found, offset, page_size = [], 0, 250
        while True:
            res    = _retry(lambda: yf.screen(query, size=page_size, offset=offset,
                                               sortField='intradaymarketcap', sortAsc=False),
                             attempts=3, base_delay=3.0)
            quotes = res.get('quotes', [])
            found.extend(q['symbol'] for q in quotes if q.get('symbol'))
            offset += page_size
            if not quotes or offset >= res.get('total', 0):
                break
        found = sorted(set(found))
        if len(found) < 200:
            raise ValueError(f"only found {len(found)} names — screener response looks wrong")

        print(f"Universe source: Yahoo Finance screener — US, NYSE/NASDAQ,"
              f"market cap > ${MARKET_CAP_MIN:,.0f} ({len(found)} names, pre-domicile-filter)")

        meta_cache = _load_universe_meta_cache()
        now_iso    = pd.Timestamp.utcnow().isoformat()
        cutoff     = pd.Timestamp.utcnow() - pd.Timedelta(days=UNIVERSE_META_TTL_DAYS)

        def _is_fresh(entry):
            try:
                return pd.Timestamp(entry["fetched_at"]) >= cutoff
            except Exception:
                return False

        to_fetch = [s for s in found if not _is_fresh(meta_cache.get(s, {}))]
        print(f"Fetching sector/domicile metadata for {len(to_fetch)} names not yet cached "
              f"or past the {UNIVERSE_META_TTL_DAYS}-day TTL "
              f"({len(found) - len(to_fetch)} served from cache) ...")

        n_failed = 0
        for i, sym in enumerate(to_fetch):
            try:
                info    = _retry(lambda: yf.Ticker(sym).info, attempts=3, base_delay=1.5)
                sector  = info.get("sector") or "Unknown"
                country = info.get("country") or "Unknown"
                meta_cache[sym] = {"sector": sector, "country": country, "fetched_at": now_iso}
            except Exception as e:
                print(f"     Warning: failed to fetch metadata for {sym} ({e})")
                n_failed += 1
            if (i + 1) % 50 == 0:
                print(f"     ... {i + 1}/{len(to_fetch)} lookups done")
                _save_universe_meta_cache(meta_cache)

        _save_universe_meta_cache(meta_cache)
        if n_failed:
            print(f"   Warning: {n_failed}/{len(to_fetch)} metadata lookups failed after retries "
                  f"— excluded from this run's universe, will retry next run.")

        universe = {}
        n_foreign = 0
        for sym in found:
            entry = meta_cache.get(sym)
            if entry is None or entry.get("country") != "United States":
                n_foreign += 1
            else:
                universe[sym] = {"sector": entry["sector"], "cost_bps": 0.001}

        print(f"   Domicile filter: dropped {n_foreign} foreign-domiciled/unknown names "
              f"({len(universe)} US-domiciled remain)")
        if len(universe) < 150:
            raise ValueError(f"only {len(universe)} US-domiciled names after filter — looks wrong")

        return universe
    except Exception as e:
        raise RuntimeError(f"Could not build Yahoo screener universe ({e}). "
                            f"Refusing to fall back to a static universe — fix the "
                            f"screener and re-run.") from e

# FUNDAMENTAL SANITY / VETO LAYER
FUNDAMENTAL_VETO_CACHE = {}

def fundamental_veto(sym):
    if sym in FUNDAMENTAL_VETO_CACHE:
        return FUNDAMENTAL_VETO_CACHE[sym]
    try:
        info = yf.Ticker(sym).info
    except Exception as e:
        result = ("flag", f"fundamentals lookup failed ({e}) — trading on momentum alone")
        FUNDAMENTAL_VETO_CACHE[sym] = result
        return result

    reasons = []
    verdict = "ok"

    total_cash    = info.get("totalCash")
    net_income    = info.get("netIncomeToCommon")
    op_cf         = info.get("operatingCashflow")
    fcf           = info.get("freeCashflow")
    debt_to_eq    = info.get("debtToEquity")
    current_ratio = info.get("currentRatio")
    quick_ratio   = info.get("quickRatio")
    sector        = info.get("sector") or ""
    industry      = info.get("industry") or ""
    earn_qgrowth  = info.get("earningsQuarterlyGrowth")

    # 1. Financial distress / solvency — current/quick ratio assume a normal
    # operating-company balance sheet (inventory, receivables, short-term
    # liabilities) and aren't meaningful for banks/financials, whose balance
    # sheets are structured around deposits/loans instead.
    is_financial = (sector == "Financial Services")
    if not is_financial:
        if current_ratio is not None and current_ratio < 1.0:
            reasons.append(f"current ratio {current_ratio:.2f} < 1.0")
            verdict = "exclude"
        if quick_ratio is not None and quick_ratio < 0.5:
            reasons.append(f"quick ratio {quick_ratio:.2f} — weak near-term liquidity")
            verdict = "exclude"

    # 2. Severe earnings deterioration
    if earn_qgrowth is not None and earn_qgrowth < -0.50:
        reasons.append(f"YoY quarterly earnings growth {earn_qgrowth:.0%}")
        if verdict == "ok":
            verdict = "flag"

    # 3. Excessive leverage
    if debt_to_eq is not None and debt_to_eq > 300:
        reasons.append(f"debt/equity {debt_to_eq:.0f}% — highly levered")
        if verdict == "ok":
            verdict = "flag"

    # 4. Binary/event-driven exposure 
    if sector == "Healthcare" and "Biotechnology" in industry and (net_income or 0) < 0:
        reasons.append("clinical-stage/event-driven biotech with negative net income — "
                        "returns likely catalyst-driven (trial/FDA), not price persistence")
        if verdict == "ok":
            verdict = "flag"

    # 5. Cash runway for unprofitable companies
    if net_income is not None and net_income < 0:
        burn = None
        if op_cf is not None and op_cf < 0:
            burn = -op_cf
        elif fcf is not None and fcf < 0:
            burn = -fcf
        if burn and total_cash is not None:
            runway_years = total_cash / burn
            if runway_years < 1.0:
                reasons.append(f"cash runway ~{runway_years:.1f}y at current burn")
                verdict = "exclude"
            elif runway_years < 2.0:
                reasons.append(f"cash runway ~{runway_years:.1f}y at current burn")
                if verdict == "ok":
                    verdict = "flag"

    result = (verdict, "; ".join(reasons) if reasons else "no fundamental red flags")
    FUNDAMENTAL_VETO_CACHE[sym] = result
    return result

# PRICE SNAPSHOT (reproducibility):
# Freezing prices alone isn't enough — six months from now the live Yahoo
# screener would return a different candidate set (names falling below the
# cap threshold, acquisitions, metadata changes), silently changing what's
# supposedly a frozen experiment. So the candidate universe (tickers +
# sector/cost_bps) is frozen and reloaded alongside prices, not rebuilt from
# today's screener when replaying a snapshot.
SNAPSHOT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_snapshot")
SNAPSHOT_PARQUET  = os.path.join(SNAPSHOT_DIR, "prices.parquet")
SNAPSHOT_UNIVERSE = os.path.join(SNAPSHOT_DIR, "universe.csv")
SNAPSHOT_MANIFEST = os.path.join(SNAPSHOT_DIR, "manifest.json")
USE_FROZEN_SNAPSHOT = os.environ.get("NYSE_ALPHA_USE_SNAPSHOT", "0") == "1"
SAVE_SNAPSHOT        = os.environ.get("NYSE_ALPHA_FREEZE_SNAPSHOT", "0") == "1"

if USE_FROZEN_SNAPSHOT:
    if not os.path.exists(SNAPSHOT_UNIVERSE):
        raise RuntimeError(
            f"NYSE_ALPHA_USE_SNAPSHOT=1 but no frozen universe at {SNAPSHOT_UNIVERSE}. "
            f"Run once with NYSE_ALPHA_FREEZE_SNAPSHOT=1 to create one."
        )
    _univ_df = pd.read_csv(SNAPSHOT_UNIVERSE)
    CANDIDATE_UNIVERSE = {
        row.ticker: {"sector": row.sector, "cost_bps": row.cost_bps}
        for row in _univ_df.itertuples()
    }
    print(f"Loaded frozen candidate universe from {SNAPSHOT_UNIVERSE} "
          f"({len(CANDIDATE_UNIVERSE)} names) — sector/cost_bps frozen alongside prices, "
          f"not re-screened from today's Yahoo data.")
else:
    CANDIDATE_UNIVERSE = build_universe()

if UNIVERSE_WATCHLIST:
    _missing_from_screen = sorted(UNIVERSE_WATCHLIST - set(CANDIDATE_UNIVERSE.keys()))
    if _missing_from_screen:
        print(f"   Warning: watchlist names not in screened universe, excluded "
              f"(failed market-cap/domicile/exchange filter): {_missing_from_screen}")
    CANDIDATE_UNIVERSE = {s: v for s, v in CANDIDATE_UNIVERSE.items() if s in UNIVERSE_WATCHLIST}
    print(f"   Watchlist restriction active (universe_watchlist.json): candidate "
          f"universe narrowed to {len(CANDIDATE_UNIVERSE)} names")
MACRO_TICKERS = ["^VIX", "^TNX", "LQD"]
BENCH_TICK    = "^GSPC"

# 1. DATA DOWNLOAD & LIQUIDITY SCREEN
print("=" * 70)
print("LONG-ONLY MOMENTUM MODEL")
print("Systematic Universe · Momentum/RS · HMM Exposure Control · Inverse-Vol Sizing")
print("=" * 70)
print("\n[1/8] Downloading market data and applying liquidity screen")

def _tickers_missing_close(df, tickers):
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            return list(tickers)
        close_cols = df["Close"]
        return [t for t in tickers if t not in close_cols.columns or close_cols[t].notna().sum() == 0]
    return [] if "Close" in df.columns and df["Close"].notna().any() else list(tickers)

def _download_prices_with_retry(tickers, start, attempts=4, base_delay=20.0, max_missing_ratio=0.05):
    last_df, last_missing = None, list(tickers)
    for attempt in range(attempts):
        try:
            df = yf.download(tickers, start=start, interval="1d",
                              auto_adjust=True, progress=False)
        except Exception as e:
            df = None
            print(f"   Price download attempt {attempt + 1}/{attempts} raised {e!r}")
        if df is not None and not df.empty:
            last_df = df
            last_missing = _tickers_missing_close(df, tickers)
            if len(last_missing) <= max(5, int(len(tickers) * max_missing_ratio)):
                return df, last_missing
            print(f"   {len(last_missing)}/{len(tickers)} tickers came back with no data "
                  f"(likely Yahoo throttling, not real absence) — retrying full batch "
                  f"(attempt {attempt + 1}/{attempts}) ...")
        if attempt < attempts - 1:
            time.sleep(base_delay * (attempt + 1))
    return last_df, last_missing

all_tickers = list(CANDIDATE_UNIVERSE.keys()) + MACRO_TICKERS + [BENCH_TICK]

if USE_FROZEN_SNAPSHOT:
    if not os.path.exists(SNAPSHOT_PARQUET):
        raise RuntimeError(
            f"NYSE_ALPHA_USE_SNAPSHOT=1 but no snapshot at {SNAPSHOT_PARQUET}. "
            f"Run once with NYSE_ALPHA_FREEZE_SNAPSHOT=1 to create one."
        )
    raw = pd.read_parquet(SNAPSHOT_PARQUET)
    with open(SNAPSHOT_MANIFEST, "r") as f:
        _manifest = json.load(f)
    _missing_after_retry = _manifest.get("missing_tickers", [])
    print(f"Loaded frozen price snapshot from {SNAPSHOT_PARQUET} "
          f"(downloaded {_manifest.get('downloaded_at')}, {_manifest.get('n_rows')} rows, "
          f"{len(_missing_after_retry)} missing tickers).")
    _raw_tickers = set(raw.columns.get_level_values(1)) if isinstance(raw.columns, pd.MultiIndex) else set()
    _univ_not_in_snapshot = [s for s in CANDIDATE_UNIVERSE if s not in _raw_tickers]
    if _univ_not_in_snapshot:
        raise RuntimeError(
            f"Frozen universe.csv references {len(_univ_not_in_snapshot)} ticker(s) not present "
            f"in the frozen prices.parquet ({_univ_not_in_snapshot[:10]}"
            f"{'...' if len(_univ_not_in_snapshot) > 10 else ''}) — the two snapshot files are out "
            f"of sync. Re-freeze both together with NYSE_ALPHA_FREEZE_SNAPSHOT=1."
        )
else:
    raw, _missing_after_retry = _download_prices_with_retry(all_tickers, CFG["start"])
    if raw is None:
        raise RuntimeError("Price data download failed after retries — no usable response from Yahoo Finance.")
    if SAVE_SNAPSHOT:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        raw.to_parquet(SNAPSHOT_PARQUET)
        pd.DataFrame([
            {"ticker": s, "sector": v["sector"], "cost_bps": v["cost_bps"]}
            for s, v in CANDIDATE_UNIVERSE.items()
        ]).to_csv(SNAPSHOT_UNIVERSE, index=False)
        manifest = {
            "downloaded_at":       pd.Timestamp.now(tz="UTC").isoformat(),
            "start_date":          CFG["start"],
            "n_tickers_requested": len(all_tickers),
            "missing_tickers":     _missing_after_retry,
            "n_rows":              len(raw),
            "date_range":          [str(raw.index.min()), str(raw.index.max())],
            "yfinance_version":    yf.__version__,
            "pandas_version":      pd.__version__,
        }
        with open(SNAPSHOT_MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved frozen price snapshot to {SNAPSHOT_DIR} "
              f"({len(raw)} rows, {len(_missing_after_retry)} missing tickers).")

def extract(raw, field, tickers):
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw[field][tickers]
    else:
        df = raw
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df

def get_factor(raw, ticker):
    s = (raw["Close"][ticker].ffill()
         if isinstance(raw.columns, pd.MultiIndex) else raw.ffill())
    s.index = s.index.tz_localize(None) if s.index.tz else s.index
    return s

cand_syms  = list(CANDIDATE_UNIVERSE.keys())
close_all  = extract(raw, "Close",  cand_syms)
high_all   = extract(raw, "High",   cand_syms)
low_all    = extract(raw, "Low",    cand_syms)
vol_all    = extract(raw, "Volume", cand_syms)

# Yahoo's batch download sometimes includes a blank row for US market holidays
# (all-NaN across every ticker) instead of omitting the date entirely. That's
# not a per-name data gap — the market simply wasn't open — so it's dropped
# outright rather than left as a NaN that would poison every rolling window
# (e.g. the 63-day ADV screen) it touches for the following ~63 sessions.
_phantom_rows = close_all.isna().all(axis=1)
if _phantom_rows.any():
    print(f"Dropping {int(_phantom_rows.sum())} phantom no-trading-day row(s) "
          f"(all-NaN across the universe, not a real per-name gap): "
          f"{[d.strftime('%Y-%m-%d') for d in close_all.index[_phantom_rows]]}")
    close_all = close_all.loc[~_phantom_rows]
    high_all  = high_all.loc[~_phantom_rows]
    low_all   = low_all.loc[~_phantom_rows]
    vol_all   = vol_all.loc[~_phantom_rows]

bench_s  = get_factor(raw, BENCH_TICK)
vix_s    = get_factor(raw, "^VIX")
tnx_s    = get_factor(raw, "^TNX").ffill()  # no .bfill(): backfilling the start of
                                             # the series would use future data to
                                             # invent a value before TNX's first print
lqd_s    = get_factor(raw, "LQD")

symbols = [s for s in cand_syms
           if s in close_all.columns and close_all[s].notna().any()]
dropped_syms = sorted(set(cand_syms) - set(symbols))
if dropped_syms:
    print(f"DROP (no data): {dropped_syms}")
print(f"Candidate universe: {len(symbols)} names with data"
      f"(Yahoo Finance screener, see build_universe())")
print("CAVEAT — SURVIVORSHIP BIAS")

close  = close_all[symbols]
high   = high_all[symbols].reindex(close.index)
low    = low_all[symbols].reindex(close.index)
volume = vol_all[symbols].reindex(close.index)

def compute_atr(sym, period=None):
    period = period or CFG["atr_period"]
    if sym not in close.columns:
        return None
    h, l, c = high[sym], low[sym], close[sym]
    prev_c  = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

price   = close
returns = price.pct_change(fill_method=None).dropna(how="all")

if len(returns) == 0:
    raise RuntimeError()

bench_price  = bench_s.reindex(price.index,  method="ffill")
vix_price    = vix_s.reindex(price.index,    method="ffill")
tnx_price    = tnx_s.reindex(price.index,    method="ffill")
lqd_price    = lqd_s.reindex(price.index,    method="ffill")

if pd.isna(tnx_price.iloc[-1]):
    tnx_price.iloc[-1] = get_live_tnx()

bench_ret  = bench_price.pct_change().reindex(returns.index).fillna(0)

meta = {s: CANDIDATE_UNIVERSE[s] for s in symbols}
cols = symbols

# POINT-IN-TIME LIQUIDITY / COVERAGE SCREEN
MIN_COVERAGE     = 0.95
MIN_ADV_USD      = 20_000_000   
MIN_PRICE        = 5.00         
MIN_HISTORY_DAYS = 260         

dollar_vol    = close[symbols] * volume[symbols]
adv_trailing  = dollar_vol.rolling(63, min_periods=63).median()

notna_mat = close[symbols].notna()
expanding_cov = pd.DataFrame(index=notna_mat.index, columns=symbols, dtype=float)
for sym in symbols:
    col = notna_mat[sym]
    if not col.any():
        expanding_cov[sym] = 0.0
        continue
    first_idx = col.idxmax()
    expanding_cov.loc[first_idx:, sym] = col.loc[first_idx:].expanding(min_periods=1).mean()

history_days  = close[symbols].notna().expanding(min_periods=1).sum()

eligible = (
    (expanding_cov >= MIN_COVERAGE) &
    (adv_trailing  >= MIN_ADV_USD)  &
    (close[symbols] >= MIN_PRICE)   &
    (history_days  >= MIN_HISTORY_DAYS)
).reindex(returns.index).fillna(False)
eligible_liquidity = eligible.copy()

print(f"Point-in-time screen: coverage>={MIN_COVERAGE:.0%}, "
      f"ADV(63d)>=${MIN_ADV_USD:,}, price>=${MIN_PRICE}, history>={MIN_HISTORY_DAYS}d")
print(f"Eligible names as of latest date: "
      f"{eligible.iloc[-1][eligible.iloc[-1]].index.tolist()}")

# SANITY GATE: with ~380 candidate names, the eligible count on the latest
# date should never legitimately be near zero — that only happens from a
# data-pipeline failure (e.g. a phantom all-NaN calendar row poisoning every
# name's rolling ADV window at once, as a market-holiday row from Yahoo did
# here). Fail loudly rather than silently handing back an empty/near-empty
# target that could be mistaken for a real "hold nothing" signal.
MIN_ELIGIBLE_FLOOR = 20
_latest_eligible_n = int(eligible.iloc[-1].sum())
if _latest_eligible_n < MIN_ELIGIBLE_FLOOR:
    raise RuntimeError(
        f"Only {_latest_eligible_n} names passed the liquidity/coverage screen "
        f"on {eligible.index[-1].date()} (floor is {MIN_ELIGIBLE_FLOOR}, out of "
        f"{eligible.shape[1]} candidates) — this smells like a data problem, not "
        f"a real market condition. Refusing to emit a target. Check for a fresh "
        f"phantom all-NaN row (market holiday) or a broken download before rerunning."
    )

# TRANSACTION COST MODEL:
IMPACT_COEFF = 0.1
_cost_sig = {s: returns[s].rolling(21).std().reindex(returns.index).fillna(0) for s in symbols}
_cost_adv = {
    s: dollar_vol[s].rolling(21).median().reindex(returns.index).clip(lower=5_000_000).fillna(5_000_000)
    for s in symbols
}

def compute_transaction_costs(w_df):
    turnover_per_sym_l = w_df.reindex(returns.index).diff().abs()
    cost = pd.Series(0.0, index=returns.index)
    for s in symbols:
        fixed_cost = turnover_per_sym_l[s] * meta[s]["cost_bps"]
        trade_val  = (turnover_per_sym_l[s] * CFG["backtest_nav"]).fillna(0)
        impact     = IMPACT_COEFF * _cost_sig[s] * np.sqrt((trade_val / _cost_adv[s]).clip(lower=0))
        cost += fixed_cost + impact.clip(upper=0.02)
    return cost

def _daily_trade_cost(prev_w, day_w, date):
    # Buy-and-hold days between rebalances have zero turnover across the board —
    # skip the per-symbol loop entirely rather than scanning all ~400 columns
    # on every one of ~2000+ trading days for what's almost always a no-op.
    turnover_today = (day_w - prev_w).abs()
    traded = turnover_today[turnover_today > 0.0]
    if traded.empty:
        return 0.0
    cost = 0.0
    for s, t in traded.items():
        fixed_cost = t * meta[s]["cost_bps"]
        trade_val  = t * CFG["backtest_nav"]
        impact     = IMPACT_COEFF * _cost_sig[s].loc[date] * np.sqrt(max(trade_val / _cost_adv[s].loc[date], 0.0))
        cost += fixed_cost + min(impact, 0.02)
    return cost

def _check_no_missing_held_returns(day_w, ret_row, date):
    # (day_w * ret_row).sum() skips NaN by default — a held name with a
    # missing Yahoo observation would silently contribute 0% that day
    # instead of failing the run, now that ffill no longer manufactures a
    # value to fall back on. A real trading day can't have a genuinely
    # missing return for a name we're actually holding, so this fails loud.
    # Fully vectorized (no per-symbol Python loop) since this runs on every
    # day of every backtest/ablation pass.
    bad = (day_w.abs() > 1e-9) & ret_row.isna()
    if bad.any():
        raise RuntimeError(
            f"Missing return on {pd.Timestamp(date).date()} for held name(s) "
            f"{day_w.index[bad].tolist()} (nonzero weight, no price data) — "
            f"refusing to silently treat this as a 0% return. Check the price "
            f"download for this date."
        )

# 2. REGIME DETECTION: HMM (primary) + MACRO OVERLAY + FAST STRESS OVERRIDE
print("[2/8] HMM regime filter + Macro overlay")

hmm_feat = pd.DataFrame({
    "spx": bench_ret,
    "vix": vix_price.pct_change().reindex(bench_ret.index).fillna(0),
    "lqd": lqd_price.pct_change().reindex(bench_ret.index).fillna(0),
}).dropna()

hmm_feat_raw = hmm_feat.values

HMM_MIN_TRAIN   = CFG["hmm_min_train"]
HMM_REFIT_EVERY = CFG["hmm_refit_every"]
n_hmm = len(hmm_feat)

if n_hmm < HMM_MIN_TRAIN:
    raise RuntimeError(f"Only {n_hmm} days of regime-feature history"
                       f"{HMM_MIN_TRAIN} for the first HMM fit.")

refit_cuts = list(range(HMM_MIN_TRAIN, n_hmm, HMM_REFIT_EVERY)) + [n_hmm]

def hmm_filtered_states(model, X):
    n_states = model.n_components
    reg = 1e-6 * np.eye(model.covars_[0].shape[0])
    log_emit = np.column_stack([
        multivariate_normal.logpdf(X, mean=model.means_[k], cov=model.covars_[k] + reg)
        for k in range(n_states)
    ])
    log_start = np.log(model.startprob_ + 1e-300)
    log_trans = np.log(model.transmat_ + 1e-300)

    T = len(X)
    states = np.zeros(T, dtype=int)
    log_alpha = log_start + log_emit[0]
    log_alpha -= logsumexp(log_alpha)
    states[0] = np.argmax(log_alpha)
    for t in range(1, T):
        log_alpha = logsumexp(log_alpha[:, None] + log_trans, axis=0) + log_emit[t]
        log_alpha -= logsumexp(log_alpha)
        states[t] = np.argmax(log_alpha)
    return states

def fit_expanding_hmm(n_states, state_labels):

    assert len(state_labels) == n_states
    regime      = pd.Series(state_labels[len(state_labels) // 2], index=hmm_feat.index, dtype=object)
    model       = None
    n_refits_ok = 0
    model_mu = model_std = None  

    for i, cut in enumerate(refit_cuts[:-1]):
        seg_end = refit_cuts[i + 1]

        train_raw = hmm_feat_raw[:cut]
        win_mu    = train_raw.mean(axis=0)
        win_std   = train_raw.std(axis=0)
        win_std[win_std == 0] = 1.0
        train_X   = (train_raw - win_mu) / win_std

        best_model, best_score = None, -np.inf
        for seed in range(CFG["hmm_restarts"]):
            try:
                m = hmmlib.GaussianHMM(
                    n_components    = n_states,
                    covariance_type = "full",
                    n_iter          = 300,
                    random_state    = seed,
                    tol             = 1e-6,
                )
                m.fit(train_X)
                score = m.score(train_X)
                if score > best_score:
                    best_score, best_model = score, m
            except Exception:
                continue

        if best_model is not None:
            model, model_mu, model_std = best_model, win_mu, win_std
            n_refits_ok += 1

        if model is None:
            continue

        train_states = model.predict((train_raw - model_mu) / model_std)
        train_idx    = hmm_feat.index[:cut]
        fwd_bench_ret = bench_ret.shift(-1)
        label_idx    = train_idx[:-1]
        label_states = train_states[:-1]
        state_spx_mean = {
            s: fwd_bench_ret.reindex(label_idx[label_states == s]).mean()
            for s in range(n_states)
        }
        sorted_states = sorted(
            state_spx_mean,
            key=lambda s: state_spx_mean[s] if pd.notna(state_spx_mean[s]) else -np.inf,
            reverse=True,
        )

        state_to_regime = {sorted_states[k]: state_labels[k] for k in range(n_states)}

        full_X     = (hmm_feat_raw[:seg_end] - model_mu) / model_std
        filtered   = hmm_filtered_states(model, full_X)
        seg_states = filtered[cut:seg_end]
        if len(seg_states) > 0:
            seg_idx = hmm_feat.index[cut:seg_end]
            regime.loc[seg_idx] = pd.Series(seg_states, index=seg_idx).map(state_to_regime)

    return regime, n_refits_ok

hmm_regime, n_refits_ok = fit_expanding_hmm(CFG["hmm_states"], ["risk_on", "neutral", "risk_off"])

if n_refits_ok == 0:
    raise RuntimeError()

print(f"HMM expanding-window refits: {n_refits_ok}/{len(refit_cuts)-1} converged"
      f"(every {HMM_REFIT_EVERY}d, min train {HMM_MIN_TRAIN}d)")
print("HMM regime distribution (point-in-time, no full-sample look-ahead):")
for reg, cnt in hmm_regime.value_counts().sort_index().items():
    print(f"     {reg:>4}: {cnt:>5} days  ({cnt/len(hmm_regime):.1%})")

vix_below_avg  = (vix_price < vix_price.rolling(50).mean()).astype(int)
rates_falling  = (tnx_price.diff(5) < 0).astype(int)
credit_healthy = (lqd_price.pct_change(5) > 0).astype(int)
macro_score    = (vix_below_avg + rates_falling + credit_healthy
                  ).reindex(returns.index).fillna(0)

vix_spike     = (vix_price > vix_price.rolling(5).mean() * 1.30).astype(int)
credit_stress = (lqd_price.pct_change(3) < -0.015).astype(int)
stress_flag   = ((vix_spike + credit_stress) >= 1
                 ).reindex(returns.index).fillna(0)

def composite_regime(date):
    if stress_flag.get(date, 0) == 1:
        return "stress"
    hmm_r = hmm_regime.get(date, "neutral")
    ms    = int(macro_score.get(date, 1))
    if ms == 0:
        return "risk_off"
    if ms == 3 and hmm_r != "risk_off":
        return "risk_on"
    return hmm_r

regime_series = pd.Series(
    {d: composite_regime(d) for d in returns.index}, dtype=str
)

reg_counts = regime_series.value_counts()
print("Composite regime distribution:")
for reg, cnt in reg_counts.sort_index().items():
    print(f"     {reg:>6}: {cnt:>5} days  ({cnt / len(regime_series):.1%})")

# 3. SIGNALS
print("[3/8] Computing signals ...")

def cs_rank(df):
    return 2 * (df.rank(axis=1, pct=True, na_option="keep") - 0.5)

def cs_zscore(df, winsor=0.02):
    lo = df.quantile(winsor, axis=1)
    hi = df.quantile(1 - winsor, axis=1)
    clipped = df.clip(lower=lo, upper=hi, axis=0)
    mu = clipped.mean(axis=1)
    sd = clipped.std(axis=1).replace(0, np.nan)
    return clipped.sub(mu, axis=0).div(sd, axis=0)

SKIP_DAYS = 5
mom_1m  = price.shift(SKIP_DAYS) / price.shift(SKIP_DAYS + 21)  - 1
mom_3m  = price.shift(SKIP_DAYS) / price.shift(SKIP_DAYS + 63)  - 1
mom_6m  = price.shift(SKIP_DAYS) / price.shift(SKIP_DAYS + 126) - 1
mom_12m = price.shift(SKIP_DAYS) / price.shift(SKIP_DAYS + 252) - 1

vol_63 = returns.rolling(63).std().replace(0, np.nan)

mom_raw     = 0.15 * mom_1m + 0.30 * mom_3m + 0.30 * mom_6m + 0.15 * mom_12m
mom_vol_adj = mom_raw / vol_63
sig_mom     = cs_rank(mom_vol_adj)

bench_6m     = bench_price.pct_change(126)
rel_strength = price.pct_change(126, fill_method=None).sub(bench_6m, axis=0)
sig_rs       = cs_rank(rel_strength)

alpha_final = (0.90 * sig_mom.fillna(0) + 0.10 * sig_rs.fillna(0))
alpha_final = alpha_final.ewm(halflife=5).mean().reindex(returns.index).fillna(0)

_sig_df = pd.concat([
    sig_mom.stack().rename("mom"),
    sig_rs.stack().rename("rs"),
], axis=1).dropna()
_corr = _sig_df.corr()
print("Signal cross-correlation (flag if |r| > 0.6):")
print(_corr.round(2).to_string())
high_corr = [(i, j) for i in _corr.columns for j in _corr.columns
             if i < j and abs(_corr.loc[i, j]) > 0.6]
if high_corr:
    for i, j in high_corr:
        print(f"HIGH CORRELATION: {i} vs {j} = {_corr.loc[i,j]:.2f}")
else:
    print("All signal pairs below 0.6 threshold")

# 3b. ALPHA RESEARCH DIAGNOSTICS
print("[3b/8] Alpha research diagnostics (rank-IC ablation, report only)")

IC_HORIZON = 21
fwd_ret = price.pct_change(IC_HORIZON, fill_method=None).shift(-IC_HORIZON).reindex(returns.index)

def rank_ic_stats(signal_df, label):
    masked = signal_df.reindex_like(eligible_liquidity).where(eligible_liquidity)
    ic = masked.corrwith(fwd_ret, axis=1, method="spearman").dropna()
    if len(ic) == 0:
        print(f"   {label:<32} insufficient data")
        return
    print(f"   {label:<32} mean IC={ic.mean():+.4f}  std={ic.std():.4f}  "
          f"hit-rate={(ic > 0).mean():.1%}  n={len(ic)}")

print(f"Rank-IC vs fwd {IC_HORIZON}d return "
      f"(diagnostic only — uses future data, never fed to the live signal):")
print("1/3/6/12 weighting ablation:")
rank_ic_stats(cs_rank(mom_6m),                        "M6 alone")
rank_ic_stats(cs_rank(0.5 * mom_3m + 0.5 * mom_6m),   "M3+M6 (50/50)")
rank_ic_stats(cs_rank(mom_raw),                       "Full 1/3/6/12 (raw)")
rank_ic_stats(cs_rank(mom_vol_adj),                   "+ vol adjustment")
rank_ic_stats(0.90 * sig_mom.fillna(0) + 0.10 * sig_rs.fillna(0), "+ RS (90/10)")
rank_ic_stats(alpha_final,                            "+ EWM smoothing (current live alpha)")

sig_mom_z     = cs_zscore(mom_vol_adj)
sig_mom_blend = 0.5 * sig_mom.fillna(0) + 0.5 * sig_mom_z.fillna(0)

print("Rank vs z-score (same vol-adj momentum input, different cross-sectional transform):")
rank_ic_stats(sig_mom,       "Percentile rank (current)")
rank_ic_stats(sig_mom_z,     "Winsorized z-score")
rank_ic_stats(sig_mom_blend, "50/50 rank+z blend")

# 3c. MOMENTUM FACTOR HEALTH
print("[3c/8] Momentum factor health (exposure modifier)")

FACTOR_TERCILE   = 0.30
mom_score_lagged = mom_vol_adj.shift(1)

def _factor_leg_return(t_idx):
    scores = mom_score_lagged.iloc[t_idx].where(eligible.iloc[t_idx]).dropna()
    if len(scores) < 10:
        return np.nan
    n_leg = max(1, int(len(scores) * FACTOR_TERCILE))
    high_syms = scores.nlargest(n_leg).index
    low_syms  = scores.nsmallest(n_leg).index
    day_ret   = returns[cols].iloc[t_idx]
    return day_ret[high_syms].mean() - day_ret[low_syms].mean()

mom_factor_ret = pd.Series(
    [_factor_leg_return(i) for i in range(len(returns.index))],
    index=returns.index,
).fillna(0)

factor_ret_3m  = mom_factor_ret.rolling(63).sum()
factor_cum     = (1 + mom_factor_ret).cumprod()
factor_dd      = factor_cum / factor_cum.cummax() - 1

FACTOR_CRASH_RET, FACTOR_CRASH_DD = -0.05, -0.15
MOM_FACTOR_SCALE = {"healthy": 1.00, "weak": 0.75, "crash": 0.50}

def factor_health(date):
    ret3m = factor_ret_3m.get(date, np.nan)
    dd    = factor_dd.get(date, np.nan)
    if pd.isna(ret3m):
        return "healthy"
    if ret3m <= FACTOR_CRASH_RET or dd <= FACTOR_CRASH_DD:
        return "crash"
    if ret3m < 0:
        return "weak"
    return "healthy"

factor_health_series  = pd.Series({d: factor_health(d) for d in returns.index})
factor_exposure_scale = factor_health_series.map(MOM_FACTOR_SCALE)

print(f"Momentum factor health thresholds: crash if 3m factor return<={FACTOR_CRASH_RET:.0%} "
      f"or drawdown<={FACTOR_CRASH_DD:.0%}, else weak if 3m return<0, else healthy "
      f"→ exposure scale {MOM_FACTOR_SCALE}")
for reg, cnt in factor_health_series.value_counts().sort_index().items():
    print(f"     {reg:>7}: {cnt:>5} days  ({cnt / len(factor_health_series):.1%})")
print(f"   Latest: {factor_health_series.iloc[-1]} "
      f"(3m factor ret={factor_ret_3m.iloc[-1]:+.2%}, "
      f"factor DD={factor_dd.iloc[-1]:+.2%}, "
      f"exposure scale={factor_exposure_scale.iloc[-1]:.0%})")

# 4. TREND / RELATIVE-STRENGTH / VOLATILITY GATES
print("[4/8] Applying trend/RS/volatility confirmation gates")

ma_50, ma_200 = price.rolling(50).mean(), price.rolling(200).mean()
VOL_CEILING = 0.65
vol_ann     = vol_63 * np.sqrt(252)

trend_gate = (
    (price > ma_50) & (ma_50 > ma_200) & (rel_strength > 0) & (vol_ann <= VOL_CEILING) &
    (mom_3m > 0) & (mom_6m > 0)
).reindex(returns.index).fillna(False)

eligible = eligible & trend_gate
print(f"Gates: price>50DMA, 50DMA>200DMA, RS>0 vs S&P500, vol<={VOL_CEILING:.0%} ann., "
      f"M3m>0, M6m>0")
print(f"   Tradable names as of latest date: "
      f"{eligible.iloc[-1][eligible.iloc[-1]].index.tolist()}")

# 5. SIZING FUNCTIONS
print("[5/8] Initialising sizing engine "
      f"(sizing_method='{CFG['sizing_method']}') ...")

LOOKBACK_COV = 63

def _to_array(x):
    return x.values if hasattr(x, "values") else np.asarray(x)

def inverse_vol_weights(ret_window):
    r   = _to_array(ret_window)
    vol = r.std(axis=0)
    pos = vol[vol > 0]
    fallback = pos.mean() if len(pos) > 0 else 1.0
    vol = np.where(vol > 0, vol, fallback)
    inv = 1.0 / vol
    return inv / inv.sum()

def semi_covariance_matrix(ret_input):
    r         = _to_array(ret_input)
    n_obs, n  = r.shape
    port_mean = r.mean(axis=1)
    down_mask = port_mean < 0
    r_down    = r[down_mask]

    if len(r_down) < 10:
        return np.cov(r.T) * 252 + np.eye(n) * 1e-6

    semi_cov  = (r_down.T @ r_down) / n_obs * 252
    semi_cov += np.eye(n) * 1e-6
    return semi_cov

def cvar_optimal_weights(ret_window, prev_w_arr, n_assets, alpha_vec=None):
    confidence = CFG["cvar_confidence"]
    cvar_limit = CFG["cvar_limit"]
    lambda_to  = CFG["lambda_turnover"]
    lambda_a   = CFG["lambda_alpha"]

    R  = _to_array(ret_window)
    T  = R.shape[0]
    w0 = np.ones(n_assets) / n_assets
    if prev_w_arr is None or len(prev_w_arr) != n_assets:
        prev_w_arr = w0.copy()
    if alpha_vec is None or len(alpha_vec) != n_assets:
        alpha_vec = np.zeros(n_assets)

    w = cp.Variable(n_assets)
    z = cp.Variable()
    u = cp.Variable(T, nonneg=True)

    cvar_expr = z + (1.0 / ((1.0 - confidence) * T)) * cp.sum(u)
    turnover_penalty = lambda_to * cp.norm1(w - prev_w_arr)
    alpha_term = lambda_a * (alpha_vec @ w)

    objective = cp.Minimize(cvar_expr + turnover_penalty - alpha_term)
    constraints = [
        cp.sum(w) == 1,
        w >= CFG["min_weight"],
        w <= CFG["max_weight"],
        u >= -R @ w - z,
        cvar_expr <= cvar_limit,
    ]
    prob = cp.Problem(objective, constraints)

    for solver in [cp.CLARABEL, cp.ECOS]:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                result = np.clip(w.value, CFG["min_weight"], CFG["max_weight"])
                result /= result.sum()
                achieved = float(
                    z.value
                    + (1.0 / ((1.0 - confidence) * T)) * np.sum(np.maximum(u.value, 0))
                )
                if achieved > cvar_limit * 1.05:
                    print(f"Delta CVaR soft breach ({solver.__name__ if hasattr(solver,'__name__') else solver}): "
                          f"achieved={achieved:.4f} > limit={cvar_limit:.4f}")
                return result
        except Exception as e:
            print(f"Delta CVXPY {solver} failed: {e}")
            continue

    print("Both solvers failed — equal-weight fallback")
    return w0

def kelly_downside(alpha_vec, ret_window, fraction=None):
    fraction = fraction or CFG["kelly_fraction"]
    semi_cov = semi_covariance_matrix(ret_window)
    try:
        inv_semi = np.linalg.inv(semi_cov)
        w_k      = fraction * inv_semi @ alpha_vec
        w_k      = np.clip(w_k, 0, None)
        return w_k / w_k.sum() if w_k.sum() > 0 else np.ones(len(alpha_vec)) / len(alpha_vec)
    except np.linalg.LinAlgError:
        return np.ones(len(alpha_vec)) / len(alpha_vec)

def cap_and_redistribute(w, max_weight):

    w = w.copy()
    for _ in range(25):
        over = w[w > max_weight + 1e-9]
        if over.empty:
            break
        excess = (over - max_weight).sum()
        w[over.index] = max_weight
        room = w[w < max_weight - 1e-9]
        if room.empty or room.sum() <= 0:
            break
        w[room.index] += excess * room / room.sum()
    return w

def enforce_sector_cap(w, names, cap):
    w = w.copy()
    for _ in range(25):
        sector_alloc = {}
        for s in names:
            sec = meta[s]["sector"]
            sector_alloc[sec] = sector_alloc.get(sec, 0) + w.get(s, 0)
        violated = {sec: alloc for sec, alloc in sector_alloc.items() if alloc > cap * 1.001}
        if not violated:
            break
        for sec, alloc in violated.items():
            trim = cap / alloc
            for s in names:
                if meta[s]["sector"] == sec and s in w.index:
                    w[s] *= trim
    return w

# 6. PORTFOLIO CONSTRUCTION LOOP
print("[6/8] Running portfolio construction loop")

weights           = pd.DataFrame(0.0, index=returns.index, columns=cols)
pending_w         = pd.Series(0.0, index=cols)
rolling_eq        = [1.0]
frozen_days       = 0
post_freeze_rebal = False
circuit_breaker   = pd.Series(False, index=returns.index)
port_ret_hist     = []
vol_derisked      = False
dd_peak_ref       = 1.0
prev_day_w        = pd.Series(0.0, index=cols)

rebal_dates = pd.DatetimeIndex(
    returns.index.to_series().resample("W-FRI").last().dropna().values
)
WARMUP = CFG["warmup_days"]

for t_idx in range(len(returns.index)):
    date = returns.index[t_idx]

    if t_idx < WARMUP:
        weights.iloc[t_idx] = 0.0
        rolling_eq.append(rolling_eq[-1])
        continue

    was_frozen  = frozen_days > 0
    day_w       = pd.Series(0.0, index=cols) if was_frozen else pending_w
    _check_no_missing_held_returns(day_w, returns[cols].iloc[t_idx], date)
    day_cost    = _daily_trade_cost(prev_day_w, day_w, date)
    day_ret_gross = (day_w * returns[cols].iloc[t_idx]).sum()
    day_ret     = day_ret_gross - day_cost
    prev_day_w  = day_w
    port_ret_hist.append(day_ret)
    rolling_eq.append(rolling_eq[-1] * (1 + day_ret))
    weights.iloc[t_idx] = day_w
    circuit_breaker.iloc[t_idx] = was_frozen
    curr_eq = rolling_eq[-1]
    dd_peak_ref = max(dd_peak_ref, curr_eq)

    if was_frozen:
        # Time-boxed freeze:
        frozen_days -= 1
        if frozen_days == 0:
            post_freeze_rebal = True
            dd_peak_ref = curr_eq
        else:
            pending_w = pd.Series(0.0, index=cols)
            continue

    if not post_freeze_rebal:
        # Skip the trigger check on the resume day itself:
        dd_peak = (curr_eq / dd_peak_ref) - 1
        if dd_peak < CFG["dd_stop"]:
            frozen_days = CFG["dd_cooldown"]
            pending_w = pd.Series(0.0, index=cols)
            continue

    if date in rebal_dates or post_freeze_rebal:
        post_freeze_rebal = False
        alpha_row = alpha_final.iloc[t_idx]
        n_slots   = CFG["top_n"]

        elig_row  = eligible.iloc[t_idx]
        qualified = alpha_row.where(elig_row)
        qualified = qualified[qualified > CFG["score_threshold"]]
        top_syms  = qualified.nlargest(n_slots).index.tolist()

        if len(top_syms) == 0:
            pending_w = pd.Series(0.0, index=cols)
            continue

        invested_fraction = len(top_syms) / n_slots
        ret_window = returns[top_syms].iloc[max(0, t_idx - LOOKBACK_COV + 1):t_idx + 1]

        if len(ret_window) < 20:
            slot_w = np.ones(len(top_syms)) / len(top_syms)
        elif CFG["sizing_method"] == "cvar_kelly":
            prev_w_arr = pending_w.reindex(top_syms).fillna(0).values
            alpha_vec  = alpha_row[top_syms].clip(lower=0).values
            cvar_w = cvar_optimal_weights(
                ret_window.values, prev_w_arr, len(top_syms), alpha_vec
            )
            kel_w  = kelly_downside(alpha_vec, ret_window.values)
            slot_w = 0.75 * cvar_w + 0.25 * kel_w
            slot_w = slot_w / slot_w.sum()
        else:
            slot_w = inverse_vol_weights(ret_window.values)

        target_w = pd.Series(slot_w * invested_fraction, index=top_syms)
        target_w = target_w.clip(lower=CFG["min_weight"])
        target_w = cap_and_redistribute(target_w, CFG["max_weight"])
        if target_w.sum() > invested_fraction:
            target_w *= invested_fraction / target_w.sum()
        new_w = enforce_sector_cap(target_w, top_syms, CFG["sector_cap"])

        candidate = pd.Series(0.0, index=cols)
        candidate[top_syms] = new_w

        relevant = pending_w[pending_w.abs() > 1e-9].index.union(candidate[candidate != 0].index)
        final_w = pending_w.copy()
        for s in relevant:
            if pending_w[s] > 0 and candidate[s] == 0:
                final_w[s] = 0.0                             
            elif abs(candidate[s] - pending_w[s]) > CFG["rebal_band"]:
                final_w[s] = candidate[s]                    

        active_names = final_w[final_w.abs() > 1e-9].index.tolist()
        final_w = enforce_sector_cap(final_w, active_names, CFG["sector_cap"])

        if final_w.sum() > 1.0:
            final_w /= final_w.sum()

        if regime_series.get(date, "neutral") == "stress":
            final_w = final_w * CFG["stress_scale"]

        # Portfolio volatility targeting:
        VOL_TARGET_LOOKBACK_LIVE = 21
        _recent_ret = pd.Series(port_ret_hist[-VOL_TARGET_LOOKBACK_LIVE:])
        if len(_recent_ret) >= 10 and _recent_ret.std() > 0:
            _realized_vol = _recent_ret.std() * np.sqrt(252)
            if not vol_derisked and _realized_vol > CFG["vol_high"]:
                vol_derisked = True
            elif vol_derisked and _realized_vol < CFG["vol_low"]:
                vol_derisked = False
            if vol_derisked:
                final_w = final_w * min(1.0, CFG["vol_high"] / _realized_vol)

        pending_w = final_w.copy()

weights = weights.fillna(0)

latest_target_w = pending_w.copy()

# DIAGNOSTICS EXPORT
DIAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)
alpha_final.to_parquet(os.path.join(DIAG_DIR, "alpha_final.parquet"))
returns[cols].to_parquet(os.path.join(DIAG_DIR, "returns.parquet"))
close[cols].to_parquet(os.path.join(DIAG_DIR, "close.parquet"))
eligible.to_parquet(os.path.join(DIAG_DIR, "eligible.parquet"))
eligible_liquidity.to_parquet(os.path.join(DIAG_DIR, "eligible_liquidity.parquet"))
weights.to_parquet(os.path.join(DIAG_DIR, "weights.parquet"))
print(f"[diag] Signal/return/eligibility panels exported to {DIAG_DIR}")

# 6b. CONFIRMATION GATE ABLATION
print("[6b/8] Confirmation gate ablation (report only)")

trend_gate_B = (
    (price > ma_200) & (rel_strength > 0) & (vol_ann <= VOL_CEILING) &
    (mom_3m > 0) & (mom_6m > 0)
).reindex(returns.index).fillna(False)

trend_gate_C = (
    (mom_3m > 0) & (mom_6m > 0)
).reindex(returns.index).fillna(False)

trend_gate_vol50 = (
    (price > ma_50) & (ma_50 > ma_200) & (rel_strength > 0) & (vol_ann <= 0.50) &
    (mom_3m > 0) & (mom_6m > 0)
).reindex(returns.index).fillna(False)

trend_gate_vol80 = (
    (price > ma_50) & (ma_50 > ma_200) & (rel_strength > 0) & (vol_ann <= 0.80) &
    (mom_3m > 0) & (mom_6m > 0)
).reindex(returns.index).fillna(False)

GATE_VARIANTS = {
    "A: current (P>50>200, RS>0, vol<=65%, M3/M6>0)": eligible,
    "B: simplified (P>200, RS>0, vol<=65%, M3/M6>0)": eligible_liquidity & trend_gate_B,
    "C: momentum-only (M3/M6>0)":                     eligible_liquidity & trend_gate_C,
    "A, vol<=50%":                                     eligible_liquidity & trend_gate_vol50,
    "A, vol<=80%":                                     eligible_liquidity & trend_gate_vol80,
}

VOL_TARGET_LOOKBACK = 21

def run_gate_ablation(elig_mask, exposure_scale=None, top_n=None,
                       sizing_variant=None, vol_target=None, vol_band=None,
                       dd_stop=None, max_weight=None, sector_cap=None,
                       atr_stop_mult=None, stop_stats=None, alpha_source=None):
    alpha_src    = alpha_source if alpha_source is not None else alpha_final
    dd_stop_v    = dd_stop if dd_stop is not None else CFG["dd_stop"]
    max_weight_v = max_weight if max_weight is not None else CFG["max_weight"]
    sector_cap_v = sector_cap if sector_cap is not None else CFG["sector_cap"]

    if vol_band is False:
        vol_band_v = None
    elif vol_band is None:
        vol_band_v = (CFG["vol_low"], CFG["vol_high"])
    else:
        vol_band_v = vol_band
    w_l           = pd.DataFrame(0.0, index=returns.index, columns=cols)
    pend          = pd.Series(0.0, index=cols)
    eq_l          = [1.0]
    frz           = 0
    postf         = False
    port_ret_hist = []
    vol_derisked  = False
    dd_peak_ref   = 1.0
    stop_active   = {}
    stop_exit_count = 0
    _atr_local_cache = {}
    prev_day_w    = pd.Series(0.0, index=cols)

    for t_idx in range(len(returns.index)):
        date = returns.index[t_idx]

        if t_idx < WARMUP:
            w_l.iloc[t_idx] = 0.0
            eq_l.append(eq_l[-1])
            continue

        was_frozen  = frz > 0
        day_w       = pd.Series(0.0, index=cols) if was_frozen else pend
        _check_no_missing_held_returns(day_w, returns[cols].iloc[t_idx], date)
        day_cost    = _daily_trade_cost(prev_day_w, day_w, date)
        day_ret     = (day_w * returns[cols].iloc[t_idx]).sum() - day_cost
        prev_day_w  = day_w
        port_ret_hist.append(day_ret)
        eq_l.append(eq_l[-1] * (1 + day_ret))
        w_l.iloc[t_idx] = day_w
        curr_eq = eq_l[-1]
        dd_peak_ref = max(dd_peak_ref, curr_eq)

        if atr_stop_mult is not None:
            for s in list(stop_active.keys()):
                if pend.get(s, 0) <= 1e-9:
                    stop_active.pop(s, None)
            for s in pend[pend > 1e-9].index:
                if s not in close.columns:
                    continue
                px = close[s].iloc[t_idx]
                if pd.isna(px):
                    continue
                if s not in stop_active:
                    stop_active[s] = {"entry": px, "high": px, "floor": -np.inf}
                st = stop_active[s]
                st["high"] = max(st["high"], px)
                if s not in _atr_local_cache:
                    _atr_local_cache[s] = compute_atr(s)
                atr_series = _atr_local_cache[s]
                atr_val = atr_series.iloc[t_idx] if atr_series is not None else np.nan
                if pd.notna(atr_val):
                    st["floor"] = max(st["floor"], max(st["entry"], st["high"]) - atr_stop_mult * atr_val)
                    if px < st["floor"]:
                        pend[s] = 0.0
                        stop_active.pop(s, None)
                        stop_exit_count += 1

        if was_frozen:

            frz -= 1
            if frz == 0:
                # Final frozen day: rebalance now, at this day's close, instead of
                # adding one more zero-exposure session (see production loop).
                postf = True
                dd_peak_ref = curr_eq
            else:
                pend = pd.Series(0.0, index=cols)
                continue

        if not postf:

            dd_peak = (curr_eq / dd_peak_ref) - 1
            if dd_peak < dd_stop_v:
                frz  = CFG["dd_cooldown"]
                pend = pd.Series(0.0, index=cols)
                continue

        if date in rebal_dates or postf:
            postf     = False
            alpha_row = alpha_src.iloc[t_idx]
            n_slots   = top_n if top_n is not None else CFG["top_n"]

            elig_row  = elig_mask.iloc[t_idx]
            qualified = alpha_row.where(elig_row)
            qualified = qualified[qualified > CFG["score_threshold"]]
            top_syms  = qualified.nlargest(n_slots).index.tolist()

            if len(top_syms) == 0:
                pend = pd.Series(0.0, index=cols)
                continue

            invested_fraction = len(top_syms) / n_slots
            ret_window = returns[top_syms].iloc[max(0, t_idx - LOOKBACK_COV + 1):t_idx + 1]

            active_sizing = sizing_variant if sizing_variant is not None else CFG["sizing_method"]

            if len(ret_window) < 20:
                slot_w = np.ones(len(top_syms)) / len(top_syms)
            elif active_sizing == "equal":
                slot_w = np.ones(len(top_syms)) / len(top_syms)
            elif active_sizing == "cvar_kelly":
                prev_w_arr = pend.reindex(top_syms).fillna(0).values
                alpha_vec  = alpha_row[top_syms].clip(lower=0).values
                cvar_w = cvar_optimal_weights(
                    ret_window.values, prev_w_arr, len(top_syms), alpha_vec
                )
                kel_w  = kelly_downside(alpha_vec, ret_window.values)
                slot_w = 0.75 * cvar_w + 0.25 * kel_w
                slot_w = slot_w / slot_w.sum()
            else:
                slot_w = inverse_vol_weights(ret_window.values)

            target_w = pd.Series(slot_w * invested_fraction, index=top_syms)
            target_w = target_w.clip(lower=CFG["min_weight"])
            target_w = cap_and_redistribute(target_w, max_weight_v)
            if target_w.sum() > invested_fraction:
                target_w *= invested_fraction / target_w.sum()
            new_w = enforce_sector_cap(target_w, top_syms, sector_cap_v)

            candidate = pd.Series(0.0, index=cols)
            candidate[top_syms] = new_w

            relevant = pend[pend.abs() > 1e-9].index.union(candidate[candidate != 0].index)
            final_w = pend.copy()
            for s in relevant:
                if pend[s] > 0 and candidate[s] == 0:
                    final_w[s] = 0.0                             
                elif abs(candidate[s] - pend[s]) > CFG["rebal_band"]:
                    final_w[s] = candidate[s]                    

            active_names = final_w[final_w.abs() > 1e-9].index.tolist()
            final_w = enforce_sector_cap(final_w, active_names, sector_cap_v)

            if final_w.sum() > 1.0:
                final_w /= final_w.sum()

            if exposure_scale is None:
                if regime_series.get(date, "neutral") == "stress":
                    final_w = final_w * CFG["stress_scale"]
        
            else:
                final_w = final_w * exposure_scale.get(date, 1.0)

            if vol_target is not None:
                recent = pd.Series(port_ret_hist[-VOL_TARGET_LOOKBACK:])
                if len(recent) >= 10 and recent.std() > 0:
                    realized_vol = recent.std() * np.sqrt(252)
                    final_w = final_w * min(1.0, vol_target / realized_vol)

            # vol_band: hysteresis pair (vol_low, vol_high)
            if vol_band_v is not None:
                vlo, vhi = vol_band_v
                recent = pd.Series(port_ret_hist[-VOL_TARGET_LOOKBACK:])
                if len(recent) >= 10 and recent.std() > 0:
                    realized_vol = recent.std() * np.sqrt(252)
                    if not vol_derisked and realized_vol > vhi:
                        vol_derisked = True
                    elif vol_derisked and realized_vol < vlo:
                        vol_derisked = False
                    if vol_derisked:
                        final_w = final_w * min(1.0, vhi / realized_vol)

            pend = final_w.copy()

    if stop_stats is not None:
        stop_stats["stop_exits"] = stop_exit_count

    return w_l.fillna(0), eq_l

def gate_backtest_metrics(w_df, eq_list):
    eq  = pd.Series(eq_list[1:], index=returns.index)
    ret = eq.pct_change().dropna()
    n_years = len(eq) / 252
    cagr    = eq.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 and eq.iloc[-1] > 0 else np.nan
    sharpe  = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else np.nan
    max_dd  = (eq / eq.cummax() - 1).min()
    turnover = w_df.diff().abs().sum(axis=1).mean()

    downside = ret[ret < 0]
    sortino  = (ret.mean() / downside.std()) * np.sqrt(252) if len(downside) > 1 and downside.std() > 0 else np.nan
    calmar   = cagr / abs(max_dd) if pd.notna(cagr) and max_dd < 0 else np.nan

    holds = []
    for col in cols:
        active = w_df[col].abs() > 1e-9
        run_id = (active != active.shift()).cumsum()
        for _, grp in active[active].groupby(run_id[active]):
            holds.append(len(grp))
    avg_hold = np.mean(holds) if holds else np.nan

    return dict(cagr=cagr, sharpe=sharpe, sortino=sortino, calmar=calmar,
                max_dd=max_dd, turnover=turnover, avg_hold=avg_hold)

core_qualify = (
    (mom_3m > 0) & (mom_6m > 0) & (rel_strength > 0)
).reindex(returns.index).fillna(False) & eligible_liquidity

def avg_entry_delay(w_df):
    delays = []
    for col in cols:
        qual_dates = returns.index[core_qualify[col]]
        held_dates = returns.index[w_df[col].abs() > 1e-9]
        if len(qual_dates) == 0 or len(held_dates) == 0:
            continue
        first_qual = qual_dates[0]
        after = held_dates[held_dates >= first_qual]
        if len(after) == 0:
            continue
        delay = returns.index.get_loc(after[0]) - returns.index.get_loc(first_qual)
        if delay >= 0:
            delays.append(delay)
    return np.mean(delays) if delays else np.nan

print(f"{'Variant':<52}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'MaxDD':>8}{'Turnover':>10}{'AvgHold':>9}{'EntryDelay':>12}")
for label, mask in GATE_VARIANTS.items():
    if label.startswith("A:"):
        w_variant, eq_variant = weights, rolling_eq
    else:
        w_variant, eq_variant = run_gate_ablation(mask)
    m = gate_backtest_metrics(w_variant, eq_variant)
    delay = avg_entry_delay(w_variant)
    print(f"{label:<52}{m['cagr']:>8.1%}{m['sharpe']:>8.2f}{m['sortino']:>9.2f}{m['calmar']:>8.2f}"
          f"{m['max_dd']:>8.1%}{m['turnover']:>10.3f}{m['avg_hold']:>9.1f}{delay:>12.1f}")
print("Turnover = mean daily sum|Δweight|; AvgHold/EntryDelay in trading days.")
print("Diagnostic only — the live eligible/weights/latest_target_w above are unaffected.")

# 6c. REGIME LAYER ABLATION
print("[6c/8] Regime layer ablation (report only)")

stress_flag_r = stress_flag.reindex(returns.index).fillna(0)

scale_no_control  = pd.Series(1.0, index=returns.index)
scale_stress_only = pd.Series(
    np.where(stress_flag_r == 1, CFG["stress_scale"], 1.0), index=returns.index
)

def _composite_no_macro(date):
    if stress_flag_r.get(date, 0) == 1:
        return "stress"
    return hmm_regime.get(date, "neutral")

regime_no_macro = pd.Series({d: _composite_no_macro(d) for d in returns.index})
scale_hmm_stress_no_macro = regime_no_macro.map(
    {"risk_on": 1.0, "neutral": 1.0, "risk_off": 1.0, "stress": CFG["stress_scale"]}
).fillna(1.0)

REGIME_SCALE_RISKOFF75 = {
    "risk_on": 1.0, "neutral": 1.0, "risk_off": 0.75, "stress": CFG["stress_scale"],
}
scale_riskoff75 = regime_series.map(REGIME_SCALE_RISKOFF75).fillna(1.0)

REGIME_VARIANTS = {
    "1: no regime control (always 100%)":        scale_no_control,
    "2: stress override only (ignore HMM/macro)": scale_stress_only,
    "3: HMM + stress, no macro":                  scale_hmm_stress_no_macro,
    "4: HMM+macro+stress (current)":              None, 
    "5: current + risk_off=75%":                  scale_riskoff75,
}

print(f"{'Variant':<45}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'MaxDD':>8}")
for label, scale in REGIME_VARIANTS.items():
    if scale is None:
        w_variant, eq_variant = weights, rolling_eq
    else:
        w_variant, eq_variant = run_gate_ablation(eligible, exposure_scale=scale)
    m = gate_backtest_metrics(w_variant, eq_variant)
    print(f"{label:<45}{m['cagr']:>8.1%}{m['sharpe']:>8.2f}{m['sortino']:>9.2f}"
          f"{m['calmar']:>8.2f}{m['max_dd']:>8.1%}")
print("Diagnostic only — the live regime scaling used above is unaffected.")

# 6d. HMM STATE-COUNT ABLATION
print("[6d/8] HMM state-count ablation: 2 vs 3 states (report only)")

hmm_regime_2s, n_refits_2s = fit_expanding_hmm(2, ["risk_on", "risk_off"])
print(f"2-state HMM refits: {n_refits_2s}/{len(refit_cuts)-1} converged")
for reg, cnt in hmm_regime_2s.value_counts().sort_index().items():
    print(f"     {reg:>8}: {cnt:>5} days  ({cnt/len(hmm_regime_2s):.1%})")

def _composite_regime_2s(date):
    if stress_flag.get(date, 0) == 1:
        return "stress"
    hmm_r = hmm_regime_2s.get(date, "risk_on")
    ms    = int(macro_score.get(date, 1))
    if ms == 0:
        return "risk_off"
    if ms == 3 and hmm_r != "risk_off":
        return "risk_on"
    return hmm_r

regime_series_2s = pd.Series({d: _composite_regime_2s(d) for d in returns.index})
reg_counts_2s = regime_series_2s.value_counts()
print("Composite regime distribution (2-state HMM + same macro/stress overlay):")
for reg, cnt in reg_counts_2s.sort_index().items():
    print(f"     {reg:>7}: {cnt:>5} days  ({cnt / len(regime_series_2s):.1%})")

scale_riskoff75_2s = regime_series_2s.map(REGIME_SCALE_RISKOFF75).fillna(1.0)

print(f"{'Variant':<45}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'MaxDD':>8}")
w_3s, eq_3s = run_gate_ablation(eligible, exposure_scale=scale_riskoff75)
m_3s = gate_backtest_metrics(w_3s, eq_3s)
print(f"{'3-state HMM, risk_off=75% (from 6c #5)':<45}{m_3s['cagr']:>8.1%}{m_3s['sharpe']:>8.2f}"
      f"{m_3s['sortino']:>9.2f}{m_3s['calmar']:>8.2f}{m_3s['max_dd']:>8.1%}")
w_2s, eq_2s = run_gate_ablation(eligible, exposure_scale=scale_riskoff75_2s)
m_2s = gate_backtest_metrics(w_2s, eq_2s)
print(f"{'2-state HMM, risk_off=75%':<45}{m_2s['cagr']:>8.1%}{m_2s['sharpe']:>8.2f}"
      f"{m_2s['sortino']:>9.2f}{m_2s['calmar']:>8.2f}{m_2s['max_dd']:>8.1%}")
print("If 2-state matches or beats 3-state here, the extra 'neutral' state isn't earning "
      "its complexity for this strategy. Diagnostic only — production stays 3-state.")

# 6e. SELECTION ROBUSTNESS
print("[6e/8] Selection robustness: top-N sensitivity + score dispersion (report only)")

TOPN_VARIANTS = [3, 4, 5, 7]
print(f"{'top_n':<10}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'MaxDD':>8}{'AvgHold':>9}")
for n in TOPN_VARIANTS:
    if n == CFG["top_n"]:
        w_variant, eq_variant = weights, rolling_eq
    else:
        w_variant, eq_variant = run_gate_ablation(eligible, top_n=n)
    m = gate_backtest_metrics(w_variant, eq_variant)
    print(f"{n:<10}{m['cagr']:>8.1%}{m['sharpe']:>8.2f}{m['sortino']:>9.2f}"
          f"{m['calmar']:>8.2f}{m['max_dd']:>8.1%}{m['avg_hold']:>9.1f}")

DISP_HORIZON = 21
dispersion_records = []
for t_idx, date in enumerate(returns.index):
    if date not in rebal_dates:
        continue
    alpha_row = alpha_final.iloc[t_idx]
    elig_row  = eligible.iloc[t_idx]
    qualified = alpha_row.where(elig_row)
    qualified = qualified[qualified > CFG["score_threshold"]]
    top_syms  = qualified.nlargest(CFG["top_n"]).index.tolist()
    if len(top_syms) < 2:
        continue
    top_scores = qualified[top_syms]

    fwd_slice = returns[cols].iloc[t_idx + 1 : t_idx + 1 + DISP_HORIZON]
    w_slice   = weights[cols].iloc[t_idx + 1 : t_idx + 1 + DISP_HORIZON]
    if len(fwd_slice) == 0:
        continue
    fwd_port_ret = (w_slice * fwd_slice).sum(axis=1).sum()
    dispersion_records.append((top_scores.std(), top_scores.mean(), fwd_port_ret))

if len(dispersion_records) >= 5:
    disp_df   = pd.DataFrame(dispersion_records, columns=["dispersion", "avg_level", "fwd_ret"])
    ic_disp   = disp_df["dispersion"].corr(disp_df["fwd_ret"], method="spearman")
    ic_level  = disp_df["avg_level"].corr(disp_df["fwd_ret"], method="spearman")
    print(f"Score dispersion (std of top-{CFG['top_n']} scores) vs next-{DISP_HORIZON}d realized "
          f"portfolio return: Spearman IC={ic_disp:+.3f}  n={len(disp_df)} rebalances")
    print(f"Average score level (mean of top-{CFG['top_n']} scores) vs next-{DISP_HORIZON}d realized "
          f"portfolio return: Spearman IC={ic_level:+.3f}")
    print("Positive IC on avg_level / negative IC on dispersion would support scaling exposure "
          "by signal strength — not implemented, diagnostic only.")
else:
    print("Score dispersion diagnostic: insufficient rebalance history.")

# 6f. SIZING ABLATION
print("[6f/8] Sizing architecture ablation (report only)")

VOL_TARGET_DEFAULT = 0.15

def sizing_backtest_metrics(w_df, eq_list):
    m = gate_backtest_metrics(w_df, eq_list)
    eq  = pd.Series(eq_list[1:], index=returns.index)
    ret = eq.pct_change().dropna()
    var95  = ret.quantile(0.05)
    tail   = ret[ret <= var95]
    cvar95 = tail.mean() if len(tail) > 0 else np.nan
    ann_vol = ret.std() * np.sqrt(252)
    hhi = (w_df ** 2).sum(axis=1)
    concentration = hhi[hhi > 0].mean()
    m.update(cvar95=cvar95, ann_vol=ann_vol, concentration=concentration)
    return m

SIZING_VARIANTS = {
    "A: equal weight":                       dict(sizing_variant="equal",       vol_target=None),
    "B: inverse-vol (current)":               None, 
    f"C: inverse-vol + vol target({VOL_TARGET_DEFAULT:.0%})":
                                               dict(sizing_variant="inverse_vol", vol_target=VOL_TARGET_DEFAULT),
    "D: CVaR/downside-Kelly (experimental)":  dict(sizing_variant="cvar_kelly",  vol_target=None),
}

print(f"{'Variant':<40}{'Sharpe':>8}{'MaxDD':>8}{'CVaR95':>9}{'AnnVol':>8}{'Turnover':>10}{'Concentr.':>10}")
for label, kwargs in SIZING_VARIANTS.items():
    if kwargs is None:
        w_variant, eq_variant = weights, rolling_eq
    else:
        w_variant, eq_variant = run_gate_ablation(eligible, **kwargs)
    m = sizing_backtest_metrics(w_variant, eq_variant)
    print(f"{label:<40}{m['sharpe']:>8.2f}{m['max_dd']:>8.1%}{m['cvar95']:>9.2%}"
          f"{m['ann_vol']:>8.1%}{m['turnover']:>10.3f}{m['concentration']:>10.3f}")
print("Concentration = mean daily HHI (sum of squared weights) on invested days; "
      "lower = more diversified. CVaR95 = mean daily return in the worst 5% of days.")
print(f"Diagnostic only — CFG['sizing_method']='{CFG['sizing_method']}' and the live weights "
      f"above are unaffected.")

# 6g. RISK PARAMETER SENSITIVITY: 
print("[6g/8] Risk parameter sensitivity: circuit breaker / position cap / sector cap (report only)")

DD_STOP_VARIANTS     = [-0.04, -0.06, -0.08, -0.10]
MAX_WEIGHT_VARIANTS  = [0.15, 0.20, 0.25, 0.35]
SECTOR_CAP_VARIANTS  = [0.30, 0.45, 0.60]

def _print_risk_variant_row(label, w_df, eq_list):
    m = gate_backtest_metrics(w_df, eq_list)
    print(f"{label:<22}{m['cagr']:>8.1%}{m['sharpe']:>8.2f}{m['sortino']:>9.2f}"
          f"{m['calmar']:>8.2f}{m['max_dd']:>8.1%}{m['turnover']:>10.3f}")

print(f"{'Variant':<22}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'MaxDD':>8}{'Turnover':>10}")
print("-- Drawdown circuit breaker (dd_stop, current = "
      f"{CFG['dd_stop']:.0%}) --")
for v in DD_STOP_VARIANTS:
    if v == CFG["dd_stop"]:
        w_variant, eq_variant = weights, rolling_eq
    else:
        w_variant, eq_variant = run_gate_ablation(eligible, dd_stop=v)
    _print_risk_variant_row(f"dd_stop={v:.0%}", w_variant, eq_variant)

print(f"-- Position cap (max_weight, current = {CFG['max_weight']:.0%}) --")
for v in MAX_WEIGHT_VARIANTS:
    if v == CFG["max_weight"]:
        w_variant, eq_variant = weights, rolling_eq
    else:
        w_variant, eq_variant = run_gate_ablation(eligible, max_weight=v)
    _print_risk_variant_row(f"max_weight={v:.0%}", w_variant, eq_variant)

print(f"-- Sector cap (sector_cap, current = {CFG['sector_cap']:.0%}) --")
for v in SECTOR_CAP_VARIANTS:
    if v == CFG["sector_cap"]:
        w_variant, eq_variant = weights, rolling_eq
    else:
        w_variant, eq_variant = run_gate_ablation(eligible, sector_cap=v)
    _print_risk_variant_row(f"sector_cap={v:.0%}", w_variant, eq_variant)
print("Diagnostic only — the live dd_stop/max_weight/sector_cap above are unaffected.")

# 6h. ATR STOP-LOSS SENSITIVITY:
print("[6h/8] ATR stop-loss multiplier sensitivity (report only, EOD approximation)")

def _position_runs(w_df, sym):
    active = w_df[sym].abs() > 1e-9
    run_id = (active != active.shift()).cumsum()
    runs = []
    for rid, grp in active[active].groupby(run_id[active]):
        idx = grp.index
        runs.append((idx[0], idx[-1]))
    return runs

_atr_cache = {}

def atr_stop_diagnostic(mult):
    total_runs, triggered = 0, 0
    ret_diffs = []
    for sym in cols:
        if sym not in close.columns:
            continue
        if sym not in _atr_cache:
            _atr_cache[sym] = compute_atr(sym)
        atr_series = _atr_cache[sym]
        if atr_series is None:
            continue
        c = close[sym]
        for start, end in _position_runs(weights, sym):
            run_idx = returns.index[(returns.index >= start) & (returns.index <= end)]
            if len(run_idx) < 2:
                continue
            total_runs += 1
            entry_price   = c.reindex(run_idx).iloc[0]
            trailing_high = entry_price
            stop_level    = -np.inf
            trigger_date  = None
            for d in run_idx[1:]:
                px, atr_val = c.get(d, np.nan), atr_series.get(d, np.nan)
                if pd.isna(px) or pd.isna(atr_val):
                    continue
                trailing_high = max(trailing_high, px)
                stop_level = max(stop_level, max(entry_price, trailing_high) - mult * atr_val)
                if px < stop_level:
                    trigger_date = d
                    break
            if trigger_date is not None:
                triggered += 1
                model_exit_ret = c.get(end, np.nan) / entry_price - 1
                stop_exit_ret  = c.get(trigger_date, np.nan) / entry_price - 1
                ret_diffs.append(stop_exit_ret - model_exit_ret)
    return dict(
        total_runs=total_runs, triggered=triggered,
        hit_rate=triggered / total_runs if total_runs else np.nan,
        avg_ret_diff=np.mean(ret_diffs) if ret_diffs else np.nan,
    )

ATR_MULT_VARIANTS = [1.5, 2.0, 2.5, 3.0, 3.5]
print(f"{'ATR mult':<12}{'Runs':>7}{'Triggered':>11}{'HitRate':>10}{'AvgRetDiff':>13}")
for mult in ATR_MULT_VARIANTS:
    r = atr_stop_diagnostic(mult)
    tag = "  <- current" if mult == CFG["atr_stop_mult"] else ""
    print(f"{mult:<12}{r['total_runs']:>7}{r['triggered']:>11}{r['hit_rate']:>10.1%}"
          f"{r['avg_ret_diff']:>+13.2%}{tag}")
print("EOD approximation, diagnostic only — the live target-portfolio weights are unaffected.")

# 6i. ATR STOP-LOSS FULL PORTFOLIO BACKTEST:
print("[6i/8] ATR stop-loss full portfolio backtest: no-stop vs 2.0x/2.5x/3.0x (report only)")

ATR_BACKTEST_VARIANTS = {"No stop": None, "2.0x ATR": 2.0, "2.5x ATR (current)": 2.5, "3.0x ATR": 3.0}
print(f"{'Variant':<22}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'MaxDD':>8}"
      f"{'Turnover':>10}{'StopExits':>11}")
for label, mult in ATR_BACKTEST_VARIANTS.items():
    stats = {}
    if mult is None:
        w_variant, eq_variant = weights, rolling_eq
    else:
        w_variant, eq_variant = run_gate_ablation(eligible, atr_stop_mult=mult, stop_stats=stats)
    m = gate_backtest_metrics(w_variant, eq_variant)
    print(f"{label:<22}{m['cagr']:>8.1%}{m['sharpe']:>8.2f}{m['sortino']:>9.2f}{m['calmar']:>8.2f}"
          f"{m['max_dd']:>8.1%}{m['turnover']:>10.3f}{stats.get('stop_exits', 0):>11}")
print("StopExits = count of ATR-triggered forced exits across the backtest (whipsaw proxy)")

# 6j. PORTFOLIO VOLATILITY TARGETING (vol_low/vol_high)
print("[6j/8] Portfolio volatility targeting: vol_low/vol_high band (NOW LIVE in section 6)")

vol_band_current = (CFG["vol_low"], CFG["vol_high"])
print(f"{'Variant':<36}{'CAGR':>8}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'MaxDD':>8}{'AnnVol':>8}")
w_live, eq_live = weights, rolling_eq
m_live = sizing_backtest_metrics(w_live, eq_live)
vb_label = f"vol_band=({CFG['vol_low']:.0%},{CFG['vol_high']:.0%}) — LIVE"
print(f"{vb_label:<36}{m_live['cagr']:>8.1%}{m_live['sharpe']:>8.2f}{m_live['sortino']:>9.2f}"
      f"{m_live['calmar']:>8.2f}{m_live['max_dd']:>8.1%}{m_live['ann_vol']:>8.1%}")
w_novol, eq_novol = run_gate_ablation(eligible, vol_band=False)
m_novol = sizing_backtest_metrics(w_novol, eq_novol)
print(f"{'No vol control (counterfactual)':<36}{m_novol['cagr']:>8.1%}{m_novol['sharpe']:>8.2f}"
      f"{m_novol['sortino']:>9.2f}{m_novol['calmar']:>8.2f}{m_novol['max_dd']:>8.1%}"
      f"{m_novol['ann_vol']:>8.1%}")

# 7. TRANSACTION COST MODEL
print("[7/8] Computing transaction costs (fixed spread + sqrt market-impact approximation)")

turnover_per_sym = weights.diff().abs()
cost_daily       = compute_transaction_costs(weights)

port_ret_gross = (weights * returns[cols]).sum(axis=1)
port_ret = (port_ret_gross - cost_daily).replace([np.inf, -np.inf], np.nan).fillna(0)

nonzero_mask = port_ret != 0
first_nz     = port_ret[nonzero_mask].index[0] if nonzero_mask.any() else port_ret.index[0]
port_ret     = port_ret.loc[first_nz:]

# 8. PERFORMANCE METRICS
ANN = 252
equity   = (1 + port_ret).cumprod()
dd_port  = equity / equity.cummax() - 1
turnover = turnover_per_sym.sum(axis=1)

sharpe       = port_ret.mean() / port_ret.std() * np.sqrt(ANN) if port_ret.std() > 0 else 0
down_ret     = port_ret[port_ret < 0]
downside_dev = np.sqrt((down_ret ** 2).mean()) * np.sqrt(ANN) if len(down_ret) > 0 else 1e-9
sortino      = (port_ret.mean() * ANN) / downside_dev
max_dd       = dd_port.min()
total_ret    = equity.iloc[-1] - 1  
ann_ret      = (1 + total_ret) ** (ANN / len(port_ret)) - 1
calmar       = ann_ret / abs(max_dd) if max_dd != 0 else 0
ret_1m       = (1 + port_ret.tail(21)).prod() - 1
ann_vol      = port_ret.std() * np.sqrt(ANN)

bench_ret_bt = bench_ret.reindex(port_ret.index).fillna(0)
bench_equity = (1 + bench_ret_bt).cumprod()
dd_bench     = bench_equity / bench_equity.cummax() - 1

excess_ret   = port_ret - bench_ret_bt
# Relative wealth (equity/bench_equity), not compounded return differences —
# (1+r_p).cumprod() vs (1+r_b).cumprod() diverge from the true ratio over a
# multi-year backtest, so cumulative/period active return is computed as an
# actual wealth ratio throughout.
cum_active      = equity / bench_equity
port_1m_growth  = (1 + port_ret.tail(21)).prod()
bench_1m_growth = (1 + bench_ret_bt.tail(21)).prod()
bench_1m     = bench_1m_growth - 1
alpha_1m     = port_1m_growth / bench_1m_growth - 1
info_ratio   = excess_ret.mean() / excess_ret.std() * np.sqrt(ANN) if excess_ret.std() > 0 else 0

total_cost       = cost_daily.reindex(port_ret.index).sum()
monthly_cost_est = total_cost / (len(port_ret) / 21)
avg_weekly_to    = turnover.reindex(port_ret.index).resample("W").sum().mean()

port_cvar_95 = float(-port_ret[port_ret <= np.percentile(port_ret, 5)].mean())

print(f"\n{'='*65}")
print(f"NYSE MOMENTUM")
print(f"{'='*65}")
print(f"  Universe ({len(symbols)} names): {', '.join(symbols)}")
print(f"{'─'*65}")
print(f"  Total Return:          {total_ret:>10.2%}")
print(f"  Annual Return:         {ann_ret:>10.2%}")
print(f"  Annual Volatility:     {ann_vol:>10.2%}")
print(f"  Sharpe Ratio:          {sharpe:>10.4f}   {'✓' if sharpe >= 1.0 else 'Delta'} target ≥ 1.0")
print(f"  Sortino Ratio:         {sortino:>10.4f}   {'✓' if sortino >= 1.5 else 'Delta'} target ≥ 1.5")
print(f"  Calmar Ratio:          {calmar:>10.4f}   {'✓' if calmar >= 1.0 else 'Delta'} target ≥ 1.0")
print(f"  Information Ratio:     {info_ratio:>10.4f}")
print(f"  Max Drawdown:          {max_dd:>10.2%}   {'✓' if max_dd >= -0.15 else 'Delta'} target > -15%")
print(f"  Realised CVaR (95%):   {port_cvar_95:>10.4f}   daily  {'✓' if port_cvar_95 < 0.035 else 'Delta'} target < 0.035")
print(f"  1-Month Return:        {ret_1m:>10.2%}")
print(f"  Win Rate:              {(port_ret > 0).mean():>10.1%}")
print(f"  Total Costs Paid:      {total_cost:>10.2%}  (spread + impact)")
print(f"  Cost / Month (est):    {monthly_cost_est:>10.2%}")
print(f"  Avg Weekly TO:         {avg_weekly_to:>10.4f}   {'✓' if avg_weekly_to < 0.20 else 'Delta'} target < 0.20")
print(f"{'─'*65}")
print(f"  1M Active Ret vs S&P:  {alpha_1m:>10.2%}")
print(f"  Benchmark 1M:          {bench_1m:>10.2%}")

print(f"\n{'─'*65}")
print("SINGLE-FACTOR MARKET REGRESSION (OLS, raw returns — no risk-free adjustment)")
print(f"{'─'*65}")

_idx = port_ret.index
_F   = pd.DataFrame({
    "market": bench_ret_bt.reindex(_idx).fillna(0),
}, index=_idx).dropna()
_y = port_ret.reindex(_F.index).dropna()
_F = _F.reindex(_y.index)

X                         = np.column_stack([np.ones(len(_F)), _F.values])
betas, _, _, _            = lstsq(X, _y.values, rcond=None)
alpha_daily, beta_market  = betas

fitted    = X @ betas
resid     = _y.values - fitted
idio_vol  = resid.std() * np.sqrt(ANN)
r_squared = 1 - np.var(resid) / np.var(_y.values)

print(f"  Alpha (daily):         {alpha_daily:>10.4%}")
print(f"  Beta vs S&P 500:       {beta_market:>10.4f}")
print(f"  Idiosyncratic Vol:     {idio_vol:>10.2%}")
print(f"  R² (mkt factor):       {r_squared:>10.4f}")
print(f"\n{'─'*65}")
print("CORRELATION / CONCENTRATION RISK (monitoring only, not a trade veto)")
print(f"{'─'*65}")

_held_syms = [s for s in latest_target_w.index if abs(latest_target_w[s]) > 0.001]
CORR_LOOKBACK = 63
if len(_held_syms) >= 2:
    _corr_window = returns[_held_syms].iloc[-CORR_LOOKBACK:].dropna(how="all")
    _corr_matrix = _corr_window.corr()
    _off_diag    = _corr_matrix.where(~np.eye(len(_held_syms), dtype=bool))
    max_pairwise_corr = float(_off_diag.max().max())
    avg_pairwise_corr = float(_off_diag.stack().mean())

    _w_vec   = latest_target_w[_held_syms].values
    _cov_ann = _corr_window.cov().values * 252
    port_var = float(_w_vec @ _cov_ann @ _w_vec)
    if port_var > 0:
        marginal_contrib = _cov_ann @ _w_vec
        var_contrib       = _w_vec * marginal_contrib / port_var
        max_var_contrib   = float(var_contrib.max())
    else:
        max_var_contrib = np.nan

    if _corr_window.shape[0] > len(_held_syms) and _corr_matrix.notna().all().all():
        eigvals = np.linalg.eigvalsh(_corr_matrix.values)
        pc1_explained = float(eigvals.max() / eigvals.sum())
    else:
        pc1_explained = np.nan

    print(f"  Held names:            {_held_syms}")
    print(f"  Max pairwise corr:     {max_pairwise_corr:>10.2f}")
    print(f"  Avg pairwise corr:     {avg_pairwise_corr:>10.2f}")
    print(f"  Max var. contribution: {max_var_contrib:>10.2f}   (single name's share of "
          f"total portfolio variance)")
    print(f"  PC1 variance explained:{pc1_explained:>10.2f}   (fraction of correlation-matrix "
          f"variance in the first principal component — high = one dominant economic bet "
          f"despite sector caps)")
else:
    max_pairwise_corr = avg_pairwise_corr = max_var_contrib = pc1_explained = np.nan
    print(f"Fewer than 2 held names ({_held_syms}) — correlation diagnostic skipped.")
    print("Sector caps limit sector-LABEL concentration, not correlation — two different ")

print(f"\n{'─'*65}")
print("COMBINATORIAL TIME-SLICE STABILITY")
print(f"{'─'*65}")

N_SPLITS = CFG["cpcv_n"]
K_TEST   = CFG["cpcv_k"]
EMBARGO  = CFG["embargo_days"]

# Sliced from the already net-of-cost port_ret (not reconstructed from
# weights x gross returns) so this reads as validation of the finished,
# tradeable strategy rather than a pre-cost approximation of it.
_port_ret_arr = port_ret.reindex(weights.index).fillna(0)
T_cv     = len(_port_ret_arr)
fold_sz  = T_cv // N_SPLITS

combos        = list(itertools.combinations(range(N_SPLITS), K_TEST))
test_slice_sh = []
rest_slice_sh = []

for test_folds in combos:
    test_idx  = []
    train_idx = []
    for f in range(N_SPLITS):
        s_f = f * fold_sz
        e_f = (f + 1) * fold_sz if f < N_SPLITS - 1 else T_cv
        if f in test_folds:
            test_idx.extend(range(s_f, e_f))
        else:
            is_adj = any(
                abs(s_f - tf * fold_sz) <= EMBARGO or
                abs(e_f - (tf + 1) * fold_sz) <= EMBARGO
                for tf in test_folds
            )
            if is_adj:
                emb_s = max(0, s_f - EMBARGO)
                emb_e = min(T_cv, e_f + EMBARGO)
                train_idx.extend(
                    range(emb_e, e_f) if s_f < emb_s else range(s_f, min(e_f, emb_s))
                )
            else:
                train_idx.extend(range(s_f, e_f))

    if not test_idx or not train_idx:
        continue

    rest_r  = _port_ret_arr.iloc[train_idx]
    rest_sh = rest_r.mean() / rest_r.std() * np.sqrt(ANN) if rest_r.std() > 0 else 0

    test_r  = _port_ret_arr.iloc[test_idx]
    test_sh = test_r.mean() / test_r.std() * np.sqrt(ANN) if test_r.std() > 0 else 0

    rest_slice_sh.append(rest_sh)
    test_slice_sh.append(test_sh)

oos_arr        = np.array(test_slice_sh)
is_arr         = np.array(rest_slice_sh)
neg_slice_frac = (oos_arr < 0).mean()

print(f"  Combinations:            {len(combos)} (C({N_SPLITS},{K_TEST}))")
print(f"  Train/test buffer:       {EMBARGO} days (no refitting happens here, so this "
      f"is a slice-adjacency gap, not a leakage-prevention embargo)")
print(f"  Mean Sharpe (rest-of-history): {is_arr.mean():.4f}")
print(f"  Mean Sharpe (test slice):      {oos_arr.mean():.4f}")
print(f"  Test-slice Sharpe StdDev:      {oos_arr.std():.4f}")
print(f"  Fraction of test slices <0:    {neg_slice_frac:.3f}   "
      f"{'LOW' if neg_slice_frac < 0.30 else 'Delta HIGH'}")

if neg_slice_frac > 0.50:
    raise RuntimeError(
        f"{neg_slice_frac:.2f} of time slices show negative Sharpe"
        f"before dashboard. Review signal construction."
    )

# 8b. ALPHA IC / SIGNAL DECAY:
print(f"\n{'─'*65}")
print("ALPHA IC / SIGNAL DECAY")
print(f"{'─'*65}")

def rank_ic_series(signal_df, horizon):
    fwd = price.pct_change(horizon, fill_method=None).shift(-horizon).reindex(returns.index)
    masked = signal_df.reindex_like(eligible_liquidity).where(eligible_liquidity)
    return masked.corrwith(fwd, axis=1, method="spearman").dropna()

IC_HORIZONS_NAMED = {"1w": 5, "1m": 21, "3m": 63}
print("Rank IC by horizon (mean, std, IC-IR=mean/std, hit-rate):")
for label, h in IC_HORIZONS_NAMED.items():
    ic = rank_ic_series(alpha_final, h)
    if len(ic) == 0:
        print(f"  {label:<4} insufficient data")
        continue
    ic_ir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
    print(f"  {label:<4} mean IC={ic.mean():+.4f}  std={ic.std():.4f}  IC-IR={ic_ir:+.3f}  "
          f"hit-rate={(ic > 0).mean():.1%}  n={len(ic)}")

DECAY_HORIZONS = [5, 10, 20, 40, 60]
print("Signal decay — mean rank IC vs holding horizon:")
decay_ics = {}
for h in DECAY_HORIZONS:
    ic = rank_ic_series(alpha_final, h)
    decay_ics[h] = ic.mean() if len(ic) else np.nan
    print(f"  {h:>3}d: IC={decay_ics[h]:+.4f}")
_valid_decay = {h: v for h, v in decay_ics.items() if pd.notna(v)}
if _valid_decay:
    peak_h = max(_valid_decay, key=lambda h: _valid_decay[h])
    print(f"  Peak IC at {peak_h}d — compare against actual AvgHold reported in the "
          f"selection/sizing ablations above to judge whether rebalance frequency is "
          f"leaving predictability on the table or churning past it.")

QUANT_HORIZON = 21
fwd21 = price.pct_change(QUANT_HORIZON, fill_method=None).shift(-QUANT_HORIZON).reindex(returns.index)
quint_rows = []
for t_idx, date in enumerate(returns.index):
    if date not in rebal_dates:
        continue
    row = alpha_final.iloc[t_idx].where(eligible_liquidity.iloc[t_idx]).dropna()
    if len(row) < 15:
        continue
    fwd_row = fwd21.iloc[t_idx]
    try:
        qcuts = pd.qcut(row, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    except ValueError:
        continue
    for sym in row.index:
        r = fwd_row.get(sym, np.nan)
        if pd.notna(r):
            quint_rows.append((int(qcuts[sym]), r))

print(f"Forward-return quintiles (Q5=highest score...Q1=lowest), fwd {QUANT_HORIZON}d return:")
if quint_rows:
    qdf = pd.DataFrame(quint_rows, columns=["quintile", "fwd_ret"])
    qmeans = qdf.groupby("quintile")["fwd_ret"].mean().sort_index()
    for q, r in qmeans.items():
        print(f"  Q{q}: mean fwd {QUANT_HORIZON}d return = {r:+.2%}  (n={int((qdf['quintile']==q).sum())})")
    monotonic = all(qmeans.iloc[i] <= qmeans.iloc[i + 1] for i in range(len(qmeans) - 1))
    print(f"  Monotonic Q1<=Q2<=...<=Q5: {'YES' if monotonic else 'NO'}")
else:
    print("  Insufficient data for quintile analysis.")

# 8c. PARAMETER SENSITIVITY:
print(f"\n{'─'*65}")
print("PARAMETER SENSITIVITY — SKIP PERIOD (0/5/10/21d)")
print(f"{'─'*65}")

def alpha_variant_for_skip(skip_days):
    m1  = price.shift(skip_days) / price.shift(skip_days + 21)  - 1
    m3  = price.shift(skip_days) / price.shift(skip_days + 63)  - 1
    m6  = price.shift(skip_days) / price.shift(skip_days + 126) - 1
    m12 = price.shift(skip_days) / price.shift(skip_days + 252) - 1
    raw = 0.15 * m1 + 0.30 * m3 + 0.30 * m6 + 0.15 * m12
    return cs_rank(raw / vol_63)

for skip in [0, 5, 10, 21]:
    tag = "  <- current" if skip == SKIP_DAYS else ""
    rank_ic_stats(alpha_variant_for_skip(skip), f"skip={skip}d{tag}")
print("A plateau across skip values (not one sharp peak) is what supports the current "
      "choice; a single magical value would be a classic overfit signature.")

# 8d. STACKED MODULE ABLATION:
print(f"\n{'─'*65}")
print("STACKED MODULE ABLATION — marginal contribution of each layer")
print(f"{'─'*65}")

flat_scale = pd.Series(1.0, index=returns.index)
alpha_A = sig_mom
alpha_B = 0.90 * sig_mom.fillna(0) + 0.10 * sig_rs.fillna(0)

STACK_VARIANTS = {
    "A: Momentum only": dict(
        elig_mask=eligible_liquidity, alpha_source=alpha_A, exposure_scale=flat_scale,
        sizing_variant="equal", max_weight=1.0, sector_cap=1.0, dd_stop=-0.99,
        vol_band=False, atr_stop_mult=None),
    "B: + RS": dict(
        elig_mask=eligible_liquidity, alpha_source=alpha_B, exposure_scale=flat_scale,
        sizing_variant="equal", max_weight=1.0, sector_cap=1.0, dd_stop=-0.99,
        vol_band=False, atr_stop_mult=None),
    "C: + Confirmation": dict(
        elig_mask=eligible, alpha_source=alpha_final, exposure_scale=flat_scale,
        sizing_variant="equal", max_weight=1.0, sector_cap=1.0, dd_stop=-0.99,
        vol_band=False, atr_stop_mult=None),
    "D: + Regime": dict(
        elig_mask=eligible, alpha_source=alpha_final, exposure_scale=None,
        sizing_variant="equal", max_weight=1.0, sector_cap=1.0, dd_stop=-0.99,
        vol_band=False, atr_stop_mult=None),
    "E: + Inverse-Vol sizing": dict(
        elig_mask=eligible, alpha_source=alpha_final, exposure_scale=None,
        sizing_variant=None, max_weight=1.0, sector_cap=1.0, dd_stop=-0.99,
        vol_band=False, atr_stop_mult=None),
    "F: + Risk controls (current, LIVE)": None,
}

print(f"{'Variant':<38}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'Turnover':>10}{'CVaR95':>9}")
for label, kwargs in STACK_VARIANTS.items():
    if kwargs is None:
        w_variant, eq_variant = weights, rolling_eq
    else:
        elig_arg = kwargs.pop("elig_mask")
        w_variant, eq_variant = run_gate_ablation(elig_arg, **kwargs)
    m = sizing_backtest_metrics(w_variant, eq_variant)
    print(f"{label:<38}{m['cagr']:>8.1%}{m['sharpe']:>8.2f}{m['max_dd']:>8.1%}"
          f"{m['turnover']:>10.3f}{m['cvar95']:>9.2%}")

# 8e. WALK-FORWARD OOS — FIXED CONFIG:
print(f"\n{'─'*65}")
print("WALK-FORWARD OOS — FIXED CONFIG (current parameters, expanding annual folds)")
print(f"{'─'*65}")

def _slice_metrics(ret_slice):
    if len(ret_slice) < 20 or ret_slice.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, max_dd=np.nan, n=len(ret_slice))
    eq = (1 + ret_slice).cumprod()
    n_years = len(ret_slice) / 252
    sharpe = ret_slice.mean() / ret_slice.std() * np.sqrt(252)
    cagr = eq.iloc[-1] ** (1 / n_years) - 1 if eq.iloc[-1] > 0 else np.nan
    max_dd = (eq / eq.cummax() - 1).min()
    return dict(sharpe=sharpe, cagr=cagr, max_dd=max_dd, n=len(ret_slice))

MIN_TRAIN_YEARS = 2
_years   = sorted(set(returns.index.year))
wf_years = _years[MIN_TRAIN_YEARS:]

print(f"{'Test Year':<12}{'TrainSharpe':>13}{'OOS Sharpe':>12}{'OOS CAGR':>10}{'OOS MaxDD':>10}{'n(days)':>9}")
wf_fixed_rows = []
for y in wf_years:
    train_dates = returns.index[returns.index.year < y]
    test_dates  = returns.index[returns.index.year == y]
    if len(train_dates) < 100 or len(test_dates) < 20:
        continue
    m_train = _slice_metrics(port_ret.reindex(train_dates).dropna())
    m_test  = _slice_metrics(port_ret.reindex(test_dates).dropna())
    wf_fixed_rows.append((y, m_train["sharpe"], m_test["sharpe"]))
    print(f"{y:<12}{m_train['sharpe']:>13.2f}{m_test['sharpe']:>12.2f}"
          f"{m_test['cagr']:>10.1%}{m_test['max_dd']:>10.1%}{m_test['n']:>9}")

if wf_fixed_rows:
    _oos1 = [r[2] for r in wf_fixed_rows if pd.notna(r[2])]
    if _oos1:
        print(f"Mean OOS Sharpe across {len(_oos1)} folds: {np.mean(_oos1):.2f}  "
              f"(std {np.std(_oos1):.2f})")
else:
    print("Insufficient history for annual walk-forward folds.")

# 8f. WALK-FORWARD OOS — TRAIN -> OPTIMISE -> FREEZE -> TEST:
print(f"\n{'─'*65}")
print("WALK-FORWARD OOS — TRAIN -> OPTIMISE -> FREEZE -> TEST (scoped grid search)")
print(f"{'─'*65}")
print("Scope: searches top_n x vol_ceiling x atr_stop_mult only")
print("Candidates are ranked and tested on NET-of-cost returns (fixed spread + sqrt market-impact "
      "approximation applied to each candidate's own turnover) — gross Sharpe would favour "
      "high-turnover configs the live cost model would penalise.")

WF_TOPN = [3, 5, 7]
WF_VOLCEIL_MASKS = {
    0.50: eligible_liquidity & trend_gate_vol50,
    0.65: eligible,
    0.80: eligible_liquidity & trend_gate_vol80,
}
WF_ATR = [None, 2.0, 2.5, 3.0]

n_combos = len(WF_TOPN) * len(WF_VOLCEIL_MASKS) * len(WF_ATR)
print(f"Running {n_combos} candidate configs once over full history (cached, then sliced per fold) ...")
wf_combo_cache = {}
for _n in WF_TOPN:
    for _vc, _mask in WF_VOLCEIL_MASKS.items():
        for _atr in WF_ATR:
            _w_c, _eq_c = run_gate_ablation(_mask, top_n=_n, atr_stop_mult=_atr)
            _eq_series = pd.Series(_eq_c[1:], index=returns.index)
            # run_gate_ablation() already updates NAV net of _daily_trade_cost();
            # do not subtract transaction costs a second time here.
            _net_ret = _eq_series.pct_change(fill_method=None).fillna(0)
            wf_combo_cache[(_n, _vc, _atr)] = _net_ret

print(f"{'Test Year':<12}{'Chosen(topN,vol,atr)':<24}{'TrainSharpe':>13}{'OOS Sharpe':>12}"
      f"{'OOS CAGR':>10}{'OOS MaxDD':>10}")
wf_opt_rows = []
for y in wf_years:
    train_dates = returns.index[returns.index.year < y]
    test_dates  = returns.index[returns.index.year == y]
    if len(train_dates) < 100 or len(test_dates) < 20:
        continue
    best_combo, best_sharpe = None, -np.inf
    for combo, ret_series in wf_combo_cache.items():
        train_ret = ret_series.reindex(train_dates).dropna()
        if len(train_ret) < 50 or train_ret.std() == 0:
            continue
        sh = train_ret.mean() / train_ret.std() * np.sqrt(252)
        if sh > best_sharpe:
            best_sharpe, best_combo = sh, combo
    if best_combo is None:
        continue
    test_ret = wf_combo_cache[best_combo].reindex(test_dates).dropna()
    m_test = _slice_metrics(test_ret)
    _n, _vc, _atr = best_combo
    combo_str = f"n={_n},vol={_vc:.0%},atr={_atr if _atr else 'off'}"
    wf_opt_rows.append((y, best_sharpe, m_test["sharpe"]))
    print(f"{y:<12}{combo_str:<24}{best_sharpe:>13.2f}{m_test['sharpe']:>12.2f}"
          f"{m_test['cagr']:>10.1%}{m_test['max_dd']:>10.1%}")

if wf_opt_rows:
    _oos2 = [r[2] for r in wf_opt_rows if pd.notna(r[2])]
    if _oos2:
        print(f"Mean OOS Sharpe across {len(_oos2)} folds: {np.mean(_oos2):.2f}  "
              f"(std {np.std(_oos2):.2f})")
        print("Compare against the fixed-config walk-forward above: if optimising per fold "
              "isn't meaningfully better OOS, the extra search isn't earning its complexity.")

# 8g. RECENT-PERIOD ROBUSTNESS CHECK — LAST 12 MONTHS
print(f"\n{'─'*65}")
print("RECENT-PERIOD ROBUSTNESS CHECK — LAST 12 MONTHS")
print(f"{'─'*65}")

def _holdout_metrics(ret_slice, bench_slice, turnover_slice):
    if len(ret_slice) < 20 or ret_slice.std() == 0:
        return None
    eq        = (1 + ret_slice).cumprod()
    n_years   = len(ret_slice) / ANN
    tot_ret   = eq.iloc[-1] - 1
    ann_ret_  = (1 + tot_ret) ** (1 / n_years) - 1
    ann_vol_  = ret_slice.std() * np.sqrt(ANN)
    sharpe_   = ret_slice.mean() / ret_slice.std() * np.sqrt(ANN)
    down_     = ret_slice[ret_slice < 0]
    dd_dev    = np.sqrt((down_ ** 2).mean()) * np.sqrt(ANN) if len(down_) > 0 else 1e-9
    sortino_  = (ret_slice.mean() * ANN) / dd_dev
    dd_curve  = eq / eq.cummax() - 1
    max_dd_   = dd_curve.min()
    calmar_   = ann_ret_ / abs(max_dd_) if max_dd_ != 0 else 0
    cvar95_   = float(-ret_slice[ret_slice <= np.percentile(ret_slice, 5)].mean())
    win_rate_ = (ret_slice > 0).mean()
    avg_wto_  = turnover_slice.reindex(ret_slice.index).resample("W").sum().mean()

    bench_aligned = bench_slice.reindex(ret_slice.index).fillna(0)
    bench_eq      = (1 + bench_aligned).cumprod()
    bench_tot     = bench_eq.iloc[-1] - 1

    Xb        = np.column_stack([np.ones(len(bench_aligned)), bench_aligned.values])
    betas, *_ = lstsq(Xb, ret_slice.values, rcond=None)
    beta_     = betas[1]
    fitted    = Xb @ betas
    resid     = ret_slice.values - fitted
    r2_       = 1 - np.var(resid) / np.var(ret_slice.values) if np.var(ret_slice.values) > 0 else np.nan

    return dict(n=len(ret_slice), total_ret=tot_ret, ann_ret=ann_ret_, ann_vol=ann_vol_,
                sharpe=sharpe_, sortino=sortino_, calmar=calmar_, max_dd=max_dd_,
                cvar95=cvar95_, win_rate=win_rate_, avg_wto=avg_wto_,
                bench_total_ret=bench_tot, excess_total_ret=tot_ret - bench_tot,
                beta=beta_, r_squared=r2_)

HOLDOUT_MONTHS = 12
_holdout_start = port_ret.index[-1] - pd.DateOffset(months=HOLDOUT_MONTHS)
_is_ret        = port_ret[port_ret.index < _holdout_start]
_oos_ret       = port_ret[port_ret.index >= _holdout_start]

_is_m  = _holdout_metrics(_is_ret, bench_ret_bt, turnover)
_oos_m = _holdout_metrics(_oos_ret, bench_ret_bt, turnover)

if _oos_m is None:
    print(f"Fewer than 20 trading days since {_holdout_start.date()} — holdout too short to score.")
else:
    print(f"Holdout window: {_oos_ret.index[0].date()} -> {_oos_ret.index[-1].date()}  "
          f"({_oos_m['n']} trading days, ~{HOLDOUT_MONTHS}mo)")
    print(f"Everything before {_holdout_start.date()} is the 'development' period this "
          f"config was effectively shaped against. This window overlaps the period used "
          f"during parameter tuning, so despite the 'holdout' framing it is a robustness "
          f"check, not clean OOS evidence — see PARAMETER FREEZE below for the real cutover.")
    print(f"\n  {'Metric':<20}{'Pre-Holdout':>14}{'Holdout (OOS)':>16}")
    _rows = [
        ("CAGR",             "ann_ret",          "{:.1%}"),
        ("Sharpe",           "sharpe",           "{:.2f}"),
        ("Sortino",          "sortino",          "{:.2f}"),
        ("Calmar",           "calmar",           "{:.2f}"),
        ("Max Drawdown",     "max_dd",           "{:.1%}"),
        ("Ann. Volatility",  "ann_vol",          "{:.1%}"),
        ("CVaR (95%)",       "cvar95",           "{:.4f}"),
        ("Win Rate",         "win_rate",         "{:.1%}"),
        ("Avg Weekly TO",    "avg_wto",          "{:.3f}"),
        ("Beta vs S&P",      "beta",             "{:.2f}"),
        ("R² (mkt factor)",  "r_squared",        "{:.2f}"),
        ("Excess Ret vs S&P","excess_total_ret", "{:+.1%}"),
    ]
    for label, key, fmt in _rows:
        is_val  = fmt.format(_is_m[key])  if _is_m  and pd.notna(_is_m.get(key))  else "n/a"
        oos_val = fmt.format(_oos_m[key]) if pd.notna(_oos_m.get(key)) else "n/a"
        print(f"  {label:<20}{is_val:>14}{oos_val:>16}")
    print(f"\n  Portfolio return over holdout: {_oos_m['total_ret']:+.1%}   "
          f"S&P 500 over same window: {_oos_m['bench_total_ret']:+.1%}")

PARAMETER_FREEZE_DATE = "2026-09-17"
print(f"\n{'─'*65}")
print(f"  PARAMETER FREEZE: {PARAMETER_FREEZE_DATE}")
print(f"{'─'*65}")
print(f"top_n/dd_stop/max_weight (and the fixes in this pass: mandatory-exit "
      f"rebalancing, equity OHLCV ffill removed entirely (NaNs preserved, was "
      f"a capped 2-day ffill) with a phantom-holiday-row guard, forward-return "
      f"HMM ranking, HMM model/scaler consistency, factor-health-free regime "
      f"ablation, net-of-cost 8f grid search, net-NAV risk engine (now shared "
      f"by production and run_gate_ablation), W-FRI last-session rebalance "
      f"dates, circuit-freeze exposure resumption, frozen candidate universe "
      f"alongside the price snapshot, relative-wealth active-return math, a "
      f"hard fail on missing returns in held names, and a minimum-eligible-"
      f"names sanity gate) are "
      f"frozen as of {PARAMETER_FREEZE_DATE}. Every backtest "
      f"metric above — including the 'robustness check' section — was computed "
      f"with knowledge of data up to and including this freeze date, so none of "
      f"it is genuine prospective OOS. Only results generated by running this "
      f"exact configuration on dates AFTER {PARAMETER_FREEZE_DATE}, with no "
      f"further parameter changes, constitute true out-of-sample evidence.")

print(f"\n{'─'*65}")
print("  TARGET PORTFOLIO WEIGHTS (for next session)")
print(f"{'─'*65}")
latest_w     = latest_target_w.sort_values(ascending=False)
today_regime = regime_series.iloc[-1].upper()
today_macro  = int(macro_score.iloc[-1])
for sym, w in latest_w[latest_w.abs() > 0.001].items():
    print(f"  {sym:<12} {w:>7.1%}  sector={meta[sym]['sector']}")
print(f"  Regime: {today_regime}  |  Macro: {today_macro}/3")

print(f"\n{'─'*65}")
print("POSITION RECONCILIATION")
print(f"{'─'*65}")

_held_syms   = set(PORTFOLIO_UNIVERSE.keys())
_target_w    = latest_w[latest_w.abs() > 0.001]
_target_syms = set(_target_w.index)

for sym in sorted(_held_syms | _target_syms):
    held_w = PORTFOLIO_UNIVERSE.get(sym, {}).get("weight")
    tgt_w  = float(_target_w.get(sym, 0.0))
    in_held, in_target = sym in _held_syms, sym in _target_syms

    if in_held and in_target:
        if held_w is None:
            line = f"HOLD target {tgt_w:>6.1%}"
        else:
            delta = tgt_w - held_w
            verb  = "TRIM "if delta < -0.005 else "ADD   " if delta > 0.005 else "HOLD"
            line  = f"{verb} held {held_w:>6.1%} → target {tgt_w:>6.1%}  (Δ{delta:+.1%})"
    elif in_target:
        line = f"BUY new position, target {tgt_w:>6.1%}"
    else:
        line = "SELL close position — no longer in model target"
    print(f"  {sym:<8} {line}")

# 9. PLOTLY DASHBOARD
print("\n[8/8] Rendering dashboard ...")

w_all  = weights.loc[port_ret.index]
colors = ["#00d4aa","#ff6b6b","#ffd700","#7b68ee","#ff8c00","#00bfff",
          "#ff69b4","#32cd32","#dda0dd","#f0e68c"]

fig = make_subplots(
    rows=9, cols=1,
    shared_xaxes=False,
    vertical_spacing=0.025,
    row_heights=[0.16, 0.09, 0.09, 0.09, 0.08, 0.09, 0.07, 0.09, 0.24],
    specs=[[{"type": "xy"}]] * 8 + [[{"type": "table"}]],
    subplot_titles=(
        "Equity — NYSE Portfolio vs S&P 500",
        "Cumulative Active Return vs S&P 500",
        "Drawdowns",
        "HMM Composite Regime  (0=Low · 1=Mid · 2=High · 3=Stress)",
        "Macro Score (0–3)",
        "Portfolio Weights Over Time",
        "Zero-Exposure Days — Circuit-Breaker vs No Qualifying Names",
        "Time-Sliced Test Sharpe Distribution",
        "Proposed Trades — Target Rebalance",
    )
)

fig.add_trace(go.Scatter(x=equity.index, y=equity.values,
    name="NYSE Portfolio", line=dict(color="#00d4aa", width=2.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=bench_equity.index, y=bench_equity.values,
    name="S&P 500", line=dict(color="#ff6b6b", width=2, dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=equity.index, y=equity.cummax().values,
    name="HWM", line=dict(color="grey", width=1, dash="dot"),
    showlegend=False), row=1, col=1)

fig.add_trace(go.Scatter(x=cum_active.index, y=(cum_active.values - 1) * 100,
    name="Active Ret %", fill="tozeroy",
    line=dict(color="#ffd700", width=2),
    fillcolor="rgba(255,215,0,0.12)"), row=2, col=1)
fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=0.8, row=2, col=1)

fig.add_trace(go.Scatter(x=dd_port.index, y=dd_port.values * 100,
    name="Port DD", fill="tozeroy",
    line=dict(color="#ff4444", width=1.5),
    fillcolor="rgba(255,68,68,0.25)"), row=3, col=1)
fig.add_trace(go.Scatter(x=dd_bench.index, y=dd_bench.values * 100,
    name="Bench DD", line=dict(color="#ff6b6b", width=1, dash="dot")), row=3, col=1)
fig.add_hline(y=-15, line_dash="dash", line_color="red", line_width=1,
              annotation_text="-15% limit", annotation_position="bottom left",
              row=3, col=1)

regime_num = regime_series.reindex(port_ret.index).map(
    {"risk_on": 0, "neutral": 1, "risk_off": 2, "stress": 3}).fillna(1)
fig.add_trace(go.Scatter(x=regime_num.index, y=regime_num.values,
    name="HMM Regime", fill="tozeroy",
    line=dict(color="#7b68ee", width=1.5),
    fillcolor="rgba(123,104,238,0.20)"), row=4, col=1)

macro_plot = macro_score.reindex(port_ret.index).fillna(0)
fig.add_trace(go.Scatter(x=macro_plot.index, y=macro_plot.values,
    name="Macro Score", line=dict(color="#ffab00", width=1.5)), row=5, col=1)

avg_w = w_all.abs().mean().sort_values(ascending=False)
for i, sym in enumerate(avg_w.head(8).index):
    fig.add_trace(go.Scatter(
        x=w_all.index, y=w_all[sym].values * 100,
        name=sym, mode="lines",
        line=dict(width=0.5, color=colors[i % len(colors)]),
        stackgroup="weights",
        hovertemplate=f"{sym}: %{{y:.1f}}%<extra></extra>"), row=6, col=1)

total_w        = w_all.sum(axis=1)
cb_mask        = circuit_breaker.reindex(w_all.index).fillna(False)
risk_freeze    = cb_mask.astype(float) * 100
no_qualifiers  = ((total_w < 0.01) & (~cb_mask)).astype(float) * 100
fig.add_trace(go.Scatter(x=w_all.index, y=risk_freeze.values,
    name="Circuit-Breaker Freeze", fill="tozeroy",
    line=dict(color="#ff4444", width=1),
    fillcolor="rgba(255,68,68,0.35)"), row=7, col=1)
fig.add_trace(go.Scatter(x=w_all.index, y=no_qualifiers.values,
    name="No Qualifying Names (cash)", fill="tozeroy",
    line=dict(color="#ffab00", width=1),
    fillcolor="rgba(255,171,0,0.30)"), row=7, col=1)

fig.add_trace(go.Histogram(x=oos_arr, nbinsx=max(5, len(oos_arr) // 2),
    name="OOS Sharpe", marker_color="#ffd700", opacity=0.8), row=8, col=1)
fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5, row=8, col=1)
fig.add_vline(x=oos_arr.mean(), line_dash="dot", line_color="#00d4aa",
              line_width=1.5, row=8, col=1,
              annotation_text=f"Mean={oos_arr.mean():.2f}",
              annotation_position="top right")

trade_syms = latest_w[latest_w.abs() > 0.001].sort_values(ascending=False)
t_syms   = trade_syms.index.tolist()
t_wts    = [f"{w:.1%}" for w in trade_syms.values]
t_acts   = ["BUY" if w > 0 else "SELL" for w in trade_syms.values]
t_prices = [f"${float(close[s].iloc[-1]):,.2f}" if s in close.columns else "N/A"
            for s in t_syms]
t_costs  = [f"{meta[s]['cost_bps'] * 10000:.0f} bps"
            for s in t_syms]  # one-way — a full round trip (entry + eventual exit) is ~2x this
t_secs   = [meta[s]["sector"] for s in t_syms]
act_cols = ["rgba(0,212,170,0.25)" if a == "BUY" else "rgba(255,107,107,0.25)"
            for a in t_acts]
fnt_cols = ["#00d4aa" if a == "BUY" else "#ff6b6b" for a in t_acts]

fig.add_trace(go.Table(
    header=dict(
        values=["<b>Symbol</b>","<b>Action</b>","<b>Weight</b>",
                "<b>Last Price</b>","<b>Sector</b>","<b>Cost (1-way)</b>"],
        fill_color="#1a1a2e", font=dict(color="#aaaaaa", size=12),
        align="center", line_color="#2a2a4a"
    ),
    cells=dict(
        values=[t_syms, t_acts, t_wts, t_prices, t_secs, t_costs],
        fill_color=[
            ["#16213e"]*len(t_syms), act_cols,
            ["#16213e"]*len(t_syms), ["#16213e"]*len(t_syms),
            ["#16213e"]*len(t_syms), ["#16213e"]*len(t_syms),
        ],
        font=dict(color=[
            ["#ffffff"]*len(t_syms), fnt_cols,
            ["#ffffff"]*len(t_syms), ["#ffffff"]*len(t_syms),
            ["#aaaaaa"]*len(t_syms), ["#aaaaaa"]*len(t_syms),
        ], size=11),
        align="center", line_color="#2a2a4a", height=26
    )
), row=9, col=1)

stats = (
    f"Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f} | Calmar: {calmar:.2f} |"
    f"IR: {info_ratio:.2f} | MaxDD: {max_dd:.1%} | CVaR95: {port_cvar_95:.4f} |"
    f"1M active ret: {alpha_1m:.2%} | β_mkt: {beta_market:.2f} |"
    f"R²: {r_squared:.2f} | NegSliceFrac: {neg_slice_frac:.2f} | Macro: {today_macro}/3 |"
    f"AvgWeekTO: {avg_weekly_to:.3f}"
)
fig.update_layout(
    height=2400, template="plotly_dark",
    title=dict(
        text="<b>NYSE Model — Momentum/RS · HMM Exposure Control · Inverse-Vol Sizing</b>",
        font=dict(size=19)
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.005,
                xanchor="center", x=0.5, font=dict(size=9)),
    hovermode="x unified", margin=dict(t=110, b=40, l=70, r=35)
)
fig.add_annotation(
    text=stats, xref="paper", yref="paper",
    x=0.5, y=0.998, showarrow=False,
    font=dict(size=10, color="#aaaaaa"),
    bgcolor="rgba(0,0,0,0.55)", borderpad=5
)
for r, lbl in [(1,"Equity"),(2,"Active Ret %"),(3,"DD %"),(4,"Regime"),
               (5,"Macro"),(6,"Weight %"),(7,"Frozen %"),(8,"Count")]:
    fig.update_yaxes(title_text=lbl, row=r, col=1)
fig.update_xaxes(title_text="Date", row=6, col=1)
fig.update_xaxes(title_text="OOS Sharpe", row=8, col=1)
fig.show()

# 9b. LIVE PORTFOLIO DASHBOARD
print("\n[8/8b] Live Portfolio Dashboard")

invested_pct = float(latest_w.abs().sum())
cash_pct     = max(0.0, 1.0 - invested_pct)

fig2 = make_subplots(
    rows=4, cols=1,
    shared_xaxes=False,
    vertical_spacing=0.05,
    row_heights=[0.22, 0.21, 0.21, 0.36],
    specs=[[{"type": "table"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
    subplot_titles=(
        "",
        "Gross Exposure vs Cash Over Time",
        "Current Sector Allocation",
        "Proposed Trades — Signal Breakdown",
    )
)

header_metrics = ["NAV", "Invested %", "Cash %", "Regime", "Macro", "Vol (ann)",
                   "Beta (mkt)", "CVaR (95%)", "Max DD"]
header_values  = [
    f"${CFG['backtest_nav']:,.0f}  (backtest reference NAV)",
    f"{invested_pct:.1%}", f"{cash_pct:.1%}", today_regime, f"{today_macro}/3",
    f"{ann_vol:.1%}", f"{beta_market:.2f}", f"{port_cvar_95:.4f}", f"{max_dd:.1%}",
]
fig2.add_trace(go.Table(
    header=dict(values=["<b>LIVE PORTFOLIO</b>", ""],
                fill_color="#1a1a2e", font=dict(color="#ffd700", size=13),
                align="left", line_color="#2a2a4a"),
    cells=dict(values=[header_metrics, header_values],
               fill_color="#16213e",
               font=dict(color=["#aaaaaa", "#ffffff"], size=12),
               align="left", line_color="#2a2a4a", height=24)
), row=1, col=1)

total_w_ts = w_all.sum(axis=1).clip(lower=0, upper=1) * 100
cash_ts    = 100 - total_w_ts
fig2.add_trace(go.Scatter(x=w_all.index, y=total_w_ts.values,
    name="Invested %", fill="tozeroy", stackgroup="exposure",
    line=dict(color="#00d4aa", width=1)), row=2, col=1)
fig2.add_trace(go.Scatter(x=w_all.index, y=cash_ts.values,
    name="Cash %", fill="tonexty", stackgroup="exposure",
    line=dict(color="#555577", width=1)), row=2, col=1)

_sector_w = {}
for sym, w in latest_w[latest_w.abs() > 0.001].items():
    sec = meta[sym]["sector"]
    _sector_w[sec] = _sector_w.get(sec, 0.0) + float(w)
_sec_items = sorted(_sector_w.items(), key=lambda kv: kv[1], reverse=True)
fig2.add_trace(go.Bar(
    x=[v * 100 for _, v in _sec_items], y=[k for k, _ in _sec_items],
    orientation="h", marker_color="#7b68ee",
    name="Sector Weight"), row=3, col=1)
_sector_bar_trace = fig2.data[-1]
_cap_x = CFG["sector_cap"] * 100
fig2.add_shape(type="line", xref=_sector_bar_trace.xaxis, yref=f"{_sector_bar_trace.yaxis} domain",
               x0=_cap_x, x1=_cap_x, y0=0, y1=1,
               line=dict(color="red", dash="dash", width=1))
fig2.add_annotation(xref=_sector_bar_trace.xaxis, yref=f"{_sector_bar_trace.yaxis} domain",
                     x=_cap_x, y=1.04, text=f"{CFG['sector_cap']:.0%} cap",
                     showarrow=False, font=dict(size=10, color="red"))

_m1_row    = mom_1m.iloc[-1]
_m3_row    = mom_3m.iloc[-1]
_m6_row    = mom_6m.iloc[-1]
_m12_row   = mom_12m.iloc[-1]
_rs_row    = rel_strength.iloc[-1]
_vol_row   = vol_ann.iloc[-1]
_score_row = alpha_final.iloc[-1]

def _fmt_pct(row, sym):
    v = row.get(sym, np.nan)
    return f"{v:.1%}" if np.isfinite(v) else "N/A"

def _fmt_score(sym):
    v = _score_row.get(sym, np.nan)
    return f"{v:.2f}" if np.isfinite(v) else "N/A"

def _fmt_atr_stop(sym):
    atr_series = compute_atr(sym)
    atr_val = float(atr_series.iloc[-1]) if atr_series is not None and not atr_series.empty else np.nan
    if not np.isfinite(atr_val) or sym not in close.columns:
        return "N/A"
    last_close = float(close[sym].iloc[-1])
    return f"${last_close - CFG['atr_stop_mult'] * atr_val:,.2f}"

t_m1    = [_fmt_pct(_m1_row, s) for s in t_syms]
t_m3    = [_fmt_pct(_m3_row, s) for s in t_syms]
t_m6    = [_fmt_pct(_m6_row, s) for s in t_syms]
t_m12   = [_fmt_pct(_m12_row, s) for s in t_syms]
t_rs6m  = [_fmt_pct(_rs_row, s) for s in t_syms]
t_vol   = [_fmt_pct(_vol_row, s) for s in t_syms]
t_score = [_fmt_score(s) for s in t_syms]
t_stop  = [_fmt_atr_stop(s) for s in t_syms]

fig2.add_trace(go.Table(
    header=dict(
        values=["<b>Symbol</b>","<b>Action</b>","<b>Weight</b>","<b>Price</b>",
                "<b>M1</b>","<b>M3</b>","<b>M6</b>","<b>M12</b>","<b>RS6M</b>",
                "<b>Vol(ann)</b>","<b>Score</b>","<b>ATR Stop</b>","<b>Sector</b>"],
        fill_color="#1a1a2e", font=dict(color="#aaaaaa", size=11),
        align="center", line_color="#2a2a4a"
    ),
    cells=dict(
        values=[t_syms, t_acts, t_wts, t_prices, t_m1, t_m3, t_m6, t_m12,
                t_rs6m, t_vol, t_score, t_stop, t_secs],
        fill_color=[["#16213e"]*len(t_syms), act_cols] + [["#16213e"]*len(t_syms)]*11,
        font=dict(color=[["#ffffff"]*len(t_syms), fnt_cols] + [["#dddddd"]*len(t_syms)]*11, size=10),
        align="center", line_color="#2a2a4a", height=24
    )
), row=4, col=1)

fig2.update_layout(
    height=1550, template="plotly_dark",
    title=dict(text="<b>NYSE Momentum Model — Live Portfolio Dashboard</b>", font=dict(size=18)),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5, font=dict(size=9)),
    hovermode="x unified", margin=dict(t=90, b=40, l=90, r=35)
)
fig2.update_yaxes(title_text="Exposure %", row=2, col=1)
fig2.update_xaxes(title_text="Sector Weight %", row=3, col=1)
fig2.show()

# 10. CURRENT TARGET PORTFOLIO
print("\n" + "=" * 70)
print("  CURRENT TARGET PORTFOLIO")
print("=" * 70)

target_w_dict = {s: w for s, w in latest_target_w.items() if abs(w) > 0.001}

print("\n  Fundamental sanity check on live targets (veto layer, not a signal):")
_vetoed_w = {}
for sym in list(target_w_dict.keys()):
    verdict, reason = fundamental_veto(sym)
    if verdict == "exclude":
        print(f"    EXCLUDE {sym:<8} {reason}")
        _vetoed_w[sym] = target_w_dict.pop(sym)
    elif verdict == "flag":
        print(f"    FLAG    {sym:<8} {reason}")
    else:
        print(f"    OK      {sym:<8} {reason}")

if _vetoed_w:
    freed_w    = sum(_vetoed_w.values())
    surviving  = sum(target_w_dict.values())
    print(f"    Vetoed {list(_vetoed_w.keys())}, redistributing {freed_w:.1%} "
          f"of freed weight into {len(target_w_dict)} surviving name(s).")
    if surviving > 0:
        scale = (surviving + freed_w) / surviving
        target_w_dict = {s: w * scale for s, w in target_w_dict.items()}
        redistributed = cap_and_redistribute(pd.Series(target_w_dict), CFG["max_weight"])
        redistributed = enforce_sector_cap(redistributed, list(target_w_dict.keys()), CFG["sector_cap"])
        target_w_dict = redistributed.to_dict()
    else:
        print("    All target names vetoed — freed weight held as cash, no redistribution.")

gross_target = sum(abs(w) for w in target_w_dict.values())
if gross_target > 1.02:
    raise RuntimeError(f"Gross target weight {gross_target:.2%} exceeds 100% — "
                       f"this indicates a sizing bug, not a real target.")

print(f"\n  Target ({len(target_w_dict)} positions, gross {gross_target:.1%}) | "
      f"Regime: {today_regime} | Macro: {today_macro}/3")
for sym, w in sorted(target_w_dict.items(), key=lambda x: -x[1]):
    print(f"  {sym:<12} {w:>7.1%}  {meta[sym]['sector']}")

_closing_syms = sorted(_held_syms - set(target_w_dict.keys()))
if _closing_syms:
    print(f"\n  Closing {len(_closing_syms)} position(s) no longer in target "
          f"(vs. PORTFOLIO_UNIVERSE — keep that dict in sync with your real holdings):")
    for sym in _closing_syms:
        print(f"  {sym:<12} SELL close position — no longer in model target")

# 11. RISK ALERT SUMMARY
print(f"\n{'='*65}")
print("RISK ALERT SUMMARY")
print(f"{'='*65}")

checks = [
    (max_dd >= -0.15,       f"Max DD {max_dd:.2%}",              "target > -15%"),
    (sharpe >= 1.0,         f"Sharpe {sharpe:.2f}",              "target ≥ 1.0"),
    (sortino >= 1.5,        f"Sortino {sortino:.2f}",            "target ≥ 1.5"),
    (calmar >= 1.0,         f"Calmar {calmar:.2f}",              "target ≥ 1.0"),
    (neg_slice_frac < 0.30, f"Neg-slice fraction {neg_slice_frac:.3f}", "target < 0.30"),
    (beta_market <= 1.0,    f"Market beta {beta_market:.2f}",    "target ≤ 1.00"),
    (r_squared < 0.80,      f"R² {r_squared:.2f}",               "target < 0.80"),
    (port_cvar_95 < 0.035,  f"CVaR(95%) {port_cvar_95:.4f}",     "target < 0.035"),
    (avg_weekly_to < 0.20,  f"Avg Weekly TO {avg_weekly_to:.3f}","target < 0.20"),
    (total_cost < 0.15,     f"Total Costs {total_cost:.2%}",     "target < 15%"),
    (pd.isna(max_pairwise_corr) or max_pairwise_corr < 0.80,
        f"Max pairwise corr {max_pairwise_corr:.2f}",            "target < 0.80"),
    (pd.isna(max_var_contrib) or max_var_contrib < 0.50,
        f"Max var. contribution {max_var_contrib:.2f}",          "target < 0.50"),
    (pd.isna(pc1_explained) or pc1_explained < 0.70,
        f"PC1 variance explained {pc1_explained:.2f}",           "target < 0.70"),
]
for ok, metric, target in checks:
    icon = "✓  PASS" if ok else "⚠  WATCH"
    print(f"  {icon}   {metric:<34} {target}")

print("\n[9/9] Refreshing asset_explorer.html from this run's diagnostics ...")
try:
    import asset_explorer
    asset_explorer.build_page()
except Exception as e:
    print(f"  Skipped — asset_explorer chart regeneration failed: {e}")

print(f"\n{'='*65}")
print("  Model run complete — v5.0")
print(f"{'='*65}\n")