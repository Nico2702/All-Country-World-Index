import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NaroIX Index Construction Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f1117; }
  [data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a2f45; }
  h1, h2, h3, h4 { color: #e8eaf6; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; background: #161b27; padding: 4px 8px; border-radius: 10px; }
  .stTabs [data-baseweb="tab"] { background: #1e2536; border-radius: 8px; color: #8892b0; font-weight: 500; padding: 6px 20px; }
  .stTabs [aria-selected="true"] { background: #2979ff !important; color: #fff !important; }
  div[data-testid="metric-container"] {
    background: #161b27; border: 1px solid #2a2f45; border-radius: 12px;
    padding: 16px 20px;
  }
  .segment-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 600; margin: 2px;
  }
  .badge-large { background: #1a3a5c; color: #64b5f6; }
  .badge-mid   { background: #1a4a2a; color: #81c784; }
  .badge-small { background: #3a2a1a; color: #ffb74d; }
  .badge-em    { background: #3a1a3a; color: #ce93d8; }
  .info-box {
    background: #161b27; border-left: 4px solid #2979ff;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0;
    color: #aab4d0; font-size: 13px;
  }
  .warning-box {
    background: #2a1f00; border-left: 4px solid #ffc107;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0;
    color: #ffe082; font-size: 13px;
  }
  div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ──────────────────────────────────────────────────────────
def format_bn(val):
    if val >= 1e12: return f"{val/1e12:.2f}T"
    if val >= 1e9:  return f"{val/1e9:.2f}B"
    if val >= 1e6:  return f"{val/1e6:.2f}M"
    return f"{val:.0f}"


def compute_variant1(df_dm, large_pct, mid_pct, small_pct):
    """Global DM threshold approach.
    Sort by Total MCap desc → cumulative sum on FF MCap → segment by ENDING cum_pct.
    """
    df = df_dm.sort_values("Total MCap Y2025", ascending=False).copy()
    total_ff = df["Free Float MCap Y2025"].sum()
    df["cum_ff"] = df["Free Float MCap Y2025"].cumsum()
    df["cum_pct"] = df["cum_ff"] / total_ff * 100  # ending cumulative %

    def assign_segment(row):
        if row["cum_pct"] <= large_pct:
            return "Large Cap"
        elif row["cum_pct"] <= mid_pct:
            return "Mid Cap"
        elif row["cum_pct"] <= small_pct:
            return "Small Cap"
        else:
            return "Micro / Excluded"

    df["Segment"] = df.apply(assign_segment, axis=1)
    return df


def compute_variant2(df_dm, large_pct, mid_pct, small_pct):
    """Per-country threshold approach.
    Sort by Total MCap desc within each country → cumulative sum on FF MCap → segment by ENDING cum_pct.
    """
    results = []
    for country, grp in df_dm.groupby("Exchange Country Name"):
        grp = grp.sort_values("Total MCap Y2025", ascending=False).copy()
        total_ff = grp["Free Float MCap Y2025"].sum()
        if total_ff == 0:
            grp["Segment"] = "Micro / Excluded"
            grp["cum_ff"] = 0
            grp["cum_pct"] = 0
            results.append(grp)
            continue
        grp["cum_ff"] = grp["Free Float MCap Y2025"].cumsum()
        grp["cum_pct"] = grp["cum_ff"] / total_ff * 100

        def assign_segment(row):
            if row["cum_pct"] <= large_pct:
                return "Large Cap"
            elif row["cum_pct"] <= mid_pct:
                return "Mid Cap"
            elif row["cum_pct"] <= small_pct:
                return "Small Cap"
            else:
                return "Micro / Excluded"

        grp["Segment"] = grp.apply(assign_segment, axis=1)
        results.append(grp)
    return pd.concat(results, ignore_index=True)


def compute_variant3(df_all, eumss_pct=0.99, eumss_ff_ratio=0.50, min_ff_pct=0.15,
                      gmsr_pct=0.85, em_gmsr_ratio=0.50,
                      auto_mult=1.15, cand_mult=0.50, buf_mult=0.33,
                      atvr_dm_min=0.20, atvr_em_min=0.15,
                      adtv_dm_3m=2e6, adtv_dm_6m=2e6, adtv_em_3m=1e6, adtv_em_6m=1e6,
                      china_if=0.20, india_if=0.75, vietnam_if=0.50, saudi_if=0.50):
    """Full NaroIX V3 methodology — Steps 3-6.
    Implements EUMSS, ATVR, Inclusion Factors, GMSR and Zone assignment.
    Steps 7 (85% coverage) and 8 (secondary) handled separately in the tab.
    Returns (df_annotated, eumss_full, eumss_ff, dm_gmsr, em_gmsr)
    """
    df = df_all.copy()

    # ── Step 3: Compute EUMSS on ALL DM stocks ───────────────────────────────
    df_dm_all = df[df["Classification"] == "DM"].sort_values("Total MCap Y2025", ascending=False).copy()
    total_ff_dm = df_dm_all["Free Float MCap Y2025"].sum()
    if total_ff_dm == 0:
        df["Zone"] = "EXCLUDE"
        return df, 0, 0, 0, 0

    df_dm_all["_cum_ff"] = df_dm_all["Free Float MCap Y2025"].cumsum()
    df_dm_all["_cum_pct"] = df_dm_all["_cum_ff"] / total_ff_dm * 100
    eumss_row = df_dm_all[df_dm_all["_cum_pct"] >= eumss_pct * 100].iloc[0]
    eumss_full = eumss_row["Total MCap Y2025"]
    eumss_ff   = eumss_full * eumss_ff_ratio

    # EUMSS filter mask
    mask_eumss = (
        (df["Total MCap Y2025"] >= eumss_full) &
        (df["Free Float MCap Y2025"] >= eumss_ff) &
        (df["Free Float Percent"] >= min_ff_pct)
    )
    df["EUMSS_Pass"] = mask_eumss
    df["EUMSS_FULL"] = eumss_full
    df["EUMSS_FF"]   = eumss_ff

    # ── Step 4: ADTV + ATVR filter ───────────────────────────────────────────
    # ADTV best fallback: 12M → 6M → 3M → 1M
    df["_adtv_best"] = df["12M ADTV Y2025"].where(df["12M ADTV Y2025"] > 0,
                       df["6M ADTV Y2025"].where(df["6M ADTV Y2025"] > 0,
                       df["3M ADTV Y2025"].where(df["3M ADTV Y2025"] > 0,
                       df["1M ADTV Y2025"])))
    df["ATVR"] = np.where(df["Total MCap Y2025"] > 0,
                          df["_adtv_best"] * 252 / df["Total MCap Y2025"], 0)

    # ADTV 3M + 6M filter
    mask_adtv_dm = (df["Classification"] == "DM") & (df["3M ADTV Y2025"] >= adtv_dm_3m) & (df["6M ADTV Y2025"] >= adtv_dm_6m)
    mask_adtv_em = (df["Classification"] == "EM") & (df["3M ADTV Y2025"] >= adtv_em_3m) & (df["6M ADTV Y2025"] >= adtv_em_6m)
    df["ADTV_Pass"] = mask_adtv_dm | mask_adtv_em

    # ATVR filter
    mask_atvr_dm = (df["Classification"] == "DM") & (df["ATVR"] >= atvr_dm_min)
    mask_atvr_em = (df["Classification"] == "EM") & (df["ATVR"] >= atvr_em_min)
    df["ATVR_Pass"] = mask_atvr_dm | mask_atvr_em

    # ── Step 5: Inclusion Factor + Adj_FF_MCap ───────────────────────────────
    # Based on Exchange Country Name (not Exchange Name)
    ecn = df["Exchange Country Name"].fillna("")
    df["Inclusion_Factor_V3"] = np.where(ecn.str.upper() == "CHINA",        china_if,
                                np.where(ecn.str.upper() == "INDIA",        india_if,
                                np.where(ecn.str.upper() == "VIETNAM",      vietnam_if,
                                np.where(ecn.str.upper() == "SAUDI ARABIA", saudi_if, 1.0))))
    df["Adj_FF_MCap"] = df["Free Float MCap Y2025"] * df["Inclusion_Factor_V3"]

    # ── Step 6: Compute GMSR on EUMSS+ATVR filtered DM stocks using Adj_FF_MCap
    mask_dm_pass = mask_eumss & df["ADTV_Pass"] & df["ATVR_Pass"] & (df["Classification"] == "DM")
    df_dm_filtered = df[mask_dm_pass].sort_values("Total MCap Y2025", ascending=False).copy()
    total_adj_dm_f = df_dm_filtered["Adj_FF_MCap"].sum()
    if total_adj_dm_f == 0:
        df["Zone"] = "EXCLUDE"
        return df, eumss_full, eumss_ff, 0, 0

    df_dm_filtered["_cum_adj"] = df_dm_filtered["Adj_FF_MCap"].cumsum()
    df_dm_filtered["_cum_adj_pct"] = df_dm_filtered["_cum_adj"] / total_adj_dm_f * 100
    gmsr_row = df_dm_filtered[df_dm_filtered["_cum_adj_pct"] >= gmsr_pct * 100].iloc[0]
    dm_gmsr = gmsr_row["Total MCap Y2025"]
    em_gmsr = dm_gmsr * em_gmsr_ratio

    df["DM_GMSR"]    = dm_gmsr
    df["EM_GMSR"]    = em_gmsr
    df["GMSR_TITEL"] = np.where(df["Classification"] == "DM", dm_gmsr, em_gmsr)

    # Zone assignment (vectorized)
    ratio = np.where(df["GMSR_TITEL"] > 0, df["Total MCap Y2025"] / df["GMSR_TITEL"], 0)
    zone = np.where(~mask_eumss | ~df["ADTV_Pass"] | ~df["ATVR_Pass"],  "EXCLUDE",
           np.where(ratio > auto_mult,                  "AUTO_INCLUDE",
           np.where(ratio >= cand_mult,                 "CANDIDATE",
           np.where(ratio >= buf_mult,                  "BUFFER",
                                                        "EXCLUDE"))))
    df["Zone"] = zone

    return df, eumss_full, eumss_ff, dm_gmsr, em_gmsr


def apply_step7_coverage(df_candidates, coverage_pct=0.85, country_col="Mapping Country"):
    """Step 7: 85% coverage per country based on raw Free Float MCap.
    Inclusion Factor is NOT used here — only for final weighting in Step 9.
    Returns subset of df that passes coverage threshold per country.
    """
    results = []
    country_col = country_col if country_col in df_candidates.columns else "Exchange Country Name"
    for country, grp in df_candidates.groupby(country_col):
        grp = grp.sort_values("Free Float MCap Y2025", ascending=False).copy()
        total_ff = grp["Free Float MCap Y2025"].sum()
        if total_ff == 0:
            continue
        grp["_cum_ff"] = grp["Free Float MCap Y2025"].cumsum()
        grp["_cum_ff_pct"] = grp["_cum_ff"] / total_ff * 100
        target = coverage_pct * 100
        # Find cutoff index — include the stock that crosses the threshold
        cutoff_idx = grp[grp["_cum_ff_pct"] >= target].index
        if len(cutoff_idx) == 0:
            results.append(grp)
        else:
            include_up_to = cutoff_idx[0]
            results.append(grp.loc[:include_up_to])
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame(columns=df_candidates.columns)


def apply_step8_secondary(df_step7, df_raw_orig, eumss_full, eumss_ff, min_ff_pct,
                           atvr_dm_min, atvr_em_min,
                           china_if=0.20, india_if=0.75, vietnam_if=0.50, saudi_if=0.50,
                           adtv_dm_3m=2e6, adtv_dm_6m=2e6, adtv_em_3m=1e6, adtv_em_6m=1e6):
    """Step 8: Add secondary share classes via Entity ID matching."""
    if "Entity ID" not in df_step7.columns or "Entity ID" not in df_raw_orig.columns:
        return df_step7

    # Entity IDs already in index
    included_entity_ids = set(df_step7["Entity ID"].dropna().unique())
    included_symbols = set(df_step7["Symbol"].dropna().unique())

    # Filter raw data for secondary listings
    df_sec = df_raw_orig[
        (df_raw_orig["Listing"].fillna("") == "Secondary") &
        (df_raw_orig["Entity ID"].isin(included_entity_ids)) &
        (~df_raw_orig["Symbol"].isin(included_symbols))
    ].copy()

    if len(df_sec) == 0:
        return df_step7

    # Apply same exclusions as Step 1
    import re
    _etf_pat = re.compile(r'ETF|SICAV|%', re.IGNORECASE)
    df_sec = df_sec[
        (df_sec["Free Float MCap Y2025"].fillna(0) > 0) &
        (df_sec["Total MCap Y2025"].fillna(0) >= eumss_full) &
        (df_sec["Free Float MCap Y2025"].fillna(0) >= eumss_ff) &
        (df_sec["Free Float Percent"].fillna(0) >= min_ff_pct) &
        (~df_sec["Name"].fillna("").str.contains(_etf_pat))
    ].copy()

    if len(df_sec) == 0:
        return df_step7

    # Merge classification from step7 via Entity ID
    cls_map = df_step7[["Entity ID","Classification"]].drop_duplicates().set_index("Entity ID")["Classification"].to_dict()
    df_sec["Classification"] = df_sec["Entity ID"].map(cls_map)
    df_sec = df_sec[df_sec["Classification"].notna()].copy()

    # ADTV filter
    df_sec_dm = df_sec[df_sec["Classification"] == "DM"]
    df_sec_em = df_sec[df_sec["Classification"] == "EM"]
    df_sec_dm = df_sec_dm[(df_sec_dm["3M ADTV Y2025"] >= adtv_dm_3m) & (df_sec_dm["6M ADTV Y2025"] >= adtv_dm_6m)]
    df_sec_em = df_sec_em[(df_sec_em["3M ADTV Y2025"] >= adtv_em_3m) & (df_sec_em["6M ADTV Y2025"] >= adtv_em_6m)]
    df_sec = pd.concat([df_sec_dm, df_sec_em], ignore_index=True)

    # ATVR filter
    df_sec["_adtv_best"] = df_sec["12M ADTV Y2025"].where(df_sec["12M ADTV Y2025"] > 0,
                           df_sec["6M ADTV Y2025"].where(df_sec["6M ADTV Y2025"] > 0,
                           df_sec["3M ADTV Y2025"].where(df_sec["3M ADTV Y2025"] > 0,
                           df_sec["1M ADTV Y2025"])))
    df_sec["ATVR"] = np.where(df_sec["Total MCap Y2025"] > 0,
                               df_sec["_adtv_best"] * 252 / df_sec["Total MCap Y2025"], 0)
    df_sec = df_sec[
        ((df_sec["Classification"] == "DM") & (df_sec["ATVR"] >= atvr_dm_min)) |
        ((df_sec["Classification"] == "EM") & (df_sec["ATVR"] >= atvr_em_min))
    ].copy()

    # Apply Inclusion Factor
    ecn = df_sec["Exchange Country Name"].fillna("")
    df_sec["Inclusion_Factor_V3"] = np.where(ecn.str.upper() == "CHINA",        china_if,
                                    np.where(ecn.str.upper() == "INDIA",        india_if,
                                    np.where(ecn.str.upper() == "VIETNAM",      vietnam_if,
                                    np.where(ecn.str.upper() == "SAUDI ARABIA", saudi_if, 1.0))))
    df_sec["Adj_FF_MCap"] = df_sec["Free Float MCap Y2025"] * df_sec["Inclusion_Factor_V3"]
    df_sec["Zone"] = "SECONDARY"
    df_sec["Listing_Type"] = "Secondary"

    return pd.concat([df_step7, df_sec], ignore_index=True)


def get_dm_cutoff_stock(df_dm_seg):
    """Return the last DM stock in the World Index (V1: global sort → simply last row included)."""
    world = df_dm_seg[df_dm_seg["Segment"].isin(["Large Cap", "Mid Cap"])]
    if len(world) == 0:
        return None
    return world.iloc[-1]


def get_dm_cutoff_stock_v2(df_dm, large_pct, mid_pct, small_pct, mode="min"):
    """For Variant 2: compute per-country cutoff stocks, then return min or max Total MCap.
    mode='min': smallest cutoff stock (lowest EM threshold → more EM stocks included)
    mode='max': largest cutoff stock (highest EM threshold → fewer EM stocks included)
    Returns the selected cutoff stock row and a DataFrame of all country cutoff stocks.
    """
    country_cutoffs = []
    for country, grp in df_dm.groupby("Exchange Country Name"):
        grp = grp.sort_values("Total MCap Y2025", ascending=False).copy()
        total_ff = grp["Free Float MCap Y2025"].sum()
        if total_ff == 0:
            continue
        grp["cum_ff"]  = grp["Free Float MCap Y2025"].cumsum()
        grp["cum_pct"] = grp["cum_ff"] / total_ff * 100
        grp["Segment"] = grp["cum_pct"].apply(
            lambda x: "Large Cap" if x <= large_pct else ("Mid Cap" if x <= mid_pct else "Other")
        )
        world_grp = grp[grp["Segment"].isin(["Large Cap", "Mid Cap"])]
        if len(world_grp) == 0:
            continue
        cutoff = world_grp.iloc[-1].copy()
        cutoff["Country"] = country
        country_cutoffs.append(cutoff)

    if not country_cutoffs:
        return None, pd.DataFrame()

    df_cutoffs = pd.DataFrame(country_cutoffs)
    if mode == "min":
        selected = df_cutoffs.loc[df_cutoffs["Total MCap Y2025"].idxmin()]
    else:
        selected = df_cutoffs.loc[df_cutoffs["Total MCap Y2025"].idxmax()]
    return selected, df_cutoffs


def filter_em_by_threshold(df_em, cutoff_total_mcap, pct):
    """Include EM stocks whose Total MCap >= pct% of the DM cutoff stock Total MCap."""
    min_mcap = cutoff_total_mcap * (pct / 100.0)
    df = df_em.copy()
    df["Segment"] = df["Total MCap Y2025"].apply(
        lambda x: "EM Included" if x >= min_mcap else "EM Excluded"
    )
    return df, min_mcap


def filter_em_per_country(df_em, pct=85):
    """Apply per-country cumulative FF MCap rule to EM (same as Variant 2 for DM).
    Sort by Total MCap desc within each EM country, cumulate FF MCap,
    include all stocks up to pct% cumulative FF MCap per country.
    """
    results = []
    for country, grp in df_em.groupby("Exchange Country Name"):
        grp = grp.sort_values("Total MCap Y2025", ascending=False).copy()
        total_ff = grp["Free Float MCap Y2025"].sum()
        if total_ff == 0:
            grp["Segment"] = "EM Excluded"
            grp["cum_ff"] = 0
            grp["cum_pct"] = 0
            results.append(grp)
            continue
        grp["cum_ff"]  = grp["Free Float MCap Y2025"].cumsum()
        grp["cum_pct"] = grp["cum_ff"] / total_ff * 100
        grp["Segment"] = grp["cum_pct"].apply(
            lambda x: "EM Included" if x <= pct else "EM Excluded"
        )
        results.append(grp)
    return pd.concat(results, ignore_index=True)


def get_exclusion_reasons(df, ff_gt0=True, min_ff_pct=0.0, max_closing_price=None,
                           min_adtv_1m=0, min_adtv_3m=0, min_adtv_6m=0, min_adtv_12m=0,
                           min_liq_ratio=0.0):
    """Return a Series of exclusion reason strings per stock. Fully vectorized."""
    n = len(df)
    # Build boolean masks per criterion
    masks = {}
    if ff_gt0:
        masks["FF MCap ≤ 0 or missing"] = df["Free Float MCap Y2025"].isna() | (df["Free Float MCap Y2025"] <= 0)
    if min_ff_pct > 0:
        masks[f"FF% < {min_ff_pct*100:.0f}%"] = df["Free Float Percent"] < min_ff_pct
    if max_closing_price is not None:
        masks[f"Price > {max_closing_price:,.0f}"] = df["Closing Price"] > max_closing_price
    if min_adtv_1m  > 0: masks[f"1M ADTV < {min_adtv_1m:,.0f}"]  = df["1M ADTV Y2025"]  < min_adtv_1m
    if min_adtv_3m  > 0: masks[f"3M ADTV < {min_adtv_3m:,.0f}"]  = df["3M ADTV Y2025"]  < min_adtv_3m
    if min_adtv_6m  > 0: masks[f"6M ADTV < {min_adtv_6m:,.0f}"]  = df["6M ADTV Y2025"]  < min_adtv_6m
    if min_adtv_12m > 0: masks[f"12M ADTV < {min_adtv_12m:,.0f}"] = df["12M ADTV Y2025"] < min_adtv_12m
    if min_liq_ratio > 0:
        liq = np.where(df["Free Float MCap Y2025"] > 0,
                       df["12M ADTV Y2025"] / df["Free Float MCap Y2025"] * 252 * 100, 0)
        masks[f"Liq. Ratio < {min_liq_ratio:.1f}%"] = liq < min_liq_ratio

    if not masks:
        return pd.Series([""] * n, index=df.index)

    # Build reason strings vectorized
    reasons = pd.Series([""] * n, index=df.index)
    for label, mask in masks.items():
        has_reason = reasons != ""
        reasons = np.where(mask,
            np.where(has_reason, reasons + " | " + label, label),
            reasons)
    return pd.Series(reasons, index=df.index)


def apply_min_filters(df, ff_gt0=True, min_ff_pct=0.0, max_closing_price=None, min_adtv_1m=0, min_adtv_3m=0, min_adtv_6m=0, min_adtv_12m=0, min_liq_ratio=0.0):
    """Filter out stocks after segment computation."""
    mask = pd.Series(True, index=df.index)
    if ff_gt0:                         mask &= df["Free Float MCap Y2025"].notna() & (df["Free Float MCap Y2025"] > 0)
    if min_ff_pct  > 0:                mask &= df["Free Float Percent"] >= min_ff_pct
    if max_closing_price is not None:  mask &= df["Closing Price"] <= max_closing_price
    if min_adtv_1m  > 0:  mask &= df["1M ADTV Y2025"]  >= min_adtv_1m
    if min_adtv_3m  > 0:  mask &= df["3M ADTV Y2025"]  >= min_adtv_3m
    if min_adtv_6m  > 0:  mask &= df["6M ADTV Y2025"]  >= min_adtv_6m
    if min_adtv_12m > 0:  mask &= df["12M ADTV Y2025"] >= min_adtv_12m
    if min_liq_ratio > 0:
        liq = np.where(df["Free Float MCap Y2025"] > 0,
                       df["12M ADTV Y2025"] / df["Free Float MCap Y2025"] * 252 * 100, 0)
        mask &= liq >= min_liq_ratio
    return df[mask].copy()


def build_full_export(df_seg, classification, post_filter_fn, filter_params):
    """Build full universe export with Segment + Status + Exclusion Reason columns."""
    df = df_seg.copy()
    df["Classification"] = classification

    # Compute exclusion reasons for ALL stocks in the segmented universe
    reasons = get_exclusion_reasons(df, **filter_params)
    df["Exclusion Reason"] = reasons

    # Status: Included if has a real segment AND passes filters
    included_mask = (
        df["Segment"].isin(["Large Cap", "Mid Cap", "Small Cap", "EM Included"]) &
        (df["Exclusion Reason"] == "")
    )
    df["Status"] = np.where(included_mask, "Included", "Excluded")
    df.loc[df["Exclusion Reason"] == "", "Exclusion Reason"] = np.where(
        df.loc[df["Exclusion Reason"] == "", "Segment"].isin(
            ["Micro / Excluded", "EM Excluded", "Other"]
        ), "Below Segment Cutoff", ""
    )
    df.loc[(df["Status"] == "Included"), "Exclusion Reason"] = ""
    return df


def add_ff_weight(df):
    """Add FF MCap Weight (%) column based on Free Float MCap."""
    df = df.copy()
    total = df["Free Float MCap Y2025"].sum()
    df["FF MCap Weight (%)"] = (df["Free Float MCap Y2025"] / total * 100) if total > 0 else 0.0
    return df


def add_adjusted_weight(df, china_factor=0.20, india_factor=0.50, vietnam_factor=0.50, saudi_factor=0.50):
    """Add Adjusted Weight (%) column applying inclusion factors per country.
    Fully vectorized — no apply().
    """
    df = df.copy()
    df["Inclusion Factor"] = 1.0
    df.loc[df["Exchange Name"].isin(["SHANGHAI", "SHENZHEN"]), "Inclusion Factor"] = china_factor
    df.loc[df["Exchange Name"].isin(["BSE INDIA", "NSE INDIA"]), "Inclusion Factor"] = india_factor
    df.loc[df["Exchange Name"].isin(["HANOI STOCK EXCHANGE", "VIETNAM"]), "Inclusion Factor"] = vietnam_factor
    df.loc[df["Exchange Name"] == "SAUDI ARABIA", "Inclusion Factor"] = saudi_factor
    df["Adjusted FF MCap"] = df["Free Float MCap Y2025"] * df["Inclusion Factor"]
    total_adj = df["Adjusted FF MCap"].sum()
    df["Adjusted Weight (%)"] = (df["Adjusted FF MCap"] / total_adj * 100) if total_adj > 0 else 0.0
    return df


def style_segment_table(df, highlight_segs):
    """Highlight key segment rows with a distinct background."""
    def row_style(row):
        if row["Segment"] in highlight_segs:
            return ["background-color: #1a2a4a; font-weight: 600;"] * len(row)
        return [""] * len(row)
    return df.style.apply(row_style, axis=1)


def segment_summary(df_seg):
    mask = df_seg["Segment"] != "Micro / Excluded"
    segments = ["Large Cap", "Mid Cap", "Small Cap"]
    rows = []
    for seg in segments:
        s = df_seg[df_seg["Segment"] == seg]
        rows.append({
            "Segment": seg,
            "# Stocks": len(s),
            "FF MCap (USD)": s["Free Float MCap Y2025"].sum(),
            "Avg FF MCap (USD)": s["Free Float MCap Y2025"].mean() if len(s) > 0 else 0,
        })
    total = df_seg[mask]
    rows.append({
        "Segment": "Total Included",
        "# Stocks": len(total),
        "FF MCap (USD)": total["Free Float MCap Y2025"].sum(),
        "Avg FF MCap (USD)": total["Free Float MCap Y2025"].mean() if len(total) > 0 else 0,
    })
    return pd.DataFrame(rows)


def to_excel_download(df, sheet_name="Index"):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buf.getvalue()


def to_excel_multi(sheets: dict):
    """Export multiple DataFrames as sheets. sheets = {sheet_name: df}"""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buf.getvalue()


SEGMENT_COLORS = {
    "Large Cap": "#2979ff",
    "Mid Cap":   "#00e676",
    "Small Cap": "#ff9100",
    "Micro / Excluded": "#37474f",
}

# ─── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_excel(file):
    try:
        df = pd.read_excel(file, sheet_name="Tab1")
    except Exception:
        df = pd.read_excel(file)
    for col in ["Free Float MCap Y2025","Total MCap Y2025","Free Float Percent",
                "1M ADTV Y2025","3M ADTV Y2025","6M ADTV Y2025","12M ADTV Y2025",
                "Closing Price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    # Ensure new country columns exist (backward compat with older files)
    for col in ["Country of Incorp","Country of Risk","Country HQ"]:
        if col not in df.columns:
            df[col] = df.get("Exchange Country Name", "")
    # Rename Industry typo if present
    if "Inudstry" in df.columns and "Industry" not in df.columns:
        df.rename(columns={"Inudstry": "Industry"}, inplace=True)
    return df


@st.cache_data
def load_classification(path="Country_Classification.xlsx"):
    try:
        df = pd.read_excel(path, usecols=["Exchange Country Name", "Classification"])
        return df.set_index("Exchange Country Name")["Classification"].to_dict()
    except Exception as e:
        st.error(f"Country_Classification.xlsx konnte nicht geladen werden: {e}")
        return {}


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Data Source")
    uploaded = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    st.markdown("---")
    st.markdown("### ⚙️ Index Parameters")

    listing_filter = st.radio("Listing Type", ["Primary only", "All (Primary + Secondary)"], index=1)
    thailand_sec_type = st.radio("Thailand:", ["SHARE", "NVDR"], index=1,
        horizontal=True, key="thailand_sec_type",
        help="Für Thailand (Exchange Name = THAILAND): welcher Sec Type wird berücksichtigt? Der andere wird excludiert.")
    with st.expander("Exclusions", expanded=False):
        exclude_hk_cny = st.checkbox("HK (CNY)", value=True,
            help="Exchange Ticker enthält HKG & Trading Currency = CNY")
        exclude_country_risk_na = st.checkbox("Country of Risk = @", value=True,
            help="Country of Risk ist nicht zugewiesen (@)")
        exclude_naics_funds = st.checkbox("NAICS = Investment Funds", value=True,
            help="NAICS = Open-End Investment Funds")
        exclude_euro_mtf = st.checkbox("Exchange = Euro MTF / @", value=True,
            help="Exchange Name ist 'Euro MTF' oder '@'")
        exclude_etf_sicav = st.checkbox("Name: ETF / SICAV / %", value=True,
            help="Name enthält eigenständiges Wort 'ETF', 'SICAV' oder '%'")


    st.markdown("**DM Percentile Thresholds**")
    _la, _lb = st.columns([3, 4])
    with _la: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>Large Cap (%)</div>", unsafe_allow_html=True)
    with _lb:
        _large_raw = st.text_input("Large Cap", value="70", key="large_thr_input", label_visibility="collapsed")
    _ma, _mb = st.columns([3, 4])
    with _ma: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>Mid Cap (%)</div>", unsafe_allow_html=True)
    with _mb:
        _mid_raw = st.text_input("Mid Cap", value="85", key="mid_thr_input", label_visibility="collapsed")
    _sa, _sb = st.columns([3, 4])
    with _sa: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>Small Cap (%)</div>", unsafe_allow_html=True)
    with _sb:
        _small_raw = st.text_input("Small Cap", value="99", key="small_thr_input", label_visibility="collapsed")
    try:    large_thr = int(_large_raw)
    except: large_thr = 70
    try:    mid_thr   = int(_mid_raw)
    except: mid_thr   = 85
    try:    small_thr = int(_small_raw)
    except: small_thr = 99

    st.markdown("**EM Parameters (ACWI Threshold)**")
    st.caption("EM-Aktien müssen mindestens X% des Total MCap des letzten DM-Grenzstocks (85%-Cutoff) erreichen.")
    _ema, _emb = st.columns([3, 4])
    with _ema: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>EM Min MCap (%)</div>", unsafe_allow_html=True)
    with _emb:
        _em_thr_raw = st.text_input("EM Min MCap", value="33.3", key="em_threshold_input", label_visibility="collapsed")
    try:    em_threshold_pct = float(_em_thr_raw)
    except: em_threshold_pct = 33.3


    st.markdown("---")
    st.markdown("### 📊 Variant 3 — Dynamic (GMSR/EUMSS)")
    with st.expander("Parameter", expanded=False):
        st.markdown("**EUMSS**")
        _v3a, _v3b = st.columns([3, 4])
        with _v3a: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>Kalibrierung (%)</div>", unsafe_allow_html=True)
        with _v3b: _v3_eumss_pct = st.text_input("EUMSS Kalibrierung", value="99", key="v3_eumss_pct", label_visibility="collapsed")
        _v3c, _v3d = st.columns([3, 4])
        with _v3c: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>FF Ratio (%)</div>", unsafe_allow_html=True)
        with _v3d: _v3_eumss_ff = st.text_input("EUMSS FF Ratio", value="50", key="v3_eumss_ff", label_visibility="collapsed")
        _v3e, _v3f = st.columns([3, 4])
        with _v3e: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>Min. FF% (%)</div>", unsafe_allow_html=True)
        with _v3f: _v3_min_ff_pct = st.text_input("Min FF Pct", value="15", key="v3_min_ff_pct", label_visibility="collapsed")
        st.markdown("**GMSR**")
        _v3g, _v3h = st.columns([3, 4])
        with _v3g: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>Kalibrierung (%)</div>", unsafe_allow_html=True)
        with _v3h: _v3_gmsr_pct = st.text_input("GMSR Kalibrierung", value="85", key="v3_gmsr_pct", label_visibility="collapsed")
        _v3i, _v3j = st.columns([3, 4])
        with _v3i: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>EM GMSR Ratio (%)</div>", unsafe_allow_html=True)
        with _v3j: _v3_em_gmsr_ratio = st.text_input("EM GMSR Ratio", value="50", key="v3_em_gmsr_ratio", label_visibility="collapsed")
        st.markdown("**Zonen-Multiplikatoren**")
        _v3k, _v3l = st.columns([3, 4])
        with _v3k: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>AUTO_INCLUDE</div>", unsafe_allow_html=True)
        with _v3l: _v3_auto = st.text_input("AUTO_INCLUDE", value="1.15", key="v3_auto", label_visibility="collapsed")
        _v3m, _v3n = st.columns([3, 4])
        with _v3m: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>CANDIDATE</div>", unsafe_allow_html=True)
        with _v3n: _v3_cand = st.text_input("CANDIDATE", value="0.50", key="v3_cand", label_visibility="collapsed")
        _v3o, _v3p = st.columns([3, 4])
        with _v3o: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>BUFFER</div>", unsafe_allow_html=True)
        with _v3p: _v3_buf = st.text_input("BUFFER", value="0.33", key="v3_buf", label_visibility="collapsed")
        use_gmsr_zones = st.checkbox("GMSR Zonen-Filter aktiv", value=False, key="v3_use_gmsr_zones",
            help="Wenn aktiv: GMSR-Zonen (AUTO_INCLUDE/CANDIDATE/BUFFER/EXCLUDE) werden angewendet → weniger Stocks (~953). Wenn inaktiv: alle EUMSS+Liquidität qualifizierten Stocks direkt zur 85% Coverage → mehr Stocks (~2,400).")
        include_buffer = st.checkbox("BUFFER einschließen (Erstberechnung)", value=False, key="v3_include_buffer",
            help="Bei Erstberechnung ohne historischen State: BUFFER-Stocks trotzdem einschließen.",
            disabled=not use_gmsr_zones)
        st.markdown("**ADTV (Schritt 4) — immer aktiv**")
        _v3u, _v3v = st.columns([3, 4])
        with _v3u: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>DM 3M & 6M (USD)</div>", unsafe_allow_html=True)
        with _v3v: _v3_adtv_dm = st.text_input("DM ADTV", value="1500000", key="v3_adtv_dm_v2", label_visibility="collapsed")
        _v3w, _v3x = st.columns([3, 4])
        with _v3w: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>EM 3M & 6M (USD)</div>", unsafe_allow_html=True)
        with _v3x: _v3_adtv_em = st.text_input("EM ADTV", value="750000", key="v3_adtv_em_v2", label_visibility="collapsed")
        st.markdown("**ATVR (Schritt 4)**")
        _v3q, _v3r = st.columns([3, 4])
        with _v3q: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>DM Min. (%)</div>", unsafe_allow_html=True)
        with _v3r: _v3_atvr_dm = st.text_input("DM ATVR", value="0", key="v3_atvr_dm_v2", label_visibility="collapsed")
        _v3s, _v3t = st.columns([3, 4])
        with _v3s: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>EM Min. (%)</div>", unsafe_allow_html=True)
        with _v3t: _v3_atvr_em = st.text_input("EM ATVR", value="0", key="v3_atvr_em_v2", label_visibility="collapsed")
        st.markdown("**Schritt 8**")
        include_secondary = st.checkbox("Secondary Share Classes einschließen", value=True, key="v3_secondary",
            help="Schritt 8: Secondary Listings mit gleicher Entity ID ergänzen.")

    try:    v3_eumss_pct      = float(_v3_eumss_pct) / 100
    except: v3_eumss_pct      = 0.99
    try:    v3_eumss_ff_ratio = float(_v3_eumss_ff) / 100
    except: v3_eumss_ff_ratio = 0.50
    try:    v3_min_ff_pct     = float(_v3_min_ff_pct) / 100
    except: v3_min_ff_pct     = 0.15
    try:    v3_gmsr_pct       = float(_v3_gmsr_pct) / 100
    except: v3_gmsr_pct       = 0.85
    try:    v3_em_gmsr_ratio  = float(_v3_em_gmsr_ratio) / 100
    except: v3_em_gmsr_ratio  = 0.50
    try:    v3_auto_mult      = float(_v3_auto)
    except: v3_auto_mult      = 1.15
    try:    v3_cand_mult      = float(_v3_cand)
    except: v3_cand_mult      = 0.50
    try:    v3_buf_mult       = float(_v3_buf)
    except: v3_buf_mult       = 0.33
    try:    v3_atvr_dm_min    = float(_v3_atvr_dm) / 100
    except: v3_atvr_dm_min    = 0.20
    try:    v3_atvr_em_min    = float(_v3_atvr_em) / 100
    except: v3_atvr_em_min    = 0.15
    try:    v3_adtv_dm        = float(_v3_adtv_dm)
    except: v3_adtv_dm        = 2_000_000
    try:    v3_adtv_em        = float(_v3_adtv_em)
    except: v3_adtv_em        = 1_000_000

    st.markdown("---")
    st.markdown("### 🔍 Filter")
    filter_timing = st.radio(
        "Filter Zeitpunkt:",
        ["Post (nach Segment-Berechnung)", "Pre (vor Segment-Berechnung)"],
        index=0, horizontal=True,
        help="Post: Filter nach der 85%-Cutoff Berechnung. Pre: Filter vor der Berechnung — beeinflusst die Segment-Zuordnung."
    )

    # FF MCap > 0 is always applied as a fixed universe filter (not optional)
    dm_ff_gt0 = True
    em_ff_gt0 = True

    _ffa, _ffb = st.columns([3, 4])
    with _ffa:
        use_ff_pct = st.checkbox("FF% ≥", value=True, key="use_ff_pct",
            help="Schließt Aktien aus deren Free Float Percent unter dem Threshold liegt.")
    with _ffb:
        ff_pct_thr = st.text_input("FF% Threshold", value="10", key="ff_pct_thr",
            label_visibility="collapsed", disabled=not use_ff_pct)
    try:    min_ff_pct = float(ff_pct_thr) / 100 if use_ff_pct else 0.0
    except: min_ff_pct = 0.0

    _cpa, _cpb = st.columns([3, 4])
    with _cpa:
        use_max_price = st.checkbox("Price ≤", value=True, key="use_max_price",
            help="Schließt Aktien aus deren Closing Price über dem Threshold liegt.")
    with _cpb:
        max_price_thr = st.text_input("Max Price", value="15,000", key="max_price_thr",
            label_visibility="collapsed", disabled=not use_max_price)
    try:    max_closing_price = float(max_price_thr.replace(",","")) if use_max_price else None
    except: max_closing_price = None

    def parse_adtv(val):
        try: return max(0, int(str(val).replace(",","").replace(".","").strip()))
        except: return 0

    with st.expander("ADTV (Threshold)", expanded=False):
        st.markdown("**Developed Markets (DM)**")
        _dma, _dmb = st.columns([3, 4])
        with _dma: dm_use_1m  = st.checkbox("1M",  value=False, key="dm_use_1m")
        with _dmb: dm_min_adtv_1m  = parse_adtv(st.text_input("1M",  value="2,000,000", key="dm_1m_v2", label_visibility="collapsed", disabled=not dm_use_1m))
        _dme, _dmf = st.columns([3, 4])
        with _dme: dm_use_3m  = st.checkbox("3M",  value=True, key="dm_use_3m")
        with _dmf: dm_min_adtv_3m  = parse_adtv(st.text_input("3M",  value="2,000,000", key="dm_3m_v2", label_visibility="collapsed", disabled=not dm_use_3m))
        _dma2, _dmb2 = st.columns([3, 4])
        with _dma2: dm_use_6m  = st.checkbox("6M",  value=True, key="dm_use_6m")
        with _dmb2: dm_min_adtv_6m  = parse_adtv(st.text_input("6M", value="2,000,000", key="dm_6m_v2", label_visibility="collapsed",  disabled=not dm_use_6m))
        _dmc, _dmd = st.columns([3, 4])
        with _dmc: dm_use_12m = st.checkbox("12M", value=False, key="dm_use_12m")
        with _dmd: dm_min_adtv_12m = parse_adtv(st.text_input("12M", value="2,000,000", key="dm_12m_v2", label_visibility="collapsed", disabled=not dm_use_12m))
        if not dm_use_1m:  dm_min_adtv_1m  = 0
        if not dm_use_3m:  dm_min_adtv_3m  = 0
        if not dm_use_6m:  dm_min_adtv_6m  = 0
        if not dm_use_12m: dm_min_adtv_12m = 0

        st.markdown("**Emerging Markets (EM)**")
        _ema0, _emb0 = st.columns([3, 4])
        with _ema0: em_use_1m  = st.checkbox("1M",  value=False, key="em_use_1m")
        with _emb0: em_min_adtv_1m  = parse_adtv(st.text_input("1M",  value="1,000,000", key="em_1m_v2", label_visibility="collapsed", disabled=not em_use_1m))
        _eme, _emf = st.columns([3, 4])
        with _eme: em_use_3m  = st.checkbox("3M",  value=True, key="em_use_3m")
        with _emf: em_min_adtv_3m  = parse_adtv(st.text_input("3M",  value="1,000,000", key="em_3m_v2", label_visibility="collapsed", disabled=not em_use_3m))
        _ema, _emb = st.columns([3, 4])
        with _ema: em_use_6m  = st.checkbox("6M",  value=True, key="em_use_6m")
        with _emb: em_min_adtv_6m  = parse_adtv(st.text_input("6M", value="1,000,000", key="em_6m_v2", label_visibility="collapsed",  disabled=not em_use_6m))
        _emc, _emd = st.columns([3, 4])
        with _emc: em_use_12m = st.checkbox("12M", value=False, key="em_use_12m")
        with _emd: em_min_adtv_12m = parse_adtv(st.text_input("12M", value="1,000,000", key="em_12m_v2", label_visibility="collapsed", disabled=not em_use_12m))
        if not em_use_1m:  em_min_adtv_1m  = 0
        if not em_use_3m:  em_min_adtv_3m  = 0
        if not em_use_6m:  em_min_adtv_6m  = 0
        if not em_use_12m: em_min_adtv_12m = 0

        st.divider()
        st.markdown("""**Liquidity Ratio %** <span title="(12M ADTV / FF MCap) × 252" style="cursor:help;color:#8892b0;font-size:13px;">ⓘ</span>""", unsafe_allow_html=True)
        _lqa, _lqb = st.columns([3, 4])
        with _lqa: use_liq_ratio = st.checkbox("aktiv", value=True, key="use_liq_ratio")
        with _lqb:
            liq_ratio_thr = st.text_input("Min. %", value="10", key="liq_ratio_thr",
                label_visibility="collapsed", disabled=not use_liq_ratio)
        try:    liq_ratio_min = float(liq_ratio_thr) if use_liq_ratio else 0.0
        except: liq_ratio_min = 0.0

    st.markdown("---")
    st.markdown("### ⚖️ Weighting")
    with st.expander("Inclusion Factors", expanded=False):
        _info = "<span title='Adjusted FF MCap = FF MCap × Factor. Adjusted Weight = Adjusted FF MCap / Σ Adjusted FF MCap aller Index-Stocks.' style='cursor:help;color:#8892b0;font-size:13px;'>ⓘ</span>"

        _cna, _cnb = st.columns([4, 2])
        with _cna: use_china_factor = st.checkbox("China A-Shares", help="Adjusted FF MCap = FF MCap × Factor", value=True, key="use_china_factor")
        with _cnb: china_factor_raw = st.text_input("China", value="20", key="china_factor_input", label_visibility="collapsed", disabled=not use_china_factor)
        try:    china_inclusion_factor = float(china_factor_raw) / 100 if use_china_factor else 1.0
        except: china_inclusion_factor = 0.20

        _ina, _inb = st.columns([4, 2])
        with _ina: use_india_factor = st.checkbox("Indien", help="Adjusted FF MCap = FF MCap × Factor", value=True, key="use_india_factor")
        with _inb: india_factor_raw = st.text_input("Indien", value="75", key="india_factor_input_v2", label_visibility="collapsed", disabled=not use_india_factor)
        try:    india_inclusion_factor = float(india_factor_raw) / 100 if use_india_factor else 1.0
        except: india_inclusion_factor = 0.75

        _vna, _vnb = st.columns([4, 2])
        with _vna: use_vietnam_factor = st.checkbox("Vietnam", help="Adjusted FF MCap = FF MCap × Factor", value=True, key="use_vietnam_factor")
        with _vnb: vietnam_factor_raw = st.text_input("Vietnam", value="50", key="vietnam_factor_input", label_visibility="collapsed", disabled=not use_vietnam_factor)
        try:    vietnam_inclusion_factor = float(vietnam_factor_raw) / 100 if use_vietnam_factor else 1.0
        except: vietnam_inclusion_factor = 0.50

        _saa, _sab = st.columns([4, 2])
        with _saa: use_saudi_factor = st.checkbox("Saudi-Arabien", help="Adjusted FF MCap = FF MCap × Factor", value=True, key="use_saudi_factor")
        with _sab: saudi_factor_raw = st.text_input("Saudi-Arabien", value="50", key="saudi_factor_input", label_visibility="collapsed", disabled=not use_saudi_factor)
        try:    saudi_inclusion_factor = float(saudi_factor_raw) / 100 if use_saudi_factor else 1.0
        except: saudi_inclusion_factor = 0.50

    st.markdown("---")
    st.markdown("<div style='color:#8892b0;font-size:11px;'>NaroIX Index Construction Tool<br/>© 2025 NaroIX</div>", unsafe_allow_html=True)


# ─── Load Data ─────────────────────────────────────────────────────────────────
if uploaded:
    df_raw = load_excel(uploaded)
else:
    st.info("👆 Bitte eine Excel-Datei hochladen um zu starten.")
    st.stop()

# Save original raw data for Step 8 (Secondary Share Classes)
df_raw_original = df_raw.copy()

# ── Apply Country Classification from repo file ──────────────────────────────
country_cls = load_classification()
if country_cls:
    # New mapping logic:
    # IF Exchange Country Name == Country of Incorp → use Exchange Country Name
    # ELSE → use Country of Risk
    def map_classification(row):
        if row.get("Exchange Country Name","") == row.get("Country of Incorp",""):
            return country_cls.get(row["Exchange Country Name"])
        else:
            return country_cls.get(row.get("Country of Risk",""))

    df_raw["Mapping Country"] = df_raw.apply(
        lambda r: r["Exchange Country Name"] if r.get("Exchange Country Name","") == r.get("Country of Incorp","")
                  else r.get("Country of Risk",""), axis=1
    )
    df_raw["Classification"] = df_raw["Mapping Country"].map(country_cls)

    unmatched = df_raw[df_raw["Classification"].isna()]["Mapping Country"].unique()
    if len(unmatched) > 0:
        st.warning(f"⚠️ {len(unmatched)} Länder nicht in Country_Classification.xlsx gefunden "
                   f"und werden ignoriert: {', '.join(sorted([str(u) for u in unmatched if u]))}")
    df_raw = df_raw[df_raw["Classification"].notna()].copy()

# Apply listing filter
if listing_filter == "Primary only":
    df_raw = df_raw[df_raw["Listing"] == "Primary"].copy()

# ── Fixed Universe Filter: remove stocks with FF MCap <= 0 or missing ────────
df_raw = df_raw[df_raw["Free Float MCap Y2025"].notna() & (df_raw["Free Float MCap Y2025"] > 0)].copy()

# Exclude Hong Kong CNY stocks (Exchange Ticker contains HKG and Trading Currency = CNY)
if exclude_hk_cny:
    hk_cny_mask = (df_raw["Exchange Ticker"].str.contains("HKG", na=False)) &                   (df_raw["Trading Currency"] == "CNY")
    df_raw = df_raw[~hk_cny_mask].copy()

# Exclude Country of Risk = @
if exclude_country_risk_na and "Country of Risk" in df_raw.columns:
    df_raw = df_raw[df_raw["Country of Risk"].fillna("") != "@NA"].copy()

# Exclude NAICS = Open-End Investment Funds
if exclude_naics_funds and "NAICS" in df_raw.columns:
    df_raw = df_raw[~df_raw["NAICS"].fillna("").str.contains("Open-End Investment Fund", case=False, na=False)].copy()

# Exclude Exchange Name = Euro MTF or @
if exclude_euro_mtf:
    df_raw = df_raw[~df_raw["Exchange Name"].fillna("").isin(["Euro MTF", "@NA"])].copy()

# Exclude Name contains standalone word ETF, SICAV or %
if exclude_etf_sicav:
    import re
    _etf_pattern = re.compile(r'ETF|SICAV', re.IGNORECASE)
    df_raw = df_raw[~df_raw["Name"].fillna("").str.contains(_etf_pattern)].copy()

# Thailand Sec Type filter
_thailand_mask = df_raw["Exchange Name"].fillna("").str.upper() == "THAILAND"
_excluded_type = "NVDR" if thailand_sec_type == "SHARE" else "SHARE"
df_raw = df_raw[~(_thailand_mask & (df_raw["Sec Type"].fillna("") == _excluded_type))].copy()

df_dm_full = df_raw[df_raw["Classification"] == "DM"].copy()
df_em_full = df_raw[df_raw["Classification"] == "EM"].copy()

# Apply pre-filter if selected (before segment computation)
if filter_timing.startswith("Pre"):
    df_dm = apply_min_filters(df_dm_full, dm_ff_gt0, min_ff_pct, max_closing_price, dm_min_adtv_1m, dm_min_adtv_3m, dm_min_adtv_6m, dm_min_adtv_12m, liq_ratio_min)
    df_em = apply_min_filters(df_em_full, em_ff_gt0, min_ff_pct, max_closing_price, em_min_adtv_1m, em_min_adtv_3m, em_min_adtv_6m, em_min_adtv_12m, liq_ratio_min)
else:
    df_dm = df_dm_full.copy()
    df_em = df_em_full.copy()

# Post-segment filter lambdas — applied AFTER segment computation
_filter_is_pre = filter_timing.startswith("Pre")

def dm_post_filter(df):
    if _filter_is_pre: return df  # already applied pre-segment
    return apply_min_filters(df, dm_ff_gt0, min_ff_pct, max_closing_price, dm_min_adtv_1m, dm_min_adtv_3m, dm_min_adtv_6m, dm_min_adtv_12m, liq_ratio_min)

def em_post_filter(df):
    if _filter_is_pre: return df  # already applied pre-segment
    return apply_min_filters(df, em_ff_gt0, min_ff_pct, max_closing_price, em_min_adtv_1m, em_min_adtv_3m, em_min_adtv_6m, em_min_adtv_12m, liq_ratio_min)

# ── Pre-compute segments once — reused across all tabs ────────────────────────
_seg_dm_v1 = compute_variant1(df_dm, large_thr, mid_thr, small_thr)
_seg_dm_v2 = compute_variant2(df_dm, large_thr, mid_thr, small_thr)
_seg_em_g85 = compute_variant1(df_em, large_thr, mid_thr, small_thr)
_cutoff_v1 = get_dm_cutoff_stock(_seg_dm_v1)
_cutoff_mcap_v1 = _cutoff_v1["Total MCap Y2025"] if _cutoff_v1 is not None else 0


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📊 NaroIX Index Construction Tool")
st.markdown("<div style='color:#8892b0;margin-top:-12px;margin-bottom:20px;'>ACWI & World Index — Free Float Market Cap Methodology</div>", unsafe_allow_html=True)

# Top metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Universe", f"{len(df_raw):,}")
c2.metric("DM Stocks", f"{len(df_dm):,}")
c3.metric("EM Stocks", f"{len(df_em):,}")
c4.metric("DM FF MCap", format_bn(df_dm["Free Float MCap Y2025"].sum()))
c5.metric("EM FF MCap", format_bn(df_em["Free Float MCap Y2025"].sum()))

# Show active filter summary (post-segment filters)
active_filters = []
if dm_min_adtv_1m  > 0: active_filters.append(f"DM 1M ADTV ≥ {dm_min_adtv_1m:,.0f} USD")
if dm_min_adtv_3m  > 0: active_filters.append(f"DM 3M ADTV ≥ {dm_min_adtv_3m:,.0f} USD")
if dm_min_adtv_6m  > 0: active_filters.append(f"DM 6M ADTV ≥ {dm_min_adtv_6m:,.0f} USD")
if dm_min_adtv_12m > 0: active_filters.append(f"DM 12M ADTV ≥ {dm_min_adtv_12m:,.0f} USD")
if em_min_adtv_1m  > 0: active_filters.append(f"EM 1M ADTV ≥ {em_min_adtv_1m:,.0f} USD")
if em_min_adtv_3m  > 0: active_filters.append(f"EM 3M ADTV ≥ {em_min_adtv_3m:,.0f} USD")
if em_min_adtv_6m  > 0: active_filters.append(f"EM 6M ADTV ≥ {em_min_adtv_6m:,.0f} USD")
if em_min_adtv_12m > 0: active_filters.append(f"EM 12M ADTV ≥ {em_min_adtv_12m:,.0f} USD")
if min_ff_pct      > 0: active_filters.append(f"FF% ≥ {min_ff_pct*100:.1f}%")
if max_closing_price is not None: active_filters.append(f"Closing Price ≤ {max_closing_price:,.0f}")
if liq_ratio_min   > 0: active_filters.append(f"Liquidity Ratio ≥ {liq_ratio_min:.1f}%")
if active_filters:
    timing_label = "vor" if _filter_is_pre else "nach"
    timing_mode  = "Pre" if _filter_is_pre else "Post"
    st.markdown(f"""<div class="warning-box">
    ⚠️ <b>Aktive Filter ({timing_mode}):</b> {" &nbsp;|&nbsp; ".join(active_filters)}<br>
    Filter werden <b>{timing_label}</b> der Segment-Berechnung angewendet.
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ─── Tabs ──────────────────────────────────────────────────────────────────────
# ── Global ACWI Settings (used in Overview sensitivity table and ACWI tab) ──
_prev_variant_g = st.session_state.get("_prev_acwi_variant", "Variant 1 (Global / MSCI)")
_acwi_variant_g = st.session_state.get("acwi_variant_radio", "Variant 1 (Global / MSCI)")
if _prev_variant_g == "Variant 2 (Per-Country / Solactive)" and _acwi_variant_g == "Variant 1 (Global / MSCI)":
    st.session_state["em_method_radio"] = "Threshold (% des DM Grenzstocks)"

if _acwi_variant_g == "Variant 2 (Per-Country / Solactive)":
    _em_method_g = "Per-Country 85% (wie DM Variant 2)"
else:
    _em_method_g = st.session_state.get("em_method_radio", "Threshold (% des DM Grenzstocks)")

tab_overview, tab_v1, tab_v2, tab_v3, tab_acwi, tab_compare, tab_acwi_compare = st.tabs([
    "🌍 Universe Overview",
    "📐 Variant 1 — Global / MSCI DM Thresholds",
    "🗺️ Variant 2 — Per-Country / Solactive DM Thresholds",
    "⚡ Variant 3 — Dynamic (GMSR/EUMSS)",
    "🌐 ACWI (DM + EM)",
    "⚖️ Variant Comparison (World)",
    "🔀 ACWI Comparison (V1 vs V2)",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: UNIVERSE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Universe Overview")

    col1, col2 = st.columns(2)

    with col1:
        # DM countries
        dm_country = df_dm.groupby("Exchange Country Name").agg(
            Stocks=("Symbol", "count"),
            FF_MCap=("Free Float MCap Y2025", "sum")
        ).sort_values("FF_MCap", ascending=False).reset_index()
        dm_country["FF MCap (USD)"] = dm_country["FF_MCap"].apply(format_bn)
        dm_country["Share (%)"] = (dm_country["FF_MCap"] / dm_country["FF_MCap"].sum() * 100).round(2)

        st.markdown("**DM Countries by FF MCap**")
        st.dataframe(
            dm_country[["Exchange Country Name", "Stocks", "FF MCap (USD)", "Share (%)"]],
            use_container_width=True, height=380, hide_index=True
        )

    with col2:
        em_country = df_em.groupby("Exchange Country Name").agg(
            Stocks=("Symbol", "count"),
            FF_MCap=("Free Float MCap Y2025", "sum")
        ).sort_values("FF_MCap", ascending=False).reset_index()
        em_country["FF MCap (USD)"] = em_country["FF_MCap"].apply(format_bn)
        em_country["Share (%)"] = (em_country["FF_MCap"] / em_country["FF_MCap"].sum() * 100).round(2)

        st.markdown("**EM Countries by FF MCap**")
        st.dataframe(
            em_country[["Exchange Country Name", "Stocks", "FF MCap (USD)", "Share (%)"]],
            use_container_width=True, height=380, hide_index=True
        )

    # Treemap
    st.markdown("**FF MCap Distribution by Country**")
    all_country = df_raw.groupby(["Classification","Exchange Country Name"]).agg(
        FF_MCap=("Free Float MCap Y2025","sum")
    ).reset_index()
    fig = px.treemap(
        all_country, path=["Classification","Exchange Country Name"],
        values="FF_MCap", color="Classification",
        color_discrete_map={"DM":"#2979ff","EM":"#ce93d8"},
        template="plotly_dark",
    )
    fig.update_layout(height=450, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── ADTV Sensitivity Table ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**ADTV Threshold Sensitivity — World Index & ACWI**")
    st.caption(f"Verwendet: DM {_acwi_variant_g} + EM {_em_method_g}. Alle anderen aktiven Filter bleiben konstant.")

    # DM segments based on selected variant
    _sens_dm = _seg_dm_v1 if _acwi_variant_g == "Variant 1 (Global / MSCI)" else _seg_dm_v2

    # EM base universe based on selected method
    _sens_cutoff = _cutoff_v1["Total MCap Y2025"] if _cutoff_v1 is not None else 0
    if _em_method_g == "Threshold (% des DM Grenzstocks)":
        _sens_em_df, _ = filter_em_by_threshold(df_em, _sens_cutoff, em_threshold_pct)
        _sens_em_base_seg = "EM Included"
    elif _em_method_g == "Global 85% (wie DM Variant 1)":
        _sens_em_df = _seg_em_g85
        _sens_em_base_seg = "large_mid"
    else:  # Per-Country
        _sens_em_df = filter_em_per_country(df_em, pct=mid_thr)
        _sens_em_base_seg = "EM Included"

    # Base filters (everything except ADTV)
    def _base_filter_dm(df):
        return apply_min_filters(df, dm_ff_gt0, min_ff_pct, max_closing_price,
                                  0, 0, 0, 0, liq_ratio_min)
    def _base_filter_em(df):
        return apply_min_filters(df, em_ff_gt0, min_ff_pct, max_closing_price,
                                  0, 0, 0, 0, liq_ratio_min)

    # Use sidebar field values directly (ignoring checkbox state)
    _adtv_periods = {
        "1M":  ("1M ADTV Y2025",  parse_adtv(st.session_state.get("dm_1m_v2",  "2000000")), parse_adtv(st.session_state.get("em_1m_v2",  "1000000"))),
        "3M":  ("3M ADTV Y2025",  parse_adtv(st.session_state.get("dm_3m_v2",  "2000000")), parse_adtv(st.session_state.get("em_3m_v2",  "1000000"))),
        "6M":  ("6M ADTV Y2025",  parse_adtv(st.session_state.get("dm_6m_v2",  "2000000")), parse_adtv(st.session_state.get("em_6m_v2",  "1000000"))),
        "12M": ("12M ADTV Y2025", parse_adtv(st.session_state.get("dm_12m_v2", "2000000")), parse_adtv(st.session_state.get("em_12m_v2", "1000000"))),
    }
    _combos = [
        ("1M",), ("3M",), ("6M",), ("12M",),
        ("1M","3M"), ("1M","6M"), ("1M","12M"),
        ("3M","6M"), ("3M","12M"), ("6M","12M"),
        ("1M","3M","6M"), ("1M","3M","12M"), ("1M","6M","12M"), ("3M","6M","12M"),
        ("1M","3M","6M","12M"),
    ]

    # Pre-compute EM base (before ADTV)
    if _sens_em_base_seg == "large_mid":
        _sens_em_base = _base_filter_em(_sens_em_df[_sens_em_df["Segment"].isin(["Large Cap","Mid Cap"])])
    else:
        _sens_em_base = _base_filter_em(_sens_em_df[_sens_em_df["Segment"] == "EM Included"])

    _sens_rows = []
    for combo in _combos:
        label = " + ".join(combo)
        _dm_f = _base_filter_dm(_sens_dm[_sens_dm["Segment"].isin(["Large Cap","Mid Cap","Small Cap"])])
        _em_f = _sens_em_base.copy()
        for period in combo:
            col, dm_thr, em_thr = _adtv_periods[period]
            if dm_thr > 0: _dm_f = _dm_f[_dm_f[col] >= dm_thr]
            if em_thr > 0: _em_f = _em_f[_em_f[col] >= em_thr]

        _dm_lc = len(_dm_f[_dm_f["Segment"] == "Large Cap"])
        _dm_mc = len(_dm_f[_dm_f["Segment"] == "Mid Cap"])
        _wld   = _dm_lc + _dm_mc

        # EM segments based on method
        if _sens_em_base_seg == "large_mid":
            # Global 85%: segments already assigned
            _em_lc = len(_em_f[_em_f["Segment"] == "Large Cap"])
            _em_mc = len(_em_f[_em_f["Segment"] == "Mid Cap"])
        else:
            # Threshold or Per-Country: rank within qualified stocks
            _em_total_ff = _em_f["Free Float MCap Y2025"].sum()
            if _em_total_ff > 0 and len(_em_f) > 0:
                _em_r = _em_f.sort_values("Total MCap Y2025", ascending=False).copy()
                _em_r["cum_pct"] = _em_r["Free Float MCap Y2025"].cumsum() / _em_total_ff * 100
                _em_r["EM Seg"] = np.where(_em_r["cum_pct"] <= large_thr, "Large Cap", "Mid Cap")
                _em_lc = len(_em_r[_em_r["EM Seg"] == "Large Cap"])
                _em_mc = len(_em_r[_em_r["EM Seg"] == "Mid Cap"])
            else:
                _em_lc = _em_mc = 0

        _acwi = _wld + len(_em_f)
        _sens_rows.append({
            "ADTV Kombination": label,
            "DM Large Cap": _dm_lc,
            "DM Mid Cap":   _dm_mc,
            "World Index":  _wld,
            "EM Large Cap": _em_lc,
            "EM Mid Cap":   _em_mc,
            "ACWI Index":   _acwi,
        })

    _sens_df = pd.DataFrame(_sens_rows)
    st.dataframe(_sens_df, use_container_width=True, hide_index=True, height=575)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: VARIANT 1 — GLOBAL DM THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════
with tab_v1:
    st.subheader("Variant 1 — Global / MSCI DM Thresholds")
    st.markdown(f"""
    <div class="info-box">
    <b>Methodik:</b> Alle DM-Aktien werden global nach Free Float MCap absteigend sortiert.
    Die kumulierte FF MCap bestimmt die Segmentzugehörigkeit:<br>
    • <b>Large Cap</b>: kum. FF MCap ≤ {large_thr}% &nbsp;|&nbsp;
    • <b>Mid Cap</b>: {large_thr}% &lt; kum. FF MCap ≤ {mid_thr}% &nbsp;|&nbsp;
    • <b>Small Cap</b>: {mid_thr}% &lt; kum. FF MCap ≤ {small_thr}% <br>
    • <b>World Index (DM)</b> = Large Cap + Mid Cap (bis {mid_thr}%)
    </div>
    """, unsafe_allow_html=True)

    df_v1 = _seg_dm_v1

    # Summary table
    summary = segment_summary(df_v1)
    total_dm_ff = df_dm["Free Float MCap Y2025"].sum()

    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.markdown("**Segment Summary**")
        disp = summary.copy()
        disp["FF MCap (USD)"] = disp["FF MCap (USD)"].apply(format_bn)
        disp["Avg FF MCap (USD)"] = disp["Avg FF MCap (USD)"].apply(format_bn)
        disp["FF MCap Coverage"] = summary.apply(
            lambda r: f"{r['FF MCap (USD)']/total_dm_ff*100:.1f}%" if r["Segment"]!="Total Included" else
                      f"{r['FF MCap (USD)']/total_dm_ff*100:.1f}%", axis=1
        )
        st.dataframe(disp, use_container_width=True, hide_index=True)

        world_idx = dm_post_filter(df_v1[df_v1["Segment"].isin(["Large Cap","Mid Cap"])])
        st.success(f"🌐 **World Index (DM):** {len(world_idx):,} Stocks | {format_bn(world_idx['Free Float MCap Y2025'].sum())} FF MCap")

    with col_s2:
        # Pie chart
        seg_data = df_v1.groupby("Segment")["Free Float MCap Y2025"].sum().reset_index()
        fig_pie = px.pie(
            seg_data, names="Segment", values="Free Float MCap Y2025",
            color="Segment", color_discrete_map=SEGMENT_COLORS,
            template="plotly_dark", hole=0.45,
            title="FF MCap by Segment (Variant 1)",
        )
        fig_pie.update_layout(paper_bgcolor="#0f1117", height=300, margin=dict(t=40,b=0,l=0,r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Cumulative curve
    st.markdown("**Cumulative FF MCap Curve (DM Universe)**")
    df_curve = df_v1.copy()
    df_curve["rank"] = range(1, len(df_curve)+1)
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=df_curve["rank"], y=df_curve["cum_pct"],
        mode="lines", line=dict(color="#2979ff", width=2),
        fill="tozeroy", fillcolor="rgba(41,121,255,0.1)",
        name="Cumulative FF MCap %"
    ))
    for thr, col, lbl in [(large_thr,"#00e676",f"Large Cap ({large_thr}%)"),(mid_thr,"#ff9100",f"Mid Cap ({mid_thr}%)"),(small_thr,"#f44336",f"Small Cap ({small_thr}%)")]:
        fig_cum.add_hline(y=thr, line_dash="dash", line_color=col,
                          annotation_text=lbl, annotation_position="right")
    fig_cum.update_layout(
        template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
        height=300, xaxis_title="Rank (by FF MCap)", yaxis_title="Cumulative FF MCap (%)",
        margin=dict(t=10,b=40,l=60,r=120), showlegend=False,
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Country breakdown in World Index
    st.markdown("**World Index — Country Breakdown**")
    world_country = world_idx.groupby("Exchange Country Name").agg(
        Stocks=("Symbol","count"),
        FF_MCap=("Free Float MCap Y2025","sum"),
        Large=("Segment", lambda x: (x=="Large Cap").sum()),
        Mid=("Segment", lambda x: (x=="Mid Cap").sum()),
    ).sort_values("FF_MCap", ascending=False).reset_index()
    world_country["FF MCap (USD)"] = world_country["FF_MCap"].apply(format_bn)
    world_country["Weight (%)"] = (world_country["FF_MCap"]/world_country["FF_MCap"].sum()*100).round(2)
    st.dataframe(world_country.drop(columns="FF_MCap"), use_container_width=True, hide_index=True)

    # Download
    _fp_dm = dict(ff_gt0=dm_ff_gt0, min_ff_pct=min_ff_pct, max_closing_price=max_closing_price,
        min_adtv_1m=dm_min_adtv_1m, min_adtv_3m=dm_min_adtv_3m,
        min_adtv_6m=dm_min_adtv_6m, min_adtv_12m=dm_min_adtv_12m, min_liq_ratio=liq_ratio_min)
    export_v1 = build_full_export(df_v1, "DM", dm_post_filter, _fp_dm)
    export_v1["cum_pct"] = export_v1["cum_pct"].round(4)
    _drop = ["cum_ff"]
    _v1_universe = export_v1.drop(columns=_drop, errors="ignore")
    _v1_world    = add_adjusted_weight(add_ff_weight(_v1_universe[(_v1_universe["Segment"].isin(["Large Cap","Mid Cap"])) & (_v1_universe["Status"]=="Included")]), china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)
    st.download_button(
        "⬇️ Download Variant 1 as Excel",
        data=to_excel_multi({"Universe": _v1_universe, "World Index (DM)": _v1_world}),
        file_name="NaroIX_World_Variant1.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: VARIANT 2 — PER-COUNTRY DM THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════
with tab_v2:
    st.subheader("Variant 2 — Per-Country / Solactive DM Thresholds")
    st.markdown(f"""
    <div class="info-box">
    <b>Methodik:</b> Das Prozedere aus Variant 1 wird für <b>jedes DM-Land separat</b> angewendet.
    Innerhalb jedes Landes werden die Aktien nach FF MCap sortiert und die Segmente auf Basis der
    länderspezifischen kumulierten FF MCap-Anteile bestimmt.<br>
    • <b>World Index (DM)</b> = Large Cap + Mid Cap je Land (bis {mid_thr}% kumuliert pro Land)
    </div>
    """, unsafe_allow_html=True)

    df_v2 = _seg_dm_v2

    summary_v2 = segment_summary(df_v2)
    total_dm_ff = df_dm["Free Float MCap Y2025"].sum()

    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.markdown("**Segment Summary**")
        disp2 = summary_v2.copy()
        disp2["FF MCap (USD)"] = disp2["FF MCap (USD)"].apply(format_bn)
        disp2["Avg FF MCap (USD)"] = disp2["Avg FF MCap (USD)"].apply(format_bn)
        disp2["FF MCap Coverage"] = summary_v2.apply(
            lambda r: f"{r['FF MCap (USD)']/total_dm_ff*100:.1f}%", axis=1
        )
        st.dataframe(disp2, use_container_width=True, hide_index=True)

        world_v2 = dm_post_filter(df_v2[df_v2["Segment"].isin(["Large Cap","Mid Cap"])])
        st.success(f"🌐 **World Index (DM):** {len(world_v2):,} Stocks | {format_bn(world_v2['Free Float MCap Y2025'].sum())} FF MCap")

    with col_s2:
        seg_data2 = df_v2.groupby("Segment")["Free Float MCap Y2025"].sum().reset_index()
        fig_pie2 = px.pie(
            seg_data2, names="Segment", values="Free Float MCap Y2025",
            color="Segment", color_discrete_map=SEGMENT_COLORS,
            template="plotly_dark", hole=0.45,
            title="FF MCap by Segment (Variant 2)",
        )
        fig_pie2.update_layout(paper_bgcolor="#0f1117", height=300, margin=dict(t=40,b=0,l=0,r=0))
        st.plotly_chart(fig_pie2, use_container_width=True)

    # Per-country breakdown
    st.markdown("**Per-Country Breakdown — Variant 2**")
    country_breakdown = df_v2.groupby(["Exchange Country Name","Segment"]).agg(
        Stocks=("Symbol","count"),
        FF_MCap=("Free Float MCap Y2025","sum"),
    ).reset_index()

    country_pivot = country_breakdown.pivot_table(
        index="Exchange Country Name", columns="Segment", values="Stocks", fill_value=0
    ).reset_index()
    country_ff = df_v2.groupby("Exchange Country Name")["Free Float MCap Y2025"].sum().reset_index()
    country_ff.columns = ["Exchange Country Name","Total FF MCap"]
    country_pivot = country_pivot.merge(country_ff, on="Exchange Country Name")
    country_pivot["Total FF MCap"] = country_pivot["Total FF MCap"].apply(format_bn)

    # Reorder columns
    seg_cols = [c for c in ["Large Cap","Mid Cap","Small Cap","Micro / Excluded"] if c in country_pivot.columns]
    country_pivot = country_pivot.sort_values("Exchange Country Name")
    st.dataframe(country_pivot[["Exchange Country Name"]+seg_cols+["Total FF MCap"]], use_container_width=True, hide_index=True)

    # Stacked bar per country
    st.markdown("**Stocks per Country and Segment**")
    bar_data = df_v2[df_v2["Segment"] != "Micro / Excluded"].groupby(
        ["Exchange Country Name","Segment"]
    )["Free Float MCap Y2025"].sum().reset_index()

    fig_bar = px.bar(
        bar_data.sort_values("Free Float MCap Y2025", ascending=False),
        x="Exchange Country Name", y="Free Float MCap Y2025",
        color="Segment", barmode="stack",
        color_discrete_map=SEGMENT_COLORS,
        template="plotly_dark",
        labels={"Free Float MCap Y2025":"FF MCap (USD)","Exchange Country Name":"Country"},
    )
    fig_bar.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#161b27", height=400,
        xaxis_tickangle=-45, margin=dict(t=10,b=100,l=60,r=10)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    _fp_dm2 = dict(ff_gt0=dm_ff_gt0, min_ff_pct=min_ff_pct, max_closing_price=max_closing_price,
        min_adtv_1m=dm_min_adtv_1m, min_adtv_3m=dm_min_adtv_3m,
        min_adtv_6m=dm_min_adtv_6m, min_adtv_12m=dm_min_adtv_12m, min_liq_ratio=liq_ratio_min)
    export_v2 = build_full_export(df_v2, "DM", dm_post_filter, _fp_dm2)
    export_v2["cum_pct"] = export_v2["cum_pct"].round(4)
    _v2_universe = export_v2.drop(columns=["cum_ff"], errors="ignore")
    _v2_world    = add_adjusted_weight(add_ff_weight(_v2_universe[(_v2_universe["Segment"].isin(["Large Cap","Mid Cap"])) & (_v2_universe["Status"]=="Included")]), china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)
    st.download_button(
        "⬇️ Download Variant 2 as Excel",
        data=to_excel_multi({"Universe": _v2_universe, "World Index (DM)": _v2_world}),
        file_name="NaroIX_World_Variant2.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: VARIANT 3 — DYNAMIC (GMSR/EUMSS)
# ══════════════════════════════════════════════════════════════════════════════
with tab_v3:
    st.subheader("Variant 3 — Dynamic GMSR/EUMSS (MSCI-Style)")

    # ── Run full V3 pipeline (Steps 3-6) ────────────────────────────────────────
    _df_all = pd.concat([df_dm_full, df_em_full], ignore_index=True)
    _df_v3, _v3_eumss_full, _v3_eumss_ff, _v3_dm_gmsr, _v3_em_gmsr = compute_variant3(
        _df_all,
        eumss_pct=v3_eumss_pct, eumss_ff_ratio=v3_eumss_ff_ratio, min_ff_pct=v3_min_ff_pct,
        gmsr_pct=v3_gmsr_pct, em_gmsr_ratio=v3_em_gmsr_ratio,
        auto_mult=v3_auto_mult, cand_mult=v3_cand_mult, buf_mult=v3_buf_mult,
        atvr_dm_min=v3_atvr_dm_min, atvr_em_min=v3_atvr_em_min,
        adtv_dm_3m=v3_adtv_dm, adtv_dm_6m=v3_adtv_dm,
        adtv_em_3m=v3_adtv_em, adtv_em_6m=v3_adtv_em,
        china_if=china_inclusion_factor, india_if=india_inclusion_factor,
        vietnam_if=vietnam_inclusion_factor, saudi_if=saudi_inclusion_factor,
    )

    # Key metrics — EUMSS cards
    _km1, _km2, _km3, _km4 = st.columns(4)
    _km1.metric("EUMSS_FULL (Total MCap Min.)", format_bn(_v3_eumss_full))
    _km2.metric("EUMSS_FF (FF MCap Min.)",      format_bn(_v3_eumss_ff))
    _km3.metric("DM GMSR (85% Cutoff)",         format_bn(_v3_dm_gmsr))
    _km4.metric("EM GMSR (50% × DM GMSR)",      format_bn(_v3_em_gmsr))

    # Definitions table
    with st.expander("📖 Begriffsdefinitionen & Berechnete Werte", expanded=False):
        _def_html = f"""
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
          <tr style="background:#0f1117;color:#8892b0;text-align:left;">
            <th style="padding:8px 12px;width:15%;">Begriff</th>
            <th style="padding:8px 12px;width:18%;">Berechneter Wert</th>
            <th style="padding:8px 12px;">Definition</th>
          </tr>
        </thead>
        <tbody>
          <tr style="background:#161b27;">
            <td style="padding:10px 12px;color:#42a5f5;font-weight:700;">EUMSS</td>
            <td style="padding:10px 12px;color:#e8eaf6;">—</td>
            <td style="padding:10px 12px;color:#8892b0;">Equity Universe Minimum Size Standard. Oberbegriff für die Mindestgrößenanforderung die ein Titel erfüllen muss um überhaupt in das investierbare Universum aufgenommen zu werden. Kein fester Zahlenwert — wird bei jedem Lauf neu aus den Daten berechnet.</td>
          </tr>
          <tr style="background:#1a1f2e;">
            <td style="padding:10px 12px;color:#42a5f5;font-weight:700;">EUMSS_FULL</td>
            <td style="padding:10px 12px;color:#e8eaf6;font-weight:600;">{format_bn(_v3_eumss_full)} USD</td>
            <td style="padding:10px 12px;color:#8892b0;">Konkreter Zahlenwert des EUMSS als Total Market Cap. Berechnung: alle DM-Titel nach Total MCap absteigend sortieren, Free Float MCap kumuliert aufaddieren, der Total MCap des Titels an der {v3_eumss_pct*100:.0f}%-Marke = EUMSS_FULL. Verwendung: untere Grenze für die Total MCap jedes Titels.</td>
          </tr>
          <tr style="background:#161b27;">
            <td style="padding:10px 12px;color:#42a5f5;font-weight:700;">EUMSS_FF</td>
            <td style="padding:10px 12px;color:#e8eaf6;font-weight:600;">{format_bn(_v3_eumss_ff)} USD</td>
            <td style="padding:10px 12px;color:#8892b0;">Untere Grenze für die Free Float MCap jedes Titels. Berechnung: EUMSS_FF = EUMSS_FULL × {v3_eumss_ff_ratio*100:.0f}%. Fixer Abschlag — keine eigene Kalibrierung. Stellt sicher dass auch der tatsächlich handelbare Anteil ausreichend groß ist.</td>
          </tr>
          <tr style="background:#1a1f2e;">
            <td style="padding:10px 12px;color:#66bb6a;font-weight:700;">GMSR</td>
            <td style="padding:10px 12px;color:#e8eaf6;">—</td>
            <td style="padding:10px 12px;color:#8892b0;">Global Minimum Size Reference. Oberbegriff für den Referenzpunkt der die Segmentgrenze definiert. Bestimmt wo die Grenze zwischen Standard Index (Large + Mid Cap) und Small Cap liegt. Kein Ausschlussfilter wie EUMSS — sondern eine Segmenteinteilung. Wird bei jedem Lauf neu berechnet.</td>
          </tr>
          <tr style="background:#161b27;">
            <td style="padding:10px 12px;color:#66bb6a;font-weight:700;">DM_GMSR</td>
            <td style="padding:10px 12px;color:#e8eaf6;font-weight:600;">{format_bn(_v3_dm_gmsr)} USD</td>
            <td style="padding:10px 12px;color:#8892b0;">Konkreter Zahlenwert des GMSR für Developed Markets. Berechnung: alle verbleibenden DM-Titel nach Total MCap absteigend sortieren, Adj_FF_MCap kumuliert aufaddieren, der Total MCap des Titels an der {v3_gmsr_pct*100:.0f}%-Marke = DM_GMSR. Ankerpunkt für alle vier Zonenschwellen (× {v3_auto_mult} / × {v3_cand_mult} / × {v3_buf_mult}).</td>
          </tr>
          <tr style="background:#1a1f2e;">
            <td style="padding:10px 12px;color:#66bb6a;font-weight:700;">EM_GMSR</td>
            <td style="padding:10px 12px;color:#e8eaf6;font-weight:600;">{format_bn(_v3_em_gmsr)} USD</td>
            <td style="padding:10px 12px;color:#8892b0;">Konkreter Zahlenwert des GMSR für Emerging Markets. Berechnung: EM_GMSR = DM_GMSR × {v3_em_gmsr_ratio*100:.0f}%. Fixer MSCI-Parameter — keine eigene Kalibrierung. EM-Unternehmen sind strukturell kleiner als DM-Unternehmen; gleicher GMSR würde viele EM-Länder aus dem Index drängen.</td>
          </tr>
        </tbody>
        </table>
        """
        st.markdown(_def_html, unsafe_allow_html=True)

    # GMSR cards + zone matrix
    st.markdown("---")
    _gc1, _gc2 = st.columns(2)
    with _gc1:
        st.markdown(f"""
        <div style="background:#1a2a4a;border-radius:8px;padding:16px 20px;margin-bottom:8px;">
        <div style="color:#8892b0;font-size:12px;font-weight:600;letter-spacing:1px;">DM GMSR — Standard Index</div>
        <div style="color:#e8eaf6;font-size:28px;font-weight:700;margin:6px 0;">{format_bn(_v3_dm_gmsr)} USD</div>
        <div style="color:#8892b0;font-size:12px;">Kalibriert: 85% DM Free Float MCap Cutoff</div>
        </div>""", unsafe_allow_html=True)
    with _gc2:
        st.markdown(f"""
        <div style="background:#1a2a4a;border-radius:8px;padding:16px 20px;margin-bottom:8px;">
        <div style="color:#8892b0;font-size:12px;font-weight:600;letter-spacing:1px;">EM GMSR — Standard Index</div>
        <div style="color:#e8eaf6;font-size:28px;font-weight:700;margin:6px 0;">{format_bn(_v3_em_gmsr)} USD</div>
        <div style="color:#8892b0;font-size:12px;">= {v3_em_gmsr_ratio*100:.0f}% × DM GMSR — fixer Abschlag (kein eigener Kalibrator)</div>
        </div>""", unsafe_allow_html=True)

    # Zone matrix table
    st.markdown("**ZONENTABELLE — FULL MARKET CAP VS. GMSR**")
    _zone_matrix_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
    <thead>
      <tr style="background:#0f1117;color:#8892b0;text-align:left;">
        <th style="padding:8px 12px;">Zone</th>
        <th style="padding:8px 12px;">Multiplikator</th>
        <th style="padding:8px 12px;">DM Schwelle</th>
        <th style="padding:8px 12px;">EM Schwelle</th>
        <th style="padding:8px 12px;">Tool-Logik</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background:#0d2137;">
        <td style="padding:10px 12px;color:#42a5f5;font-weight:700;">Auto-Include</td>
        <td style="padding:10px 12px;color:#e8eaf6;">&gt; {v3_auto_mult}× GMSR</td>
        <td style="padding:10px 12px;color:#e8eaf6;">&gt; {format_bn(_v3_dm_gmsr * v3_auto_mult)}</td>
        <td style="padding:10px 12px;color:#e8eaf6;">&gt; {format_bn(_v3_em_gmsr * v3_auto_mult)}</td>
        <td style="padding:10px 12px;color:#8892b0;">Aufgenommen — restliche Screens trotzdem prüfen (ADTV, FF%)</td>
      </tr>
      <tr style="background:#0d3320;">
        <td style="padding:10px 12px;color:#66bb6a;font-weight:700;">Kandidat</td>
        <td style="padding:10px 12px;color:#e8eaf6;">{v3_cand_mult}× – {v3_auto_mult}× GMSR</td>
        <td style="padding:10px 12px;color:#e8eaf6;">{format_bn(_v3_dm_gmsr * v3_cand_mult)} – {format_bn(_v3_dm_gmsr * v3_auto_mult)}</td>
        <td style="padding:10px 12px;color:#e8eaf6;">{format_bn(_v3_em_gmsr * v3_cand_mult)} – {format_bn(_v3_em_gmsr * v3_auto_mult)}</td>
        <td style="padding:10px 12px;color:#8892b0;">Eligible — weiter zu Schritt 7 (85% Segmentation)</td>
      </tr>
      <tr style="background:#2d1f00;">
        <td style="padding:10px 12px;color:#ffa726;font-weight:700;">Pufferzone</td>
        <td style="padding:10px 12px;color:#e8eaf6;">{v3_buf_mult}× – {v3_cand_mult}× GMSR</td>
        <td style="padding:10px 12px;color:#e8eaf6;">{format_bn(_v3_dm_gmsr * v3_buf_mult)} – {format_bn(_v3_dm_gmsr * v3_cand_mult)}</td>
        <td style="padding:10px 12px;color:#e8eaf6;">{format_bn(_v3_em_gmsr * v3_buf_mult)} – {format_bn(_v3_em_gmsr * v3_cand_mult)}</td>
        <td style="padding:10px 12px;color:#8892b0;">Nur Bestandsmitglieder bleiben — keine Neuaufnahmen in dieser Zone</td>
      </tr>
      <tr style="background:#2d0d0d;">
        <td style="padding:10px 12px;color:#ef5350;font-weight:700;">Auto-Exclude</td>
        <td style="padding:10px 12px;color:#e8eaf6;">&lt; {v3_buf_mult}× GMSR</td>
        <td style="padding:10px 12px;color:#e8eaf6;">&lt; {format_bn(_v3_dm_gmsr * v3_buf_mult)}</td>
        <td style="padding:10px 12px;color:#e8eaf6;">&lt; {format_bn(_v3_em_gmsr * v3_buf_mult)}</td>
        <td style="padding:10px 12px;color:#8892b0;">Ausschluss — auch bestehende Mitglieder raus</td>
      </tr>
    </tbody>
    </table>
    """
    st.markdown(_zone_matrix_html, unsafe_allow_html=True)
    st.markdown("---")

    # Zone distribution
    _zone_colors = {"AUTO_INCLUDE": "#2979ff", "CANDIDATE": "#66bb6a", "BUFFER": "#ffa726", "EXCLUDE": "#ef5350"}
    _zones_order = ["AUTO_INCLUDE", "CANDIDATE", "BUFFER", "EXCLUDE"]

    _zone_rows = []
    for cls in ["DM", "EM"]:
        for zone in _zones_order:
            _s = _df_v3[(_df_v3["Classification"] == cls) & (_df_v3["Zone"] == zone)]
            _zone_rows.append({
                "Klassifikation": cls,
                "Zone": zone,
                "# Stocks": len(_s),
                "Total MCap (USD)": format_bn(_s["Total MCap Y2025"].sum()),
                "FF MCap (USD)": format_bn(_s["Free Float MCap Y2025"].sum()),
            })
    _zone_df = pd.DataFrame(_zone_rows)

    col_z1, col_z2 = st.columns(2)
    with col_z1:
        st.markdown("**DM — Zonen-Übersicht**")
        st.dataframe(_zone_df[_zone_df["Klassifikation"]=="DM"].drop(columns=["Klassifikation"]),
                     use_container_width=True, hide_index=True)
    with col_z2:
        st.markdown("**EM — Zonen-Übersicht**")
        st.dataframe(_zone_df[_zone_df["Klassifikation"]=="EM"].drop(columns=["Klassifikation"]),
                     use_container_width=True, hide_index=True)

    # Zone chart
    st.markdown("**Zonen-Verteilung**")
    _fig_zone = go.Figure()
    for zone in _zones_order:
        _zd = _zone_df[_zone_df["Zone"] == zone]
        _fig_zone.add_bar(
            name=zone, x=_zd["Klassifikation"], y=_zd["# Stocks"],
            marker_color=_zone_colors[zone],
            text=_zd["# Stocks"], textposition="auto",
        )
    _fig_zone.update_layout(
        barmode="group", template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
        height=320, margin=dict(t=10,b=40,l=60,r=10), legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(_fig_zone, use_container_width=True)

    # ── Step 7: 85% Coverage per country ────────────────────────────────────────
    if use_gmsr_zones:
        # With GMSR zones: only AUTO_INCLUDE + CANDIDATE (+ BUFFER if selected)
        _included_zones = ["AUTO_INCLUDE", "CANDIDATE"]
        if include_buffer:
            _included_zones.append("BUFFER")
        _df_v3_candidates = _df_v3[_df_v3["Zone"].isin(_included_zones)].copy()
    else:
        # Without GMSR zones: all stocks that passed EUMSS + Liquidity filters
        _df_v3_candidates = _df_v3[_df_v3["EUMSS_Pass"] & _df_v3["ADTV_Pass"] & _df_v3["ATVR_Pass"]].copy()
    _map_col_v3 = "Mapping Country" if "Mapping Country" in _df_v3_candidates.columns else "Exchange Country Name"
    _df_v3_step7 = apply_step7_coverage(_df_v3_candidates, coverage_pct=0.85, country_col=_map_col_v3)

    # ── Step 8: Secondary Share Classes ──────────────────────────────────────
    if include_secondary and len(_df_v3_step7) > 0:
        _df_v3_step8 = apply_step8_secondary(
            _df_v3_step7, df_raw_original,
            _v3_eumss_full, _v3_eumss_ff, v3_min_ff_pct,
            v3_atvr_dm_min, v3_atvr_em_min,
            china_if=china_inclusion_factor, india_if=india_inclusion_factor,
            vietnam_if=vietnam_inclusion_factor, saudi_if=saudi_inclusion_factor,
            adtv_dm_3m=dm_min_adtv_3m, adtv_dm_6m=dm_min_adtv_6m,
            adtv_em_3m=em_min_adtv_3m, adtv_em_6m=em_min_adtv_6m,
        )
        _n_secondary = len(_df_v3_step8) - len(_df_v3_step7)
    else:
        _df_v3_step8 = _df_v3_step7.copy()
        _n_secondary = 0

    _df_v3_included = _df_v3_step8.copy()

    # Ensure Adj_FF_MCap column exists (Step 8 stocks already have it, others may not)
    if "Adj_FF_MCap" not in _df_v3_included.columns:
        ecn = _df_v3_included["Exchange Country Name"].fillna("")
        _df_v3_included["Inclusion_Factor_V3"] = np.where(ecn.str.upper() == "CHINA", china_inclusion_factor,
                                                  np.where(ecn.str.upper() == "INDIA", india_inclusion_factor,
                                                  np.where(ecn.str.upper() == "VIETNAM", vietnam_inclusion_factor,
                                                  np.where(ecn.str.upper() == "SAUDI ARABIA", saudi_inclusion_factor, 1.0))))
        _df_v3_included["Adj_FF_MCap"] = _df_v3_included["Free Float MCap Y2025"] * _df_v3_included["Inclusion_Factor_V3"]

    # ── Step 9: Index Weighting ───────────────────────────────────────────────
    _total_adj_v3_idx = _df_v3_included["Adj_FF_MCap"].sum()
    _df_v3_included["Index_Weight"] = (_df_v3_included["Adj_FF_MCap"] / _total_adj_v3_idx * 100) if _total_adj_v3_idx > 0 else 0
    _df_v3_included["FF MCap Weight (%)"] = (_df_v3_included["Free Float MCap Y2025"] / _df_v3_included["Free Float MCap Y2025"].sum() * 100) if _df_v3_included["Free Float MCap Y2025"].sum() > 0 else 0
    _df_v3_included = _df_v3_included.sort_values("Index_Weight", ascending=False).copy()
    _df_v3_included["Rank"] = range(1, len(_df_v3_included) + 1)

    _df_v3_dm = _df_v3_included[_df_v3_included["Classification"] == "DM"]
    _df_v3_em = _df_v3_included[_df_v3_included["Classification"] == "EM"]

    # ── Pipeline Diagnostics (shown after all steps computed) ───────────────────
    with st.expander("🔍 Pipeline Diagnostik — Stocks je Schritt", expanded=True):
        _diag_rows = []
        # Step 1: Exclusions — break down by category
        import re as _re
        _df_orig = df_raw_original[df_raw_original["Free Float MCap Y2025"].fillna(0) > 0].copy()
        _n_orig_total = len(_df_orig)

        # FF MCap <= 0 (fixed universe filter)
        _n_ff_zero = len(df_raw_original) - len(_df_orig)
        _diag_rows.append({"Schritt": "1a — FF MCap ≤ 0 / fehlend", "DM": "—", "EM": "—", "Total": "—", "Exkludiert": f"-{_n_ff_zero:,}"})

        # HK CNY
        _n_hk = (_df_orig["Exchange Ticker"].str.contains("HKG", na=False) & (_df_orig["Trading Currency"] == "CNY")).sum() if exclude_hk_cny else 0
        _diag_rows.append({"Schritt": "1b — HK (CNY)", "DM": "—", "EM": "—", "Total": "—", "Exkludiert": f"-{_n_hk:,}"})

        # Country of Risk = @
        _n_cor = (_df_orig["Country of Risk"].fillna("") == "@NA").sum() if exclude_country_risk_na else 0
        _diag_rows.append({"Schritt": "1c — Country of Risk = @NA", "DM": "—", "EM": "—", "Total": "—", "Exkludiert": f"-{_n_cor:,}"})

        # NAICS = Investment Funds
        _n_naics = _df_orig["NAICS"].fillna("").str.contains("Open-End Investment Fund", case=False, na=False).sum() if exclude_naics_funds else 0
        _diag_rows.append({"Schritt": "1d — NAICS Investment Funds", "DM": "—", "EM": "—", "Total": "—", "Exkludiert": f"-{_n_naics:,}"})

        # Exchange = Euro MTF / @
        _n_euro = _df_orig["Exchange Name"].fillna("").isin(["Euro MTF", "@NA"]).sum() if exclude_euro_mtf else 0
        _diag_rows.append({"Schritt": "1e — Exchange Euro MTF / @NA", "DM": "—", "EM": "—", "Total": "—", "Exkludiert": f"-{_n_euro:,}"})

        # Name ETF / SICAV / %
        _etf_p = _re.compile(r'ETF|SICAV|%', _re.IGNORECASE)
        _n_etf = _df_orig["Name"].fillna("").str.contains(_etf_p).sum() if exclude_etf_sicav else 0
        _diag_rows.append({"Schritt": "1f — Name: ETF / SICAV / %", "DM": "—", "EM": "—", "Total": "—", "Exkludiert": f"-{_n_etf:,}"})

        # Thailand Sec Type
        _th_excl = _df_orig["Exchange Name"].fillna("").str.upper() == "THAILAND"
        _th_type = "NVDR" if thailand_sec_type == "SHARE" else "SHARE"
        _n_thai = (_th_excl & (_df_orig["Sec Type"].fillna("") == _th_type)).sum()
        _diag_rows.append({"Schritt": "1g — Thailand Sec Type", "DM": "—", "EM": "—", "Total": "—", "Exkludiert": f"-{_n_thai:,}"})

        _n_cls_total = len(df_dm_full) + len(df_em_full)
        _diag_rows.append({"Schritt": "2 — DM/EM Klassifikation", "DM": len(df_dm_full), "EM": len(df_em_full), "Total": _n_cls_total, "Exkl. DM": "—", "Exkl. EM": "—", "Exkludiert": "—"})

        _n_eumss_dm = int((_df_v3["EUMSS_Pass"] & (_df_v3["Classification"]=="DM")).sum())
        _n_eumss_em = int((_df_v3["EUMSS_Pass"] & (_df_v3["Classification"]=="EM")).sum())
        _n_eumss_total = _n_eumss_dm + _n_eumss_em
        _diag_rows.append({"Schritt": "3 — EUMSS Filter", "DM": _n_eumss_dm, "EM": _n_eumss_em, "Total": _n_eumss_total, "Exkl. DM": f"-{len(df_dm_full)-_n_eumss_dm:,}", "Exkl. EM": f"-{len(df_em_full)-_n_eumss_em:,}", "Exkludiert": f"-{_n_cls_total - _n_eumss_total:,}"})

        # 4a: ADTV (3M & 6M)
        _n_adtv_dm = int((_df_v3["EUMSS_Pass"] & _df_v3["ADTV_Pass"] & (_df_v3["Classification"]=="DM")).sum())
        _n_adtv_em = int((_df_v3["EUMSS_Pass"] & _df_v3["ADTV_Pass"] & (_df_v3["Classification"]=="EM")).sum())
        _n_adtv_total = _n_adtv_dm + _n_adtv_em
        _diag_rows.append({"Schritt": "4a — Liquidität: ADTV (3M & 6M)", "DM": _n_adtv_dm, "EM": _n_adtv_em, "Total": _n_adtv_total, "Exkl. DM": f"-{_n_eumss_dm-_n_adtv_dm:,}", "Exkl. EM": f"-{_n_eumss_em-_n_adtv_em:,}", "Exkludiert": f"-{_n_eumss_total - _n_adtv_total:,}"})

        # 4b: ATVR
        _n_atvr_dm = int((_df_v3["EUMSS_Pass"] & _df_v3["ADTV_Pass"] & _df_v3["ATVR_Pass"] & (_df_v3["Classification"]=="DM")).sum())
        _n_atvr_em = int((_df_v3["EUMSS_Pass"] & _df_v3["ADTV_Pass"] & _df_v3["ATVR_Pass"] & (_df_v3["Classification"]=="EM")).sum())
        _n_atvr_total = _n_atvr_dm + _n_atvr_em
        _diag_rows.append({"Schritt": "4b — Liquidität: ATVR", "DM": _n_atvr_dm, "EM": _n_atvr_em, "Total": _n_atvr_total, "Exkl. DM": f"-{_n_adtv_dm-_n_atvr_dm:,}", "Exkl. EM": f"-{_n_adtv_em-_n_atvr_em:,}", "Exkludiert": f"-{_n_adtv_total - _n_atvr_total:,}"})

        if use_gmsr_zones:
            _n_zone_dm = int((_df_v3["Zone"].isin(["AUTO_INCLUDE","CANDIDATE"]) & (_df_v3["Classification"]=="DM")).sum())
            _n_zone_em = int((_df_v3["Zone"].isin(["AUTO_INCLUDE","CANDIDATE"]) & (_df_v3["Classification"]=="EM")).sum())
            _n_zone_total = _n_zone_dm + _n_zone_em
            _diag_rows.append({"Schritt": "6 — Zonen (AUTO+CAND)", "DM": _n_zone_dm, "EM": _n_zone_em, "Total": _n_zone_total, "Exkl. DM": f"-{_n_atvr_dm-_n_zone_dm:,}", "Exkl. EM": f"-{_n_atvr_em-_n_zone_em:,}", "Exkludiert": f"-{_n_atvr_total - _n_zone_total:,}"})
        else:
            _n_zone_dm = _n_atvr_dm
            _n_zone_em = _n_atvr_em
            _n_zone_total = _n_atvr_total
            _diag_rows.append({"Schritt": "6 — Zonen (deaktiviert)", "DM": "—", "EM": "—", "Total": "—", "Exkl. DM": "—", "Exkl. EM": "—", "Exkludiert": "—"})

        _n_step7_dm = int((_df_v3_step7["Classification"]=="DM").sum())
        _n_step7_em = int((_df_v3_step7["Classification"]=="EM").sum())
        _n_step7_total = _n_step7_dm + _n_step7_em
        _diag_rows.append({"Schritt": "7 — 85% Coverage", "DM": _n_step7_dm, "EM": _n_step7_em, "Total": _n_step7_total, "Exkl. DM": f"-{_n_zone_dm-_n_step7_dm:,}", "Exkl. EM": f"-{_n_zone_em-_n_step7_em:,}", "Exkludiert": f"-{_n_zone_total - _n_step7_total:,}"})

        _n_final_total = len(_df_v3_included)
        _diag_rows.append({"Schritt": "8 — Secondary (+)", "DM": int(_df_v3_dm.shape[0]), "EM": int(_df_v3_em.shape[0]), "Total": _n_final_total, "Exkl. DM": f"+{int(_df_v3_dm.shape[0])-_n_step7_dm:,}", "Exkl. EM": f"+{int(_df_v3_em.shape[0])-_n_step7_em:,}", "Exkludiert": f"+{_n_final_total - _n_step7_total:,}"})
        st.dataframe(pd.DataFrame(_diag_rows), use_container_width=True, hide_index=True)
        st.caption(f"EUMSS_FULL: {format_bn(_v3_eumss_full)} | EUMSS_FF: {format_bn(_v3_eumss_ff)} | DM GMSR: {format_bn(_v3_dm_gmsr)} | EM GMSR: {format_bn(_v3_em_gmsr)} | ATVR DM ≥ {v3_atvr_dm_min*100:.0f}% | ATVR EM ≥ {v3_atvr_em_min*100:.0f}%")

    _total_adj_v3 = _df_v3_included["Adj_FF_MCap"].sum()
    _em_v3_adj    = _df_v3_em["Adj_FF_MCap"].sum()
    _em_v3_weight = _em_v3_adj / _total_adj_v3 * 100 if _total_adj_v3 > 0 else 0

    # Validation
    _sum_weights = _df_v3_included["Index_Weight"].sum()
    _max_weight  = _df_v3_included["Index_Weight"].max() if len(_df_v3_included) > 0 else 0

    st.markdown(f"**Inkludierte Stocks — Schritte 1–9** (Secondary: {_n_secondary:,})")
    _mc1, _mc2, _mc3, _mc4, _mc5, _mc6 = st.columns(6)
    _mc1.metric("DM Stocks",       f"{len(_df_v3_dm):,}")
    _mc2.metric("EM Stocks",       f"{len(_df_v3_em):,}")
    _mc3.metric("DM FF MCap",      format_bn(_df_v3_dm["Free Float MCap Y2025"].sum()))
    _mc4.metric("EM FF MCap",      format_bn(_df_v3_em["Free Float MCap Y2025"].sum()))
    _mc5.metric("EM Weight (Adj)", f"{_em_v3_weight:.2f}%")
    _mc6.metric("Σ Gewichte",      f"{_sum_weights:.4f}%", delta="✓ OK" if abs(_sum_weights - 100) < 0.001 else "⚠ Prüfen")

    if _max_weight > 10:
        st.warning(f"⚠️ Validierung: Größte Einzelposition = {_max_weight:.2f}% (Grenze: 10%)")

    # Country breakdown (using Adj_FF_MCap / Index_Weight)
    _col_dm3, _col_em3 = st.columns(2)

    with _col_dm3:
        st.markdown("**DM — Country Breakdown**")
        _dm3_c = _df_v3_dm.groupby(_map_col_v3).agg(
            Stocks=("Symbol","count"),
            FF_MCap=("Free Float MCap Y2025","sum"),
            Adj_MCap=("Adj_FF_MCap","sum"),
        ).reset_index().sort_values("Adj_MCap", ascending=False)
        _dm3_c["FF MCap (USD)"]   = _dm3_c["FF_MCap"].apply(format_bn)
        _dm3_c["FF Weight (%)"]   = (_dm3_c["FF_MCap"] / _df_v3_dm["Free Float MCap Y2025"].sum() * 100).round(4)
        _dm3_c["Index Weight (%)"]= (_dm3_c["Adj_MCap"] / _total_adj_v3 * 100).round(4)
        st.dataframe(_dm3_c[[_map_col_v3,"Stocks","FF MCap (USD)","FF Weight (%)","Index Weight (%)"]].rename(
            columns={_map_col_v3:"Land"}), use_container_width=True, hide_index=True)

    with _col_em3:
        st.markdown("**EM — Country Breakdown**")
        _em3_c = _df_v3_em.groupby(_map_col_v3).agg(
            Stocks=("Symbol","count"),
            FF_MCap=("Free Float MCap Y2025","sum"),
            Adj_MCap=("Adj_FF_MCap","sum"),
        ).reset_index().sort_values("Adj_MCap", ascending=False)
        _em3_c["FF MCap (USD)"]   = _em3_c["FF_MCap"].apply(format_bn)
        _em3_c["FF Weight (%)"]   = (_em3_c["FF_MCap"] / _df_v3_em["Free Float MCap Y2025"].sum() * 100).round(4)
        _em3_c["Index Weight (%)"]= (_em3_c["Adj_MCap"] / _total_adj_v3 * 100).round(4)
        st.dataframe(_em3_c[[_map_col_v3,"Stocks","FF MCap (USD)","FF Weight (%)","Index Weight (%)"]].rename(
            columns={_map_col_v3:"Land"}), use_container_width=True, hide_index=True)

    # ── Segment Tables ───────────────────────────────────────────────────────────
    st.markdown("---")
    _v3_col_dm, _v3_col_em = st.columns(2)

    with _v3_col_dm:
        st.markdown("**DM Segments**")
        _v3_dm_seg_rows = []
        _v3_dm_sorted = _df_v3_dm.sort_values("Total MCap Y2025", ascending=False).copy()
        _v3_dm_ff_total = _v3_dm_sorted["Free Float MCap Y2025"].sum()
        if _v3_dm_ff_total > 0:
            _v3_dm_sorted["_cum_pct"] = _v3_dm_sorted["Free Float MCap Y2025"].cumsum() / _v3_dm_ff_total * 100
            _v3_dm_sorted["_seg"] = np.where(_v3_dm_sorted["_cum_pct"] <= large_thr, "Large Cap",
                                    np.where(_v3_dm_sorted["_cum_pct"] <= mid_thr,   "Mid Cap", "Small Cap"))
        else:
            _v3_dm_sorted["_seg"] = "Small Cap"
        for _seg in ["Large Cap", "Mid Cap"]:
            _s = _v3_dm_sorted[_v3_dm_sorted["_seg"] == _seg]
            _v3_dm_seg_rows.append({"Segment": _seg, "# Stocks": len(_s),
                "FF MCap (USD)": format_bn(_s["Free Float MCap Y2025"].sum()),
                "Avg FF MCap (USD)": format_bn(_s["Free Float MCap Y2025"].mean()) if len(_s) > 0 else "—"})
        _v3_dm_seg_rows.append({"Segment": "Total Stocks - World Index", "# Stocks": len(_df_v3_dm),
            "FF MCap (USD)": format_bn(_df_v3_dm["Free Float MCap Y2025"].sum()),
            "Avg FF MCap (USD)": format_bn(_df_v3_dm["Free Float MCap Y2025"].mean()) if len(_df_v3_dm) > 0 else "—"})
        _sc = _v3_dm_sorted[_v3_dm_sorted["_seg"] == "Small Cap"]
        _v3_dm_seg_rows.append({"Segment": "Small Cap", "# Stocks": len(_sc),
            "FF MCap (USD)": format_bn(_sc["Free Float MCap Y2025"].sum()),
            "Avg FF MCap (USD)": format_bn(_sc["Free Float MCap Y2025"].mean()) if len(_sc) > 0 else "—"})
        _v3_dm_seg_rows.append({"Segment": "Total Universe", "# Stocks": len(_df_v3_dm),
            "FF MCap (USD)": format_bn(_df_v3_dm["Free Float MCap Y2025"].sum()),
            "Avg FF MCap (USD)": format_bn(_df_v3_dm["Free Float MCap Y2025"].mean()) if len(_df_v3_dm) > 0 else "—"})
        st.dataframe(style_segment_table(pd.DataFrame(_v3_dm_seg_rows),
            ["Large Cap","Mid Cap","Total Stocks - World Index"]),
            use_container_width=True, hide_index=True)

        # DM Country breakdown
        st.markdown(f"**DM Included ({len(_df_v3_dm):,} Stocks)**")
        _v3_all_dm_c = sorted(_df_v3_dm[_map_col_v3].dropna().unique())
        _v3_dm_ctry = _df_v3_dm.groupby(_map_col_v3).agg(
            Stocks=("Symbol","count"), FF_MCap=("Free Float MCap Y2025","sum"),
            Avg_Total_MCap=("Total MCap Y2025","mean")).reset_index()
        _v3_dm_ctry_full = pd.DataFrame({_map_col_v3: _v3_all_dm_c}).merge(_v3_dm_ctry, on=_map_col_v3, how="left")
        _v3_dm_ctry_full["Stocks"] = _v3_dm_ctry_full["Stocks"].fillna(0).astype(int)
        _v3_dm_ctry_full["FF_MCap"] = _v3_dm_ctry_full["FF_MCap"].fillna(0)
        _v3_dm_ctry_full = _v3_dm_ctry_full.sort_values("FF_MCap", ascending=False)
        _v3_dm_ff_sum = _v3_dm_ctry_full["FF_MCap"].sum()
        _v3_dm_ctry_full["FF MCap (USD)"] = _v3_dm_ctry_full["FF_MCap"].apply(lambda x: format_bn(x) if x > 0 else "—")
        _v3_dm_ctry_full["Avg Total MCap"] = _v3_dm_ctry_full["Avg_Total_MCap"].apply(lambda x: format_bn(x) if x and x > 0 else "—")
        _v3_dm_ctry_full["Weight (%)"] = (_v3_dm_ctry_full["FF_MCap"] / _v3_dm_ff_sum * 100).round(2)
        st.dataframe(_v3_dm_ctry_full[[_map_col_v3,"Stocks","FF MCap (USD)","Avg Total MCap","Weight (%)"]].rename(
            columns={_map_col_v3:"Land"}), use_container_width=True, hide_index=True)

    with _v3_col_em:
        st.markdown("**EM Segments**")
        _v3_em_seg_rows = []
        _v3_em_sorted = _df_v3_em.sort_values("Total MCap Y2025", ascending=False).copy()
        _v3_em_ff_total = _v3_em_sorted["Free Float MCap Y2025"].sum()
        if _v3_em_ff_total > 0:
            _v3_em_sorted["_cum_pct"] = _v3_em_sorted["Free Float MCap Y2025"].cumsum() / _v3_em_ff_total * 100
            _v3_em_sorted["_seg"] = np.where(_v3_em_sorted["_cum_pct"] <= large_thr, "Large Cap", "Mid Cap")
        else:
            _v3_em_sorted["_seg"] = "Mid Cap"
        for _seg in ["Large Cap", "Mid Cap"]:
            _s = _v3_em_sorted[_v3_em_sorted["_seg"] == _seg]
            _v3_em_seg_rows.append({"Segment": _seg, "# Stocks": len(_s),
                "FF MCap (USD)": format_bn(_s["Free Float MCap Y2025"].sum()),
                "Avg FF MCap (USD)": format_bn(_s["Free Float MCap Y2025"].mean()) if len(_s) > 0 else "—"})
        _v3_em_seg_rows.append({"Segment": "Total Stocks - EM", "# Stocks": len(_df_v3_em),
            "FF MCap (USD)": format_bn(_df_v3_em["Free Float MCap Y2025"].sum()),
            "Avg FF MCap (USD)": format_bn(_df_v3_em["Free Float MCap Y2025"].mean()) if len(_df_v3_em) > 0 else "—"})
        _v3_em_seg_rows.append({"Segment": "Small Cap", "# Stocks": "—", "FF MCap (USD)": "—", "Avg FF MCap (USD)": "—"})
        _v3_em_seg_rows.append({"Segment": "Total Universe", "# Stocks": len(_df_v3_em),
            "FF MCap (USD)": format_bn(_df_v3_em["Free Float MCap Y2025"].sum()),
            "Avg FF MCap (USD)": format_bn(_df_v3_em["Free Float MCap Y2025"].mean()) if len(_df_v3_em) > 0 else "—"})
        st.dataframe(style_segment_table(pd.DataFrame(_v3_em_seg_rows),
            ["Large Cap","Mid Cap","Total Stocks - EM"]),
            use_container_width=True, hide_index=True)

        # EM Country breakdown
        st.markdown(f"**EM Included ({len(_df_v3_em):,} Stocks)**")
        _v3_all_em_c = sorted(_df_v3_em[_map_col_v3].dropna().unique())
        _v3_em_ctry = _df_v3_em.groupby(_map_col_v3).agg(
            Stocks=("Symbol","count"), FF_MCap=("Free Float MCap Y2025","sum"),
            Avg_Total_MCap=("Total MCap Y2025","mean")).reset_index()
        _v3_em_ctry_full = pd.DataFrame({_map_col_v3: _v3_all_em_c}).merge(_v3_em_ctry, on=_map_col_v3, how="left")
        _v3_em_ctry_full["Stocks"] = _v3_em_ctry_full["Stocks"].fillna(0).astype(int)
        _v3_em_ctry_full["FF_MCap"] = _v3_em_ctry_full["FF_MCap"].fillna(0)
        _v3_em_ctry_full = _v3_em_ctry_full.sort_values("FF_MCap", ascending=False)
        _v3_em_ff_sum = _v3_em_ctry_full["FF_MCap"].sum()
        _v3_em_ctry_full["FF MCap (USD)"] = _v3_em_ctry_full["FF_MCap"].apply(lambda x: format_bn(x) if x > 0 else "—")
        _v3_em_ctry_full["Avg Total MCap"] = _v3_em_ctry_full["Avg_Total_MCap"].apply(lambda x: format_bn(x) if x and x > 0 else "—")
        _v3_em_ctry_full["Weight (%)"] = (_v3_em_ctry_full["FF_MCap"] / _v3_em_ff_sum * 100).round(2)
        st.dataframe(_v3_em_ctry_full[[_map_col_v3,"Stocks","FF MCap (USD)","Avg Total MCap","Weight (%)"]].rename(
            columns={_map_col_v3:"Land"}), use_container_width=True, hide_index=True)

    # ── Country Charts ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**V3 — Länderübersicht**")
    _v3_country_basis = st.radio("Länder-Basis:", ["Mapping Country","Country of Incorp","Exchange Country Name"],
        index=0, horizontal=True, key="v3_country_basis",
        help="Basis für die Ländergruppierung in den Charts.")
    _v3_cb = _v3_country_basis if _v3_country_basis in _df_v3_included.columns else "Exchange Country Name"

    _v3_all_adj2 = add_adjusted_weight(_df_v3_included, china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)
    _v3_total_stocks = len(_v3_all_adj2)
    _v3_total_adj2 = _v3_all_adj2["Adjusted FF MCap"].sum()

    _v3_cagg = _v3_all_adj2.groupby([_v3_cb,"Classification"]).agg(
        Stocks=("Symbol","count"), Adj_MCap=("Adjusted FF MCap","sum")).reset_index()
    _v3_cagg["Stock %"] = (_v3_cagg["Stocks"] / _v3_total_stocks * 100).round(2)
    _v3_cagg["Adj Weight %"] = (_v3_cagg["Adj_MCap"] / _v3_total_adj2 * 100).round(2)

    _v3_by_stocks = _v3_cagg.groupby(_v3_cb).agg(Stocks=("Stocks","sum"), Stock_Pct=("Stock %","sum")).reset_index().sort_values("Stocks", ascending=False)
    _v3_by_weight = _v3_cagg.groupby(_v3_cb).agg(Adj_MCap=("Adj_MCap","sum"), Weight_Pct=("Adj Weight %","sum")).reset_index().sort_values("Adj_MCap", ascending=False)

    def _v3_add_others(df, val_col, pct_col, n=30):
        top = df.head(n).copy()
        rest = df.iloc[n:]
        if len(rest) > 0:
            top = pd.concat([pd.DataFrame([{_v3_cb: f"Others ({len(rest)} Länder)", val_col: rest[val_col].sum(), pct_col: rest[pct_col].sum()}]), top], ignore_index=True)
        return top.sort_values(val_col, ascending=True)

    _v3_by_stocks_c = _v3_add_others(_v3_by_stocks, "Stocks", "Stock_Pct")
    _v3_by_weight_c = _v3_add_others(_v3_by_weight, "Adj_MCap", "Weight_Pct")

    _vc1, _vc2 = st.columns(2)
    with _vc1:
        st.markdown("**Nach Anzahl Stocks (%)**")
        _fig_v3_stocks = go.Figure(go.Bar(
            x=_v3_by_stocks_c["Stock_Pct"], y=_v3_by_stocks_c[_v3_cb], orientation="h",
            marker_color="#2979ff",
            text=_v3_by_stocks_c["Stock_Pct"].apply(lambda x: f"{x:.2f}%"), textposition="outside"))
        _fig_v3_stocks.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
            height=700, margin=dict(t=10,b=10,l=10,r=60), xaxis=dict(showgrid=False))
        st.plotly_chart(_fig_v3_stocks, use_container_width=True)
    with _vc2:
        st.markdown("**Nach Gewicht (Adj. FF MCap %)**")
        _fig_v3_weight = go.Figure(go.Bar(
            x=_v3_by_weight_c["Weight_Pct"], y=_v3_by_weight_c[_v3_cb], orientation="h",
            marker_color="#ce93d8",
            text=_v3_by_weight_c["Weight_Pct"].apply(lambda x: f"{x:.2f}%"), textposition="outside"))
        _fig_v3_weight.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
            height=700, margin=dict(t=10,b=10,l=10,r=60), xaxis=dict(showgrid=False))
        st.plotly_chart(_fig_v3_weight, use_container_width=True)

    # ── Donut + Inclusion Factor Impact ──────────────────────────────────────
    _v3_donut_data = []
    for _cls in ["DM","EM"]:
        _ff = _df_v3_included[_df_v3_included["Classification"]==_cls]["Free Float MCap Y2025"].sum()
        if _ff > 0: _v3_donut_data.append({"Label": _cls, "FF MCap": _ff, "Type": _cls})
    _v3_donut_df = pd.DataFrame(_v3_donut_data)

    _vd1, _vd2 = st.columns([1,1])
    with _vd1:
        st.markdown("**V3 Composition (DM vs EM)**")
        _fig_v3_donut = px.pie(_v3_donut_df, names="Label", values="FF MCap",
            color="Type", color_discrete_map={"DM":"#2979ff","EM":"#ce93d8"},
            template="plotly_dark", hole=0.45)
        _fig_v3_donut.update_layout(paper_bgcolor="#0f1117", height=350, margin=dict(t=10,b=10))
        st.plotly_chart(_fig_v3_donut, use_container_width=True)

    with _vd2:
        st.markdown("**Inclusion Factor Impact**")
        _v3_if_map = {"China A-Shares":(["SHANGHAI","SHENZHEN"],china_inclusion_factor),
                      "Indien":(["BSE INDIA","NSE INDIA"],india_inclusion_factor),
                      "Vietnam":(["HANOI STOCK EXCHANGE","VIETNAM"],vietnam_inclusion_factor),
                      "Saudi-Arabien":(["SAUDI ARABIA"],saudi_inclusion_factor)}
        _v3_if_rows = []
        _v3_total_ff_raw = _df_v3_included["Free Float MCap Y2025"].sum()
        _v3_if_full = add_adjusted_weight(_df_v3_included, china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)
        _v3_total_adj_if = _v3_if_full["Adjusted FF MCap"].sum()
        for _name, (_exch, _factor) in _v3_if_map.items():
            _mask = _v3_if_full["Exchange Name"].isin(_exch)
            _ff = _v3_if_full.loc[_mask,"Free Float MCap Y2025"].sum()
            _adj = _v3_if_full.loc[_mask,"Adjusted FF MCap"].sum()
            if _ff > 0:
                _v3_if_rows.append({"Land": _name, "Factor": f"{_factor*100:.0f}%",
                    "Weight (vor)": round(_ff/_v3_total_ff_raw*100,4),
                    "Weight (nach)": round(_adj/_v3_total_adj_if*100,4),
                    "Δ": round(_adj/_v3_total_adj_if*100 - _ff/_v3_total_ff_raw*100,4)})
        if _v3_if_rows:
            _v3_if_df = pd.DataFrame(_v3_if_rows)
            _v3_if_df = pd.concat([_v3_if_df, pd.DataFrame([{"Land":"Total Weight","Factor":"—",
                "Weight (vor)": round(_v3_if_df["Weight (vor)"].sum(),4),
                "Weight (nach)": round(_v3_if_df["Weight (nach)"].sum(),4),
                "Δ": round(_v3_if_df["Δ"].sum(),4)}])], ignore_index=True)
            def _style_v3_if(df):
                def row_style(row):
                    if row["Land"] == "Total Weight":
                        return ["background-color: #1a2a4a; font-weight: 600;"] * len(row)
                    return [""] * len(row)
                return df.style.apply(row_style, axis=1)
            st.dataframe(_style_v3_if(_v3_if_df), use_container_width=True, hide_index=True)

    # Download
    _v3_universe = _df_v3.copy()
    _v3_universe = _v3_universe[[c for c in _v3_universe.columns if c not in ["cum_ff","EUMSS_Pass"]]]
    st.download_button(
        "⬇️ Download Variant 3 as Excel",
        data=to_excel_multi({"Universe V3": _v3_universe, "Included": _df_v3_included}),
        file_name="NaroIX_Variant3_Dynamic.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: ACWI (DM + EM)
# ══════════════════════════════════════════════════════════════════════════════
with tab_acwi:
    st.subheader("ACWI — All Country World Index (DM + EM)")

    # Reset EM method to Threshold when switching to Variant 1
    _prev_variant = st.session_state.get("_prev_acwi_variant", "Variant 1 (Global / MSCI)")
    acwi_variant = st.radio("DM Methodik für ACWI:", ["Variant 1 (Global / MSCI)", "Variant 2 (Per-Country / Solactive)"], horizontal=True, key="acwi_variant_radio")
    if _prev_variant == "Variant 2 (Per-Country / Solactive)" and acwi_variant == "Variant 1 (Global / MSCI)":
        st.session_state["em_method_radio"] = "Threshold (% des DM Grenzstocks)"
    st.session_state["_prev_acwi_variant"] = acwi_variant

    if acwi_variant == "Variant 1 (Global / MSCI)":
        df_dm_seg = _seg_dm_v1
    else:
        df_dm_seg = _seg_dm_v2

    # ── EM Method Selection ──────────────────────────────────────────────────
    if acwi_variant == "Variant 2 (Per-Country / Solactive)":
        em_method = "Per-Country 85% (wie DM Variant 2)"
        st.markdown("""<div class="info-box">
        <b>EM Methodik:</b> Per-Country 85% (wie DM Variant 2) &nbsp;—&nbsp;
        automatisch gewählt da DM Variant 2 aktiv ist.</div>""", unsafe_allow_html=True)
    else:
        em_method = st.radio(
            "EM Methodik:",
            [
                "Threshold (% des DM Grenzstocks)",
                "Global 85% (wie DM Variant 1)",
            ],
            horizontal=True,
            key="em_method_radio",
            help="Threshold: Mindest-MCap relativ zum DM-Grenzstock. Global 85%: alle EM global sortiert, Top-85% FF MCap."
        )

    # ── DM Cutoff Stock — always use global V1 cutoff as EM threshold reference ──
    dm_seg_v1_ref = _seg_dm_v1
    cutoff_stock = _cutoff_v1
    cutoff_total_mcap = cutoff_stock["Total MCap Y2025"] if cutoff_stock is not None else 0

    df_acwi_dm = dm_post_filter(df_dm_seg[df_dm_seg["Segment"].isin(["Large Cap", "Mid Cap"])]).copy()

    if em_method == "Threshold (% des DM Grenzstocks)":
        # ── Threshold approach ───────────────────────────────────────────────
        df_em_result, em_min_mcap = filter_em_by_threshold(df_em, cutoff_total_mcap, em_threshold_pct)
        df_acwi_em = em_post_filter(df_em_result[df_em_result["Segment"] == "EM Included"]).copy()

        # Info box
        if cutoff_stock is not None:
            st.markdown(f"""
            <div class="info-box">
            <b>DM Grenzstock (globaler 85%-Cutoff — EM-Referenz):</b>
            {cutoff_stock["Name"]} ({cutoff_stock["Exchange Country Name"]}) &nbsp;|&nbsp;
            Total MCap: <b>{format_bn(cutoff_total_mcap)}</b> &nbsp;|&nbsp;
            cum. FF MCap: <b>{cutoff_stock["cum_pct"]:.3f}%</b><br>
            <b>EM Mindest Total MCap:</b> {em_threshold_pct}% × {format_bn(cutoff_total_mcap)} = <b>{format_bn(em_min_mcap)}</b>
            &nbsp;→&nbsp; <b>{len(df_acwi_em):,} EM-Aktien</b> qualifizieren
            </div>
            """, unsafe_allow_html=True)

        em_col_label = f"EM Included ({len(df_acwi_em):,} Stocks | Total MCap ≥ {format_bn(em_min_mcap)})"
        em_has_segments = False  # Threshold: no Large/Mid/Small, use ranking

        # Sensitivity chart
        st.markdown("**EM Sensitivity — Anzahl qualifizierter EM-Aktien je Threshold**")
        sens_pcts = list(range(10, 210, 10))
        sens_counts = [len(df_em[df_em["Total MCap Y2025"] >= cutoff_total_mcap * p / 100]) for p in sens_pcts]
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=sens_pcts, y=sens_counts, mode="lines+markers",
            line=dict(color="#ce93d8", width=2), marker=dict(size=5),
        ))
        fig_sens.add_vline(x=em_threshold_pct, line_dash="dash", line_color="#ffc107",
                           annotation_text=f"Current: {em_threshold_pct}% → {len(df_acwi_em)} stocks",
                           annotation_position="top right")
        fig_sens.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
            height=260, xaxis_title="EM Threshold (% of DM Cutoff Stock Total MCap)",
            yaxis_title="EM Stocks Included", margin=dict(t=10,b=40,l=60,r=10),
        )
        st.plotly_chart(fig_sens, use_container_width=True)

    elif em_method == "Global 85% (wie DM Variant 1)":
        # ── Global 85% approach (identical logic to DM Variant 1) ───────────
        df_em_result = _seg_em_g85
        df_acwi_em = em_post_filter(df_em_result[df_em_result["Segment"].isin(["Large Cap", "Mid Cap"])]).copy()
        em_min_mcap = None

        em_cutoff = get_dm_cutoff_stock(df_em_result)
        st.markdown(f"""
        <div class="info-box">
        <b>EM Global {mid_thr}%-Regel (wie DM Variant 1):</b> Alle EM-Aktien global nach
        Total MCap sortiert, kumulierte FF MCap gebildet — alle bis {mid_thr}% kommen rein.<br>
        <b>EM Grenzstock:</b> {em_cutoff["Name"] if em_cutoff is not None else "—"}
        ({em_cutoff["Exchange Country Name"] if em_cutoff is not None else "—"}) &nbsp;|&nbsp;
        Total MCap: <b>{format_bn(em_cutoff["Total MCap Y2025"]) if em_cutoff is not None else "—"}</b><br>
        <b>Ergebnis:</b> <b>{len(df_acwi_em):,} EM-Aktien</b> aus {df_acwi_em["Exchange Country Name"].nunique()} Ländern qualifizieren.
        </div>
        """, unsafe_allow_html=True)

        em_col_label = f"EM Included ({len(df_acwi_em):,} Stocks | Global {mid_thr}%)"
        em_has_segments = True  # Global 85%: segments already assigned (Large/Mid only)

    else:
        # ── Per-Country 85% approach ─────────────────────────────────────────
        df_em_result = filter_em_per_country(df_em, pct=mid_thr)
        df_acwi_em = em_post_filter(df_em_result[df_em_result["Segment"] == "EM Included"]).copy()
        em_min_mcap = None

        st.markdown(f"""
        <div class="info-box">
        <b>EM Per-Country {mid_thr}%-Regel:</b> Innerhalb jedes EM-Lands werden Aktien nach
        Total MCap absteigend sortiert und alle bis zur kumulierten FF MCap von {mid_thr}% eingeschlossen —
        analog zu DM Variant 2.<br>
        <b>Ergebnis:</b> <b>{len(df_acwi_em):,} EM-Aktien</b> aus {df_acwi_em["Exchange Country Name"].nunique()} Ländern qualifizieren.
        </div>
        """, unsafe_allow_html=True)

        # Per-country EM breakdown
        with st.expander("📋 EM Grenzstocks je Land"):
            em_cutoffs = []
            for country, grp in df_em_result.groupby("Exchange Country Name"):
                inc = grp[grp["Segment"] == "EM Included"]
                if len(inc) == 0: continue
                last = inc.sort_values("Total MCap Y2025").iloc[-1]
                em_cutoffs.append({
                    "Land": country,
                    "Grenzstock": last["Name"],
                    "# Included": len(inc),
                    "Total MCap": format_bn(last["Total MCap Y2025"]),
                    "cum. FF%": f"{last['cum_pct']:.3f}%"
                })
            st.dataframe(pd.DataFrame(em_cutoffs), use_container_width=True, hide_index=True)

        em_col_label = f"EM Included ({len(df_acwi_em):,} Stocks | Per-Country {mid_thr}%)"
        em_has_segments = True  # Per-Country 85%: segments already assigned (Large/Mid only)

    total_acwi = len(df_acwi_dm) + len(df_acwi_em)
    total_ff_acwi = df_acwi_dm["Free Float MCap Y2025"].sum() + df_acwi_em["Free Float MCap Y2025"].sum()

    # Compute Adjusted EM Weight
    _acwi_adj = add_adjusted_weight(pd.concat([df_acwi_dm, df_acwi_em], ignore_index=True), china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)
    _total_adj_acwi = _acwi_adj["Adjusted FF MCap"].sum()
    _em_adj_weight = _acwi_adj[_acwi_adj["Classification"] == "EM"]["Adjusted FF MCap"].sum() / _total_adj_acwi * 100 if _total_adj_acwi > 0 else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("ACWI Total Stocks", f"{total_acwi:,}")
    c2.metric("ACWI FF MCap", format_bn(total_ff_acwi))
    c3.metric("DM Stocks (World)", f"{len(df_acwi_dm):,}")
    c4.metric("EM Stocks", f"{len(df_acwi_em):,}")
    c5.metric("EM Weight", f"{df_acwi_em['Free Float MCap Y2025'].sum()/total_ff_acwi*100:.2f}%" if total_ff_acwi > 0 else "—")
    c6.metric("Adj. EM Weight", f"{_em_adj_weight:.2f}%")

    col_dm, col_em = st.columns(2)

    with col_dm:
        st.markdown("**DM Segments (World Index)**")
        # Use post-filtered df_acwi_dm for consistent counts
        _dm_seg_rows = []
        for seg in ["Large Cap", "Mid Cap"]:
            _s = dm_post_filter(df_dm_seg[df_dm_seg["Segment"] == seg])
            _dm_seg_rows.append({
                "Segment": seg,
                "# Stocks": len(_s),
                "FF MCap (USD)": format_bn(_s["Free Float MCap Y2025"].sum()),
                "Avg FF MCap (USD)": format_bn(_s["Free Float MCap Y2025"].mean()) if len(_s) > 0 else "—",
            })
        # Subtotal: World Index (Large + Mid)
        _dm_seg_rows.append({
            "Segment": "Total Stocks - World Index",
            "# Stocks": len(df_acwi_dm),
            "FF MCap (USD)": format_bn(df_acwi_dm["Free Float MCap Y2025"].sum()),
            "Avg FF MCap (USD)": format_bn(df_acwi_dm["Free Float MCap Y2025"].mean()) if len(df_acwi_dm) > 0 else "—",
        })
        _dm_sc = dm_post_filter(df_dm_seg[df_dm_seg["Segment"] == "Small Cap"])
        _dm_seg_rows.append({
            "Segment": "Small Cap",
            "# Stocks": len(_dm_sc),
            "FF MCap (USD)": format_bn(_dm_sc["Free Float MCap Y2025"].sum()),
            "Avg FF MCap (USD)": format_bn(_dm_sc["Free Float MCap Y2025"].mean()) if len(_dm_sc) > 0 else "—",
        })
        _dm_all = dm_post_filter(df_dm_seg[df_dm_seg["Segment"].isin(["Large Cap","Mid Cap","Small Cap"])])
        _dm_seg_rows.append({
            "Segment": "Total Universe",
            "# Stocks": len(_dm_all),
            "FF MCap (USD)": format_bn(_dm_all["Free Float MCap Y2025"].sum()),
            "Avg FF MCap (USD)": format_bn(_dm_all["Free Float MCap Y2025"].mean()) if len(_dm_all) > 0 else "—",
        })
        _dm_seg_df = pd.DataFrame(_dm_seg_rows)
        st.dataframe(
            style_segment_table(_dm_seg_df, ["Large Cap", "Mid Cap", "Total Stocks - World Index"]),
            use_container_width=True, hide_index=True
        )

        _dm_cutoff_label = format_bn(cutoff_stock["Total MCap Y2025"]) if cutoff_stock is not None else "—"
        st.markdown(f"**DM Included ({len(df_acwi_dm):,} Stocks | Total MCap ≥ {_dm_cutoff_label})**")
        _all_dm_countries = sorted(df_dm["Mapping Country"].dropna().unique()) if "Mapping Country" in df_dm.columns else sorted(df_dm["Exchange Country Name"].unique())
        _dm_country_col = "Mapping Country" if "Mapping Country" in df_acwi_dm.columns else "Exchange Country Name"
        dm_country_tbl = df_acwi_dm.groupby(_dm_country_col).agg(
            Stocks=("Symbol","count"),
            FF_MCap=("Free Float MCap Y2025","sum"),
            Avg_Total_MCap=("Total MCap Y2025","mean"),
        ).reset_index().rename(columns={_dm_country_col: "Exchange Country Name"})
        _dm_all_c = pd.DataFrame({"Exchange Country Name": _all_dm_countries})
        dm_country_tbl = _dm_all_c.merge(dm_country_tbl, on="Exchange Country Name", how="left")
        dm_country_tbl["Stocks"] = dm_country_tbl["Stocks"].fillna(0).astype(int)
        dm_country_tbl["FF_MCap"] = dm_country_tbl["FF_MCap"].fillna(0)
        dm_country_tbl["Avg_Total_MCap"] = dm_country_tbl["Avg_Total_MCap"].fillna(0)
        dm_country_tbl = dm_country_tbl.sort_values("FF_MCap", ascending=False)
        _dm_ff_sum = dm_country_tbl["FF_MCap"].sum()
        dm_country_tbl["FF MCap (USD)"] = dm_country_tbl["FF_MCap"].apply(lambda x: format_bn(x) if x > 0 else "—")
        dm_country_tbl["Avg Total MCap"] = dm_country_tbl["Avg_Total_MCap"].apply(lambda x: format_bn(x) if x > 0 else "—")
        dm_country_tbl["Weight (%)"] = dm_country_tbl["FF_MCap"].apply(lambda x: round(x / _dm_ff_sum * 100, 2) if _dm_ff_sum > 0 else 0)
        st.dataframe(dm_country_tbl[["Exchange Country Name","Stocks","FF MCap (USD)","Avg Total MCap","Weight (%)"]],
                     use_container_width=True, hide_index=True)

    with col_em:
        st.markdown(f"**EM Segments**")
        if em_has_segments:
            # Global 85%: segments are Large Cap / Mid Cap directly from compute_variant1
            # Per-Country 85%: apply compute_variant2 logic to EM (per-country Large/Mid)
            _has_lc = "Large Cap" in df_acwi_em["Segment"].values if "Segment" in df_acwi_em.columns else False
            if _has_lc:
                # Global 85% — segments already correctly set by compute_variant1
                _em_seg_assigned = df_acwi_em.copy()
                _em_seg_assigned["EM Segment"] = _em_seg_assigned["Segment"]
            else:
                # Per-Country 85% — apply per-country Large/Mid segmentation (like compute_variant2)
                _em_pc_segs = []
                for _country, _grp in df_acwi_em.groupby("Exchange Country Name"):
                    _grp = _grp.sort_values("Total MCap Y2025", ascending=False).copy()
                    _total_ff_c = _grp["Free Float MCap Y2025"].sum()
                    if _total_ff_c > 0:
                        _grp["cum_pct_c"] = _grp["Free Float MCap Y2025"].cumsum() / _total_ff_c * 100
                        _grp["EM Segment"] = np.where(_grp["cum_pct_c"] <= large_thr, "Large Cap", "Mid Cap")
                    else:
                        _grp["EM Segment"] = "Mid Cap"
                    _em_pc_segs.append(_grp)
                _em_seg_assigned = pd.concat(_em_pc_segs, ignore_index=True)
            _em_seg_rows = []
            for seg in ["Large Cap", "Mid Cap"]:
                _s = _em_seg_assigned[_em_seg_assigned["EM Segment"] == seg]
                _em_seg_rows.append({
                    "Segment": seg,
                    "# Stocks": len(_s),
                    "FF MCap (USD)": format_bn(_s["Free Float MCap Y2025"].sum()) if len(_s) > 0 else "—",
                    "Avg FF MCap (USD)": format_bn(_s["Free Float MCap Y2025"].mean()) if len(_s) > 0 else "—",
                })
            _em_lm2 = _em_seg_assigned[_em_seg_assigned["EM Segment"].isin(["Large Cap","Mid Cap"])]
            _em_seg_rows.append({
                "Segment": "Total Stocks - EM",
                "# Stocks": len(_em_lm2),
                "FF MCap (USD)": format_bn(_em_lm2["Free Float MCap Y2025"].sum()),
                "Avg FF MCap (USD)": format_bn(_em_lm2["Free Float MCap Y2025"].mean()) if len(_em_lm2) > 0 else "—",
            })
            _em_seg_rows.append({
                "Segment": "Small Cap",
                "# Stocks": "—",
                "FF MCap (USD)": "—",
                "Avg FF MCap (USD)": "—",
            })
            _em_seg_rows.append({
                "Segment": "Total Universe",
                "# Stocks": len(df_em),
                "FF MCap (USD)": format_bn(df_em["Free Float MCap Y2025"].sum()),
                "Avg FF MCap (USD)": format_bn(df_em["Free Float MCap Y2025"].mean()) if len(df_em) > 0 else "—",
            })
        else:
            # Threshold method — two separate calculations:
            # Rechnung 1: stocks that pass threshold → rank by cum FF MCap → Large (≤70%) / Mid (≤85%)
            _em_above = df_acwi_em.sort_values("Total MCap Y2025", ascending=False).copy()
            _em_above_ff = _em_above["Free Float MCap Y2025"].sum()
            if _em_above_ff > 0:
                _em_above["cum_pct"] = _em_above["Free Float MCap Y2025"].cumsum() / _em_above_ff * 100
                _em_above["EM Segment"] = _em_above["cum_pct"].apply(
                    lambda x: "Large Cap" if x <= large_thr else "Mid Cap"
                )
            else:
                _em_above["EM Segment"] = "Mid Cap"

            # Rechnung 2: stocks that do NOT pass threshold → rank → Small Cap (≤99%), rest excluded
            _em_full = df_em_result.copy()
            _em_below = _em_full[_em_full["Segment"] != "EM Included"].sort_values("Total MCap Y2025", ascending=False).copy()
            _em_below_ff = _em_below["Free Float MCap Y2025"].sum()
            if _em_below_ff > 0:
                _em_below["cum_pct"] = _em_below["Free Float MCap Y2025"].cumsum() / _em_below_ff * 100
                _em_below["EM Segment"] = _em_below["cum_pct"].apply(
                    lambda x: "Small Cap" if x <= small_thr else "Excluded"
                )
            else:
                _em_below["EM Segment"] = "Excluded"
            _em_sc = _em_below[_em_below["EM Segment"] == "Small Cap"]

            _em_seg_rows = []
            for seg in ["Large Cap", "Mid Cap"]:
                _s = _em_above[_em_above["EM Segment"] == seg]
                _em_seg_rows.append({
                    "Segment": seg,
                    "# Stocks": len(_s),
                    "FF MCap (USD)": format_bn(_s["Free Float MCap Y2025"].sum()),
                    "Avg FF MCap (USD)": format_bn(_s["Free Float MCap Y2025"].mean()) if len(_s) > 0 else "—",
                })
            _em_seg_rows.append({
                "Segment": "Total Stocks - EM",
                "# Stocks": len(_em_above),
                "FF MCap (USD)": format_bn(_em_above["Free Float MCap Y2025"].sum()),
                "Avg FF MCap (USD)": format_bn(_em_above["Free Float MCap Y2025"].mean()) if len(_em_above) > 0 else "—",
            })
            _em_seg_rows.append({
                "Segment": "Small Cap",
                "# Stocks": len(_em_sc),
                "FF MCap (USD)": format_bn(_em_sc["Free Float MCap Y2025"].sum()),
                "Avg FF MCap (USD)": format_bn(_em_sc["Free Float MCap Y2025"].mean()) if len(_em_sc) > 0 else "—",
            })
        # Total Universe only added for Threshold method (already added inside em_has_segments block)
        if not em_has_segments:
            _em_seg_rows.append({
                "Segment": "Total Universe",
                "# Stocks": len(df_em),
                "FF MCap (USD)": format_bn(df_em["Free Float MCap Y2025"].sum()),
                "Avg FF MCap (USD)": format_bn(df_em["Free Float MCap Y2025"].mean()) if len(df_em) > 0 else "—",
            })
        _em_seg_df = pd.DataFrame(_em_seg_rows)
        st.dataframe(
            style_segment_table(_em_seg_df, ["Large Cap", "Mid Cap", "Total Stocks - EM"]),
            use_container_width=True, hide_index=True
        )

        st.markdown(f"**{em_col_label}**")
        _all_em_countries = sorted(df_em["Mapping Country"].dropna().unique()) if "Mapping Country" in df_em.columns else sorted(df_em["Exchange Country Name"].unique())
        _em_country_col = "Mapping Country" if "Mapping Country" in df_acwi_em.columns else "Exchange Country Name"
        em_country = df_acwi_em.groupby(_em_country_col).agg(
            Stocks=("Symbol","count"),
            FF_MCap=("Free Float MCap Y2025","sum"),
            Avg_Total_MCap=("Total MCap Y2025","mean"),
        ).reset_index().rename(columns={_em_country_col: "Exchange Country Name"})
        _em_all_c = pd.DataFrame({"Exchange Country Name": _all_em_countries})
        em_country = _em_all_c.merge(em_country, on="Exchange Country Name", how="left")
        em_country["Stocks"] = em_country["Stocks"].fillna(0).astype(int)
        em_country["FF_MCap"] = em_country["FF_MCap"].fillna(0)
        em_country["Avg_Total_MCap"] = em_country["Avg_Total_MCap"].fillna(0)
        em_country = em_country.sort_values("FF_MCap", ascending=False)
        _em_ff_sum = em_country["FF_MCap"].sum()
        em_country["FF MCap (USD)"] = em_country["FF_MCap"].apply(lambda x: format_bn(x) if x > 0 else "—")
        em_country["Avg Total MCap"] = em_country["Avg_Total_MCap"].apply(lambda x: format_bn(x) if x > 0 else "—")
        em_country["Weight (%)"] = em_country["FF_MCap"].apply(lambda x: round(x / _em_ff_sum * 100, 2) if _em_ff_sum > 0 else 0)
        st.dataframe(em_country[["Exchange Country Name","Stocks","FF MCap (USD)","Avg Total MCap","Weight (%)"]],
                     use_container_width=True, hide_index=True)

    # ── Country Charts ───────────────────────────────────────────────────────
    st.markdown("**ACWI — Länderübersicht**")

    # Country basis selector
    _available_bases = ["Mapping Country", "Country of Incorp", "Exchange Country Name"]
    _available_bases = [b for b in _available_bases if b in pd.concat([df_acwi_dm, df_acwi_em], ignore_index=True).columns]
    _country_basis = st.radio("Länder-Basis:", _available_bases, index=0,
        horizontal=True, key="country_basis_radio",
        help="Mapping Country: Exchange wenn gleich Incorp, sonst Country of Risk. Country of Incorp: Gründungsland. Exchange Country Name: Börsenland.")

    _acwi_all = pd.concat([df_acwi_dm, df_acwi_em], ignore_index=True)
    _total_stocks = len(_acwi_all)

    # Apply inclusion factor for Adjusted Weight
    _acwi_all_adj = add_adjusted_weight(_acwi_all, china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)

    _country_col = _country_basis if _country_basis in _acwi_all_adj.columns else "Exchange Country Name"
    _country_agg = _acwi_all_adj.groupby([_country_col,"Classification"]).agg(
        Stocks=("Symbol","count"),
        Adj_MCap=("Adjusted FF MCap","sum"),
    ).reset_index()
    _country_agg["Stock %"] = (_country_agg["Stocks"] / _total_stocks * 100).round(2)
    _total_adj = _acwi_all_adj["Adjusted FF MCap"].sum()
    _country_agg["Adj Weight %"] = (_country_agg["Adj_MCap"] / _total_adj * 100).round(2)

    _by_stocks_full = _country_agg.groupby(_country_col).agg(
        Stocks=("Stocks","sum"), Stock_Pct=("Stock %","sum")
    ).reset_index().rename(columns={_country_col:"Exchange Country Name"}).sort_values("Stocks", ascending=False)

    _by_weight_full = _country_agg.groupby(_country_col).agg(
        Adj_MCap=("Adj_MCap","sum"), Weight_Pct=("Adj Weight %","sum")
    ).reset_index().rename(columns={_country_col:"Exchange Country Name"}).sort_values("Adj_MCap", ascending=False)

    # Top 30 + Others
    def _add_others(df, val_col, pct_col, n=30):
        top = df.head(n).copy()
        rest = df.iloc[n:]
        if len(rest) > 0:
            others = pd.DataFrame([{
                "Exchange Country Name": f"Others ({len(rest)} Länder)",
                val_col: rest[val_col].sum(),
                pct_col: rest[pct_col].sum(),
            }])
            top = pd.concat([others, top], ignore_index=True)
        return top.sort_values(val_col, ascending=True)

    _by_stocks = _add_others(_by_stocks_full, "Stocks", "Stock_Pct")
    _by_weight = _add_others(_by_weight_full, "Adj_MCap", "Weight_Pct")

    _col1, _col2 = st.columns(2)

    with _col1:
        st.markdown("**Nach Anzahl Stocks (%)**")
        fig_stocks = go.Figure(go.Bar(
            x=_by_stocks["Stock_Pct"],
            y=_by_stocks["Exchange Country Name"],
            orientation="h",
            marker_color="#2979ff",
            text=_by_stocks["Stock_Pct"].apply(lambda x: f"{x:.2f}%"),
            textposition="outside",
        ))
        fig_stocks.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
            height=700, margin=dict(t=10,b=10,l=10,r=60),
            xaxis_title="% Anzahl", yaxis_title=None,
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_stocks, use_container_width=True)

    with _col2:
        st.markdown("**Nach Gewicht (FF MCap %)**")
        fig_weight = go.Figure(go.Bar(
            x=_by_weight["Weight_Pct"],
            y=_by_weight["Exchange Country Name"],
            orientation="h",
            marker_color="#ce93d8",
            text=_by_weight["Weight_Pct"].apply(lambda x: f"{x:.2f}%"),
            textposition="outside",
        ))
        fig_weight.update_layout(
            template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
            height=700, margin=dict(t=10,b=10,l=10,r=60),
            xaxis_title="% Gewicht", yaxis_title=None,
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_weight, use_container_width=True)

    # ── Combined Donut ───────────────────────────────────────────────────────
    st.markdown("**ACWI Composition (DM vs EM)**")
    acwi_summary = []
    for seg in ["Large Cap", "Mid Cap"]:
        dm_ff = df_acwi_dm[df_acwi_dm["Segment"]==seg]["Free Float MCap Y2025"].sum()
        if dm_ff > 0: acwi_summary.append({"Label": f"DM {seg}", "FF MCap": dm_ff, "Type": "DM"})
    em_ff_total = df_acwi_em["Free Float MCap Y2025"].sum()
    if em_ff_total > 0: acwi_summary.append({"Label": "EM Included", "FF MCap": em_ff_total, "Type": "EM"})
    acwi_df_summary = pd.DataFrame(acwi_summary)

    _col_pie, _col_if = st.columns([1, 1])
    with _col_pie:
        fig_acwi = px.pie(
            acwi_df_summary, names="Label", values="FF MCap",
            color="Type", color_discrete_map={"DM": "#2979ff", "EM": "#ce93d8"},
            template="plotly_dark", hole=0.45,
        )
        fig_acwi.update_layout(paper_bgcolor="#0f1117", height=350, margin=dict(t=10,b=10))
        st.plotly_chart(fig_acwi, use_container_width=True)

    with _col_if:
        st.markdown("**Inclusion Factor Impact**")
        # Build full ACWI with adjusted weights
        _acwi_full = pd.concat([df_acwi_dm, df_acwi_em], ignore_index=True)
        _acwi_full_adj = add_adjusted_weight(_acwi_full, china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)
        _total_adj = _acwi_full_adj["Adjusted FF MCap"].sum()

        # Map country names to their Mapping Country / Country of Incorp / Exchange Country Name values
        # based on the active Länder-Basis toggle
        _if_country_map = {
            "China A-Shares": "CHINA",
            "Indien":         "INDIA",
            "Vietnam":        "VIETNAM",
            "Saudi-Arabien":  "SAUDI ARABIA",
        }
        _if_factor_map = {
            "China A-Shares": china_inclusion_factor,
            "Indien":         india_inclusion_factor,
            "Vietnam":        vietnam_inclusion_factor,
            "Saudi-Arabien":  saudi_inclusion_factor,
        }

        # Use same country basis as the chart toggle
        _if_basis = _country_basis if "_country_basis" in dir() else "Mapping Country"
        _if_basis_col = _if_basis if _if_basis in _acwi_full_adj.columns else "Exchange Country Name"

        _if_rows = []
        for name, country_val in _if_country_map.items():
            factor = _if_factor_map[name]
            _mask = _acwi_full_adj[_if_basis_col].fillna("").str.upper() == country_val
            _ff = _acwi_full_adj.loc[_mask, "Free Float MCap Y2025"].sum()
            _adj_sum = _acwi_full_adj.loc[_mask, "Adjusted FF MCap"].sum()
            _total_ff_raw = _acwi_full["Free Float MCap Y2025"].sum()
            # Weight vor: raw FF MCap / total raw FF MCap (without any inclusion factors)
            _acwi_no_if = _acwi_full.copy()
            _acwi_no_if["_raw_weight"] = _acwi_no_if["Free Float MCap Y2025"] / _total_ff_raw * 100
            _w_before = _acwi_no_if.loc[_mask, "_raw_weight"].sum()
            # Weight nach: adjusted FF MCap / total adjusted FF MCap
            _w_after = _adj_sum / _total_adj * 100 if _total_adj > 0 else 0
            _if_rows.append({
                "Land": name,
                "Factor": f"{factor*100:.0f}%",
                "Weight (vor)": round(_w_before, 4),
                "Weight (nach)": round(_w_after, 4),
                "Δ": round(_w_after - _w_before, 4),
            })

        if _if_rows:
            _if_df = pd.DataFrame(_if_rows)
            # Total row
            _total_row = pd.DataFrame([{
                "Land": "Total Weight",
                "Factor": "—",
                "Weight (vor)": round(_if_df["Weight (vor)"].sum(), 4),
                "Weight (nach)": round(_if_df["Weight (nach)"].sum(), 4),
                "Δ": round(_if_df["Δ"].sum(), 4),
            }])
            _if_df = pd.concat([_if_df, _total_row], ignore_index=True)
            def _style_if_table(df, highlight_vals, col):
                def row_style(row):
                    if row[col] in highlight_vals:
                        return ["background-color: #1a2a4a; font-weight: 600;"] * len(row)
                    return [""] * len(row)
                return df.style.apply(row_style, axis=1)
            st.dataframe(
                _style_if_table(_if_df, ["Total Weight"], "Land"),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("Keine aktiven Inclusion Factors.")

    # ── Download ─────────────────────────────────────────────────────────────
    _fp_dm_acwi = dict(ff_gt0=dm_ff_gt0, min_ff_pct=min_ff_pct, max_closing_price=max_closing_price,
        min_adtv_1m=dm_min_adtv_1m, min_adtv_3m=dm_min_adtv_3m,
        min_adtv_6m=dm_min_adtv_6m, min_adtv_12m=dm_min_adtv_12m, min_liq_ratio=liq_ratio_min)
    _fp_em_acwi = dict(ff_gt0=em_ff_gt0, min_ff_pct=min_ff_pct, max_closing_price=max_closing_price,
        min_adtv_1m=em_min_adtv_1m, min_adtv_3m=em_min_adtv_3m,
        min_adtv_6m=em_min_adtv_6m, min_adtv_12m=em_min_adtv_12m, min_liq_ratio=liq_ratio_min)
    export_dm_acwi = build_full_export(df_dm_seg, "DM", dm_post_filter, _fp_dm_acwi)
    export_em_acwi = build_full_export(df_em_result, "EM", em_post_filter, _fp_em_acwi)
    _acwi_universe = pd.concat([export_dm_acwi, export_em_acwi], ignore_index=True)
    _acwi_universe = _acwi_universe[[c for c in _acwi_universe.columns if c not in ["cum_ff"]]]
    _acwi_world = add_adjusted_weight(add_ff_weight(_acwi_universe[
        (_acwi_universe["Classification"] == "DM") &
        (_acwi_universe["Segment"].isin(["Large Cap","Mid Cap"])) &
        (_acwi_universe["Status"] == "Included")
    ]), china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)
    _acwi_index = add_adjusted_weight(add_ff_weight(_acwi_universe[
        (_acwi_universe["Segment"].isin(["Large Cap","Mid Cap","EM Included"])) &
        (_acwi_universe["Status"] == "Included")
    ]), china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)
    st.download_button(
        "⬇️ Download ACWI as Excel",
        data=to_excel_multi({
            "Universe":         _acwi_universe,
            "World Index (DM)": _acwi_world,
            "ACWI Index":       _acwi_index,
        }),
        file_name="NaroIX_ACWI.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("⚖️ Variant 1 vs. Variant 2 — Direct Comparison")

    df_v1_c = _seg_dm_v1
    df_v2_c = _seg_dm_v2

    world_v1 = dm_post_filter(df_v1_c[df_v1_c["Segment"].isin(["Large Cap","Mid Cap"])])
    world_v2 = dm_post_filter(df_v2_c[df_v2_c["Segment"].isin(["Large Cap","Mid Cap"])])

    # Overlap analysis
    set_v1 = set(world_v1["Symbol"].unique())
    set_v2 = set(world_v2["Symbol"].unique())
    only_v1 = set_v1 - set_v2
    only_v2 = set_v2 - set_v1
    both    = set_v1 & set_v2

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("V1 World Stocks", f"{len(set_v1):,}")
    col2.metric("V2 World Stocks", f"{len(set_v2):,}")
    col3.metric("In Both", f"{len(both):,}")
    col4.metric("Overlap %", f"{len(both)/max(len(set_v1|set_v2),1)*100:.1f}%")

    st.markdown(f"""
    <div class="info-box">
    <b>Overlap:</b> {len(both):,} Stocks in beiden Varianten &nbsp;|&nbsp;
    <b>Nur in V1:</b> {len(only_v1):,} Stocks &nbsp;|&nbsp;
    <b>Nur in V2:</b> {len(only_v2):,} Stocks
    </div>
    """, unsafe_allow_html=True)

    # Per-country stock count comparison
    st.markdown("**Stock Count per Country — V1 vs V2**")
    v1_country = world_v1.groupby("Exchange Country Name")["Symbol"].count().rename("V1 Stocks")
    v2_country = world_v2.groupby("Exchange Country Name")["Symbol"].count().rename("V2 Stocks")
    cmp = pd.concat([v1_country, v2_country], axis=1).fillna(0).astype(int)
    cmp["Δ (V2−V1)"] = cmp["V2 Stocks"] - cmp["V1 Stocks"]
    cmp = cmp.sort_values("V1 Stocks", ascending=False).reset_index()

    fig_cmp = go.Figure()
    fig_cmp.add_bar(x=cmp["Exchange Country Name"], y=cmp["V1 Stocks"], name="Variant 1", marker_color="#2979ff")
    fig_cmp.add_bar(x=cmp["Exchange Country Name"], y=cmp["V2 Stocks"], name="Variant 2", marker_color="#00e676")
    fig_cmp.update_layout(
        barmode="group", template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
        height=380, xaxis_tickangle=-45, margin=dict(t=10,b=100,l=60,r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.dataframe(cmp, use_container_width=True, hide_index=True)

    # Only-in-V1 and Only-in-V2
    if only_v1 or only_v2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Stocks only in Variant 1 ({len(only_v1)})**")
            if only_v1:
                df_only1 = df_v1_c[df_v1_c["Symbol"].isin(only_v1)][
                    ["Symbol","Name","Exchange Country Name","Segment","Free Float MCap Y2025"]
                ].copy()
                df_only1["Free Float MCap Y2025"] = df_only1["Free Float MCap Y2025"].apply(format_bn)
                st.dataframe(df_only1, use_container_width=True, hide_index=True, height=300)

        with col_b:
            st.markdown(f"**Stocks only in Variant 2 ({len(only_v2)})**")
            if only_v2:
                df_only2 = df_v2_c[df_v2_c["Symbol"].isin(only_v2)][
                    ["Symbol","Name","Exchange Country Name","Segment","Free Float MCap Y2025"]
                ].copy()
                df_only2["Free Float MCap Y2025"] = df_only2["Free Float MCap Y2025"].apply(format_bn)
                st.dataframe(df_only2, use_container_width=True, hide_index=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: ACWI COMPARISON — V1 vs V2
# ══════════════════════════════════════════════════════════════════════════════
with tab_acwi_compare:
    st.subheader("ACWI Comparison — Variant 1 vs Variant 2 (DM)")

    # ── Compute DM both variants ─────────────────────────────────────────────
    acwi_dm_v1 = _seg_dm_v1
    acwi_dm_v2 = _seg_dm_v2
    dm_world_v1 = dm_post_filter(acwi_dm_v1[acwi_dm_v1["Segment"].isin(["Large Cap","Mid Cap"])])
    dm_world_v2 = dm_post_filter(acwi_dm_v2[acwi_dm_v2["Segment"].isin(["Large Cap","Mid Cap"])])

    # ── EM: compute both methods simultaneously (same for V1 & V2) ───────────
    cutoff_ref = _cutoff_v1
    cutoff_mcap_ref = cutoff_ref["Total MCap Y2025"] if cutoff_ref is not None else 0
    em_thr_df, em_min_mcap_cmp = filter_em_by_threshold(df_em, cutoff_mcap_ref, em_threshold_pct)
    em_g85_df = _seg_em_g85
    em_pc_df  = filter_em_per_country(df_em, pct=mid_thr)
    em_thr = em_post_filter(em_thr_df[em_thr_df["Segment"] == "EM Included"])
    em_g85 = em_post_filter(em_g85_df[em_g85_df["Segment"].isin(["Large Cap","Mid Cap"])])
    em_pc  = em_post_filter(em_pc_df[em_pc_df["Segment"] == "EM Included"])

    # ── EM Info Box ──────────────────────────────────────────────────────────
    if cutoff_ref is not None:
        st.markdown(f"""
        <div class="info-box">
        <b>EM ist identisch für DM V1 & V2</b><br>
        EM Threshold ({em_threshold_pct}% von {format_bn(cutoff_mcap_ref)}): <b>{len(em_thr):,} Aktien</b> &nbsp;|&nbsp;
        EM Global {mid_thr}%: <b>{len(em_g85):,} Aktien</b> &nbsp;|&nbsp;
        EM Per-Country {mid_thr}%: <b>{len(em_pc):,} Aktien</b>
        </div>""", unsafe_allow_html=True)

    # Use Threshold EM for the combined ACWI totals (primary method)
    acwi_v1_thr = pd.concat([dm_world_v1, em_thr], ignore_index=True)
    acwi_v2_thr = pd.concat([dm_world_v2, em_thr], ignore_index=True)
    acwi_v1_pc  = pd.concat([dm_world_v1, em_pc],  ignore_index=True)
    acwi_v2_pc  = pd.concat([dm_world_v2, em_pc],  ignore_index=True)

    # ── Top-Level Metrics ────────────────────────────────────────────────────
    st.markdown("**ACWI Top-Level Vergleich**")
    st.caption("Gesamt = DM (V1 oder V2) + EM. EM-Spalten sind für beide DM-Varianten identisch.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("V1 DM Stocks",      f"{len(dm_world_v1):,}")
    col2.metric("V2 DM Stocks",      f"{len(dm_world_v2):,}", delta=f"{len(dm_world_v2)-len(dm_world_v1):+,}")
    col3.metric("EM Threshold",      f"{len(em_thr):,}")
    col4.metric("EM Global 85%",     f"{len(em_g85):,}")
    col5.metric("EM Per-Country 85%",f"{len(em_pc):,}")

    # Use threshold-based EM for weight comparison (representative)
    ff_acwi_v1 = acwi_v1_thr["Free Float MCap Y2025"].sum()
    ff_acwi_v2 = acwi_v2_thr["Free Float MCap Y2025"].sum()
    em_inc_v1 = em_thr  # for downstream charts
    em_inc_v2 = em_thr

    # ── DM/EM Weight Comparison (Adjusted Weights) ──────────────────────────
    st.markdown("**DM vs EM Gewichtung (Adjusted Weights)**")

    def _adj_ff(df):
        return add_adjusted_weight(df, china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)["Adjusted FF MCap"].sum()

    _em_methods = {
        f"EM Threshold ({em_threshold_pct}%)": em_thr,
        f"EM Global {mid_thr}%": em_g85,
        f"EM Per-Country {mid_thr}%": em_pc,
    }
    # Only meaningful combinations
    _valid_combos = [
        (f"V1 (MSCI) + EM Threshold ({em_threshold_pct}%)", dm_world_v1, em_thr),
        (f"V1 (MSCI) + EM Global {mid_thr}%",               dm_world_v1, em_g85),
        (f"V2 (Solactive) + EM Per-Country {mid_thr}%",     dm_world_v2, em_pc),
    ]
    _weight_rows = []
    for label, dm_df, em_df in _valid_combos:
        _dm_adj = _adj_ff(dm_df)
        _em_adj = _adj_ff(em_df)
        _total  = _dm_adj + _em_adj
        if _total > 0:
            _weight_rows.append({
                "Variante": label,
                "DM Weight (%)": round(_dm_adj / _total * 100, 2),
                "EM Weight (%)": round(_em_adj / _total * 100, 2),
            })

    _weight_df = pd.DataFrame(_weight_rows)
    fig_weights = go.Figure()
    fig_weights.add_bar(x=_weight_df["Variante"], y=_weight_df["DM Weight (%)"],
                        name="DM", marker_color="#2979ff",
                        text=_weight_df["DM Weight (%)"].apply(lambda x: f"{x:.2f}%"),
                        textposition="inside")
    fig_weights.add_bar(x=_weight_df["Variante"], y=_weight_df["EM Weight (%)"],
                        name="EM", marker_color="#ce93d8",
                        text=_weight_df["EM Weight (%)"].apply(lambda x: f"{x:.2f}%"),
                        textposition="inside")
    fig_weights.update_layout(
        barmode="stack", template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
        height=360, margin=dict(t=10,b=120,l=60,r=10),
        
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_weights, use_container_width=True)

    # ── Per-Country ACWI Breakdown ───────────────────────────────────────────
    st.markdown("**ACWI Stocks per Country — V1 vs V2**")
    v1_c = acwi_v1_thr.groupby(["Exchange Country Name","Classification"])["Symbol"].count().reset_index()
    v2_c = acwi_v2_thr.groupby(["Exchange Country Name","Classification"])["Symbol"].count().reset_index()
    v1_c.columns = ["Exchange Country Name","Classification","V1 Stocks"]
    v2_c.columns = ["Exchange Country Name","Classification","V2 Stocks"]
    cmp_country = v1_c.merge(v2_c, on=["Exchange Country Name","Classification"], how="outer").fillna(0)
    cmp_country[["V1 Stocks","V2 Stocks"]] = cmp_country[["V1 Stocks","V2 Stocks"]].astype(int)
    cmp_country["Δ (V2−V1)"] = cmp_country["V2 Stocks"] - cmp_country["V1 Stocks"]
    cmp_country = cmp_country.sort_values(["Classification","V1 Stocks"], ascending=[True,False])

    fig_country = go.Figure()
    for cls, col in [("DM","#2979ff"),("EM","#ce93d8")]:
        sub = cmp_country[cmp_country["Classification"]==cls]
        fig_country.add_bar(x=sub["Exchange Country Name"], y=sub["V1 Stocks"],
                            name=f"{cls} V1", marker_color=col, opacity=0.9)
        fig_country.add_bar(x=sub["Exchange Country Name"], y=sub["V2 Stocks"],
                            name=f"{cls} V2", marker_color=col, opacity=0.5,
                            marker_pattern_shape="/")
    fig_country.update_layout(
        barmode="group", template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
        height=400, xaxis_tickangle=-45, margin=dict(t=10,b=110,l=60,r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_country, use_container_width=True)

    # ── DM Overlap (V1 vs V2) ────────────────────────────────────────────────
    st.markdown("**DM Stock Overlap (Variant 1 vs Variant 2)**")
    set_dm_v1 = set(dm_world_v1["Symbol"])
    set_dm_v2 = set(dm_world_v2["Symbol"])
    only_dm_v1 = set_dm_v1 - set_dm_v2
    only_dm_v2 = set_dm_v2 - set_dm_v1
    both_dm    = set_dm_v1 & set_dm_v2

    ca, cb, cc, cd = st.columns(4)
    ca.metric("V1 DM only", f"{len(only_dm_v1):,}")
    cb.metric("In beiden",  f"{len(both_dm):,}")
    cc.metric("V2 DM only", f"{len(only_dm_v2):,}")
    cd.metric("Overlap %",  f"{len(both_dm)/max(len(set_dm_v1|set_dm_v2),1)*100:.1f}%")

    if only_dm_v1 or only_dm_v2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Nur in V1 — DM ({len(only_dm_v1)})**")
            df_o1 = acwi_dm_v1[acwi_dm_v1["Symbol"].isin(only_dm_v1)][
                ["Symbol","Name","Exchange Country Name","Segment","Total MCap Y2025","Free Float MCap Y2025"]].copy()
            df_o1["Total MCap Y2025"] = df_o1["Total MCap Y2025"].apply(format_bn)
            df_o1["Free Float MCap Y2025"] = df_o1["Free Float MCap Y2025"].apply(format_bn)
            st.dataframe(df_o1, use_container_width=True, hide_index=True, height=280)
        with col_b:
            st.markdown(f"**Nur in V2 — DM ({len(only_dm_v2)})**")
            df_o2 = acwi_dm_v2[acwi_dm_v2["Symbol"].isin(only_dm_v2)][
                ["Symbol","Name","Exchange Country Name","Segment","Total MCap Y2025","Free Float MCap Y2025"]].copy()
            df_o2["Total MCap Y2025"] = df_o2["Total MCap Y2025"].apply(format_bn)
            df_o2["Free Float MCap Y2025"] = df_o2["Free Float MCap Y2025"].apply(format_bn)
            st.dataframe(df_o2, use_container_width=True, hide_index=True, height=280)

    # ── Downloads ────────────────────────────────────────────────────────────
    st.markdown("**Downloads**")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ ACWI Variant 1 als Excel",
            data=to_excel_download(
                acwi_v1_thr[[c for c in acwi_v1_thr.columns if c not in ["cum_ff"]]], "ACWI_V1"),
            file_name="NaroIX_ACWI_V1.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with dl2:
        st.download_button(
            "⬇️ ACWI Variant 2 als Excel",
            data=to_excel_download(
                acwi_v2_thr[[c for c in acwi_v2_thr.columns if c not in ["cum_ff"]]], "ACWI_V2"),
            file_name="NaroIX_ACWI_V2.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
