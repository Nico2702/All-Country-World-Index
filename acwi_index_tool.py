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


def apply_step7_coverage(df_candidates, coverage_pct=0.85, country_col="Mapping Country", sort_col="Free Float MCap Y2025"):
    """Step 7: 85% coverage per country.
    sort_col: "Adj_FF_MCap" (default per methodology) or "Free Float MCap Y2025" (raw FF MCap).
    Returns subset of df that passes coverage threshold per country.
    """
    results = []
    country_col = country_col if country_col in df_candidates.columns else "Exchange Country Name"
    if sort_col not in df_candidates.columns:
        sort_col = "Free Float MCap Y2025"
    for country, grp in df_candidates.groupby(country_col):
        grp = grp.sort_values(sort_col, ascending=False).copy()
        total = grp[sort_col].sum()
        if total == 0:
            continue
        grp["_cum"] = grp[sort_col].cumsum()
        grp["_cum_pct"] = grp["_cum"] / total * 100
        target = coverage_pct * 100
        cutoff_idx = grp[grp["_cum_pct"] >= target].index
        if len(cutoff_idx) == 0:
            results.append(grp)
        else:
            results.append(grp.loc[:cutoff_idx[0]])
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


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Datenquelle")
    uploaded = st.file_uploader("FactSet Export (.xlsx)", type=["xlsx","xls"])

    st.markdown("---")
    st.markdown("### 🌍 Universe & Exclusions")
    thailand_sec_type = st.radio("Thailand Sec Type:", ["NVDR","SHARE"], index=0,
        horizontal=True, key="thailand_sec_type",
        help="NVDR: als Secondary behalten (investierbare Klasse). SHARE: normaler Primary.")

    _cpa, _cpb = st.columns([3,4])
    with _cpa: use_max_price = st.checkbox("Max Price ≤", value=True, key="use_max_price")
    with _cpb: _max_price_raw = st.text_input("Max Price", value="20000", key="max_price_input",
        label_visibility="collapsed", disabled=not use_max_price)
    try:    max_closing_price = float(_max_price_raw.replace(",","")) if use_max_price else None
    except: max_closing_price = 20000.0

    with st.expander("Exclusions", expanded=False):
        exclude_hk_cny         = st.checkbox("HK (CNY)", value=True, key="excl_hk")
        exclude_country_risk_na = st.checkbox("Country of Risk = @NA", value=True, key="excl_cor")
        exclude_naics_funds     = st.checkbox("NAICS Investment Funds", value=True, key="excl_naics")
        exclude_euro_mtf        = st.checkbox("Exchange Euro MTF / @NA", value=True, key="excl_euro")
        exclude_etf_sicav       = st.checkbox("Name: ETF / SICAV / %", value=True, key="excl_etf")

    st.markdown("---")
    st.markdown("### 📊 Size Segmentation")
    _la, _lb = st.columns([3,4])
    with _la: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>Large Cap (%)</div>", unsafe_allow_html=True)
    with _lb: _large_raw = st.text_input("Large", value="70", key="large_thr_input", label_visibility="collapsed")
    _ma, _mb = st.columns([3,4])
    with _ma: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>Mid Cap (%)</div>", unsafe_allow_html=True)
    with _mb: _mid_raw = st.text_input("Mid", value="85", key="mid_thr_input", label_visibility="collapsed")
    _sa, _sb = st.columns([3,4])
    with _sa: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>Small Cap (%)</div>", unsafe_allow_html=True)
    with _sb: _small_raw = st.text_input("Small", value="99", key="small_thr_input", label_visibility="collapsed")
    _ffa, _ffb = st.columns([3,4])
    with _ffa: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>Min FF% (%)</div>", unsafe_allow_html=True)
    with _ffb: _ff_raw = st.text_input("Min FF", value="15", key="min_ff_input", label_visibility="collapsed")

    try:    large_thr  = int(_large_raw)
    except: large_thr  = 70
    try:    mid_thr    = int(_mid_raw)
    except: mid_thr    = 85
    try:    small_thr  = int(_small_raw)
    except: small_thr  = 99
    try:    min_ff_pct = float(_ff_raw) / 100
    except: min_ff_pct = 0.15
    # Legacy aliases
    min_ff_pct_leg = min_ff_pct
    dm_ff_gt0 = True
    em_ff_gt0 = True

    st.markdown("---")
    st.markdown("### 💧 Liquidität")
    st.caption("Post-Filter für Tabs 2–4 | Pre-Filter für Tab 5 (GIMI)")
    _adtv_a, _adtv_b = st.columns([3,4])
    with _adtv_a: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>DM ADTV (USD)</div>", unsafe_allow_html=True)
    with _adtv_b: _adtv_dm_raw = st.text_input("DM ADTV", value="1500000", key="adtv_dm_new", label_visibility="collapsed")
    _adtv_c, _adtv_d = st.columns([3,4])
    with _adtv_c: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>EM ADTV (USD)</div>", unsafe_allow_html=True)
    with _adtv_d: _adtv_em_raw = st.text_input("EM ADTV", value="750000", key="adtv_em_new", label_visibility="collapsed")
    _atvr_a, _atvr_b = st.columns([3,4])
    with _atvr_a: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>DM ATVR Min. (%)</div>", unsafe_allow_html=True)
    with _atvr_b: _atvr_dm_raw = st.text_input("DM ATVR", value="0", key="atvr_dm_new", label_visibility="collapsed")
    _atvr_c, _atvr_d = st.columns([3,4])
    with _atvr_c: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>EM ATVR Min. (%)</div>", unsafe_allow_html=True)
    with _atvr_d: _atvr_em_raw = st.text_input("EM ATVR", value="0", key="atvr_em_new", label_visibility="collapsed")

    try:    new_adtv_dm = float(_adtv_dm_raw.replace(",",""))
    except: new_adtv_dm = 1_500_000.0
    try:    new_adtv_em = float(_adtv_em_raw.replace(",",""))
    except: new_adtv_em = 750_000.0
    try:    new_atvr_dm = float(_atvr_dm_raw) / 100
    except: new_atvr_dm = 0.0
    try:    new_atvr_em = float(_atvr_em_raw) / 100
    except: new_atvr_em = 0.0

    # Legacy aliases for old code
    dm_min_adtv_3m = new_adtv_dm; dm_min_adtv_6m = new_adtv_dm
    em_min_adtv_3m = new_adtv_em; em_min_adtv_6m = new_adtv_em
    dm_min_adtv_1m = 0; dm_min_adtv_12m = 0
    em_min_adtv_1m = 0; em_min_adtv_12m = 0
    liq_ratio_min = 0.0

    st.markdown("---")
    st.markdown("### ⚖️ Inclusion Factors")
    _cna, _cnb = st.columns([4,2])
    with _cna: use_china_factor = st.checkbox("China A-Shares", value=True, key="use_china_factor")
    with _cnb: _china_raw = st.text_input("China", value="20", key="china_factor_input", label_visibility="collapsed", disabled=not use_china_factor)
    _ina, _inb = st.columns([4,2])
    with _ina: use_india_factor = st.checkbox("Indien", value=True, key="use_india_factor")
    with _inb: _india_raw = st.text_input("Indien", value="75", key="india_factor_input_v2", label_visibility="collapsed", disabled=not use_india_factor)
    _vna, _vnb = st.columns([4,2])
    with _vna: use_vietnam_factor = st.checkbox("Vietnam", value=True, key="use_vietnam_factor")
    with _vnb: _vietnam_raw = st.text_input("Vietnam", value="50", key="vietnam_factor_input", label_visibility="collapsed", disabled=not use_vietnam_factor)
    _saa, _sab = st.columns([4,2])
    with _saa: use_saudi_factor = st.checkbox("Saudi-Arabien", value=True, key="use_saudi_factor")
    with _sab: _saudi_raw = st.text_input("Saudi", value="50", key="saudi_factor_input", label_visibility="collapsed", disabled=not use_saudi_factor)

    try:    china_inclusion_factor   = float(_china_raw)   / 100 if use_china_factor   else 1.0
    except: china_inclusion_factor   = 0.20
    try:    india_inclusion_factor   = float(_india_raw)   / 100 if use_india_factor   else 1.0
    except: india_inclusion_factor   = 0.75
    try:    vietnam_inclusion_factor = float(_vietnam_raw) / 100 if use_vietnam_factor else 1.0
    except: vietnam_inclusion_factor = 0.50
    try:    saudi_inclusion_factor   = float(_saudi_raw)   / 100 if use_saudi_factor   else 1.0
    except: saudi_inclusion_factor   = 0.50

    st.markdown("---")
    st.markdown("### 🔧 Tab-spezifisch")
    st.caption("Tab 2 — ACWI")
    _ema, _emb = st.columns([3,4])
    with _ema: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>EM Threshold (%)</div>", unsafe_allow_html=True)
    with _emb: _em_thr_raw = st.text_input("EM Threshold", value="33.3", key="em_threshold_input", label_visibility="collapsed")
    try:    em_threshold_pct = float(_em_thr_raw)
    except: em_threshold_pct = 33.3

    st.caption("Tab 5 — GIMI Method")
    _gimi_a, _gimi_b = st.columns([3,4])
    with _gimi_a: st.markdown("<div style='padding-top:8px;font-size:13px;color:#e8eaf6;'>EUMSS FF Ratio (%)</div>", unsafe_allow_html=True)
    with _gimi_b: _eumss_ff_raw = st.text_input("EUMSS FF Ratio", value="50", key="eumss_ff_ratio", label_visibility="collapsed")
    try:    new_eumss_ff_ratio = float(_eumss_ff_raw) / 100
    except: new_eumss_ff_ratio = 0.50

    st.markdown("---")
    st.markdown("<div style='color:#8892b0;font-size:11px;'>NaroIX Index Construction Tool<br/>© 2025 NaroIX</div>", unsafe_allow_html=True)


# ─── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_excel(file):
    try:
        df = pd.read_excel(file, dtype=str)
        return df
    except Exception as e:
        st.error(f"Fehler beim Laden der Datei: {e}")
        return pd.DataFrame()

@st.cache_data
def load_classification(path="Country_Classification.xlsx"):
    try:
        df = pd.read_excel(path, usecols=["Exchange Country Name", "Classification"])
        return df.set_index("Exchange Country Name")["Classification"].to_dict()
    except Exception as e:
        st.error(f"Country_Classification.xlsx konnte nicht geladen werden: {e}")
        return {}

if uploaded:
    df_raw = load_excel(uploaded)
else:
    st.info("👆 Bitte eine Excel-Datei hochladen um zu starten.")
    st.stop()

df_raw_original = df_raw.copy()

# Numeric conversion
for _col in ["Total MCap Y2025","Free Float MCap Y2025","Free Float Percent",
             "1M ADTV Y2025","3M ADTV Y2025","6M ADTV Y2025","12M ADTV Y2025","Closing Price"]:
    if _col in df_raw.columns:
        df_raw[_col] = pd.to_numeric(df_raw[_col], errors="coerce").fillna(0)
        df_raw_original[_col] = df_raw[_col]

# Classification
try:
    _cls_df = pd.read_excel("Country_Classification.xlsx", usecols=["Exchange Country Name","Classification"])
    country_cls = _cls_df.set_index("Exchange Country Name")["Classification"].to_dict()
except:
    country_cls = {}

# Apply classification to df_raw
df_raw["Mapping Country"] = df_raw.apply(
    lambda r: r["Exchange Country Name"] if r.get("Exchange Country Name","") == r.get("Country of Incorp","")
              else r.get("Country of Risk",""), axis=1)
df_raw["Classification"] = df_raw["Mapping Country"].map(country_cls)

# ── Tab 2 (ACWI) specific: build All universe (legacy) ───────────────────────
# Thailand filter
_th_mask = df_raw["Exchange Name"].fillna("").str.upper() == "THAILAND"
_excl_type = "NVDR" if thailand_sec_type == "SHARE" else "SHARE"
df_raw_all = df_raw[~(_th_mask & (df_raw["Sec Type"].fillna("") == _excl_type))].copy()

# Exclusions on All universe
df_raw_all = df_raw_all[df_raw_all["Free Float MCap Y2025"] > 0].copy()
if max_closing_price:
    df_raw_all = df_raw_all[df_raw_all["Closing Price"].fillna(0) < max_closing_price].copy()
if exclude_hk_cny:
    df_raw_all = df_raw_all[~(df_raw_all["Exchange Ticker"].str.contains("HKG", na=False) & (df_raw_all["Trading Currency"] == "CNY"))].copy()
if exclude_country_risk_na:
    df_raw_all = df_raw_all[df_raw_all["Country of Risk"].fillna("") != "@NA"].copy()
if exclude_naics_funds:
    df_raw_all = df_raw_all[~df_raw_all["NAICS"].fillna("").str.contains("Open-End Investment Fund", case=False, na=False)].copy()
if exclude_euro_mtf:
    df_raw_all = df_raw_all[~df_raw_all["Exchange Name"].fillna("").isin(["Euro MTF","@NA"])].copy()
if exclude_etf_sicav:
    import re as _re_etf
    df_raw_all = df_raw_all[~df_raw_all["Name"].fillna("").str.contains(_re_etf.compile(r'\bETF\b|\bSICAV\b|%', _re_etf.IGNORECASE))].copy()
df_raw_all = df_raw_all[df_raw_all["Classification"].notna()].copy()

df_dm_full = df_raw_all[df_raw_all["Classification"] == "DM"].copy()
df_em_full = df_raw_all[df_raw_all["Classification"] == "EM"].copy()

# Apply post-filter (for Tab 2)
def apply_post_filter(df):
    mask = ((df["Classification"]=="DM") &
            (df["3M ADTV Y2025"] >= new_adtv_dm) & (df["6M ADTV Y2025"] >= new_adtv_dm)) | \
           ((df["Classification"]=="EM") &
            (df["3M ADTV Y2025"] >= new_adtv_em) & (df["6M ADTV Y2025"] >= new_adtv_em))
    return df[mask].copy()

# Pre-compute V1 segments for Tab 2
_df_dm_post = apply_post_filter(df_dm_full)
_df_em_post = apply_post_filter(df_em_full)
_seg_dm_v1 = compute_variant1(_df_dm_post, large_thr, mid_thr, small_thr)

# DM cutoff for EM threshold
_cutoff_v1 = get_dm_cutoff_stock(_seg_dm_v1)


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:10px 0 5px'>
  <span style='font-size:28px;font-weight:700;color:#A0B4FF;letter-spacing:2px;'>NaroIX</span>
  <span style='font-size:18px;color:#8892b0;'> — Index Construction Tool</span>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_acwi, tab_gs, tab_pc, tab_gimi, tab_compare = st.tabs([
    "🌍 Universe Overview",
    "🌐 ACWI (DM + EM)",
    "📊 Global Sort (Threshold)",
    "🗺️ Per Country (Solactive)",
    "⚡ GIMI Method",
    "🔀 ACWI & Varianten Vergleich",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Universe Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("## 🌍 Universe Overview")
    st.caption("Rohdaten nach Exclusions und DM/EM Klassifikation — vor Liquiditäts- und Size-Filtern")

    # Use df_raw_all (All listings, after exclusions + classification)
    _ov_dm = df_dm_full.copy()
    _ov_em = df_em_full.copy()
    _ov_all = pd.concat([_ov_dm, _ov_em], ignore_index=True)

    # Top metrics
    _ov_c1,_ov_c2,_ov_c3,_ov_c4,_ov_c5 = st.columns(5)
    _ov_c1.metric("Total Stocks",  f"{len(_ov_all):,}")
    _ov_c2.metric("DM Stocks",     f"{len(_ov_dm):,}")
    _ov_c3.metric("EM Stocks",     f"{len(_ov_em):,}")
    _ov_c4.metric("DM FF MCap",    format_bn(_ov_dm["Free Float MCap Y2025"].sum()))
    _ov_c5.metric("EM FF MCap",    format_bn(_ov_em["Free Float MCap Y2025"].sum()))

    # Country breakdown
    _ov_col1, _ov_col2 = st.columns(2)
    with _ov_col1:
        _dm_ct_ov = _ov_dm.groupby("Exchange Country Name").agg(
            Stocks=("Symbol","count"), FF_MCap=("Free Float MCap Y2025","sum"),
            Avg_MCap=("Total MCap Y2025","mean")).reset_index().sort_values("FF_MCap",ascending=False)
        _dm_ct_ov["FF MCap (USD)"] = _dm_ct_ov["FF_MCap"].apply(format_bn)
        _dm_ct_ov["Avg MCap"]      = _dm_ct_ov["Avg_MCap"].apply(format_bn)
        _dm_ct_ov["Share (%)"]     = (_dm_ct_ov["FF_MCap"]/_dm_ct_ov["FF_MCap"].sum()*100).round(2)
        st.markdown(f"**DM Universe — {len(_ov_dm):,} Stocks**")
        st.dataframe(_dm_ct_ov[["Exchange Country Name","Stocks","FF MCap (USD)","Avg MCap","Share (%)"]],
            use_container_width=True, height=400, hide_index=True)

    with _ov_col2:
        _em_ct_ov = _ov_em.groupby("Exchange Country Name").agg(
            Stocks=("Symbol","count"), FF_MCap=("Free Float MCap Y2025","sum"),
            Avg_MCap=("Total MCap Y2025","mean")).reset_index().sort_values("FF_MCap",ascending=False)
        _em_ct_ov["FF MCap (USD)"] = _em_ct_ov["FF_MCap"].apply(format_bn)
        _em_ct_ov["Avg MCap"]      = _em_ct_ov["Avg_MCap"].apply(format_bn)
        _em_ct_ov["Share (%)"]     = (_em_ct_ov["FF_MCap"]/_em_ct_ov["FF_MCap"].sum()*100).round(2)
        st.markdown(f"**EM Universe — {len(_ov_em):,} Stocks**")
        st.dataframe(_em_ct_ov[["Exchange Country Name","Stocks","FF MCap (USD)","Avg MCap","Share (%)"]],
            use_container_width=True, height=400, hide_index=True)

    # Treemap
    st.markdown("---")
    st.markdown("**FF MCap Verteilung nach Land**")
    _ov_tree = _ov_all.groupby(["Classification","Mapping Country"]).agg(
        FF_MCap=("Free Float MCap Y2025","sum")).reset_index()
    _ov_fig = px.treemap(_ov_tree, path=["Classification","Mapping Country"],
        values="FF_MCap", color="Classification",
        color_discrete_map={"DM":"#2979ff","EM":"#ce93d8"}, template="plotly_dark")
    _ov_fig.update_layout(height=500, paper_bgcolor="#0f1117", margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(_ov_fig, use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ACWI (DM + EM) — V1 Global Sort + EM Threshold + All + Post
# ══════════════════════════════════════════════════════════════════════════════
with tab_acwi:
    st.markdown("## 🌐 ACWI (DM + EM)")
    st.caption("Listing: All (Primary + Secondary) | Filter: Post | DM: Global Sort V1 | EM: Threshold (% DM Grenzstock)")

    # EM: Threshold method
    if _cutoff_v1 is not None:
        _cutoff_mcap = _cutoff_v1["Total MCap Y2025"]
        _em_min_mcap = _cutoff_mcap * (em_threshold_pct / 100)
        df_acwi_dm = _seg_dm_v1[_seg_dm_v1["Segment"].isin(["Large Cap","Mid Cap"])].copy()
        df_acwi_em = filter_em_by_threshold(_df_em_post, _cutoff_mcap, em_threshold_pct / 100)
        df_acwi_em = apply_post_filter(df_acwi_em)

        # Add adjusted weights
        df_acwi_dm = add_adjusted_weight(df_acwi_dm, china_inclusion_factor, india_inclusion_factor,
                                          vietnam_inclusion_factor, saudi_inclusion_factor)
        df_acwi_em = add_adjusted_weight(df_acwi_em, china_inclusion_factor, india_inclusion_factor,
                                          vietnam_inclusion_factor, saudi_inclusion_factor)

        _total_adj_acwi = df_acwi_dm["Adjusted FF MCap"].sum() + df_acwi_em["Adjusted FF MCap"].sum()
        _em_adj_acwi    = df_acwi_em["Adjusted FF MCap"].sum()
        _em_w_acwi      = _em_adj_acwi / _total_adj_acwi * 100 if _total_adj_acwi > 0 else 0

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("DM Stocks",     f"{len(df_acwi_dm):,}")
        c2.metric("EM Stocks",     f"{len(df_acwi_em):,}")
        c3.metric("Total ACWI",    f"{len(df_acwi_dm)+len(df_acwi_em):,}")
        c4.metric("DM FF MCap",    format_bn(df_acwi_dm["Free Float MCap Y2025"].sum()))
        c5.metric("EM FF MCap",    format_bn(df_acwi_em["Free Float MCap Y2025"].sum()))
        c6.metric("EM Adj. Weight",f"{_em_w_acwi:.2f}%")

        st.markdown(f"**EM Threshold:** ≥ {em_threshold_pct:.1f}% × DM Grenzstock Total MCap = {format_bn(_em_min_mcap)}")

        # DM Segments table
        _col_dm_t, _col_em_t = st.columns(2)
        with _col_dm_t:
            st.markdown("**DM Segments (World Index)**")
            _dm_seg_rows = []
            for _seg in ["Large Cap","Mid Cap"]:
                _s = df_acwi_dm[df_acwi_dm["Segment"]==_seg]
                _dm_seg_rows.append({"Segment":_seg,"# Stocks":len(_s),
                    "FF MCap":format_bn(_s["Free Float MCap Y2025"].sum()),
                    "Adj. FF MCap":format_bn(_s["Adjusted FF MCap"].sum()),
                    "Weight %":f"{_s['Adjusted FF MCap'].sum()/_total_adj_acwi*100:.2f}%" if _total_adj_acwi>0 else "—"})
            _dm_tot = df_acwi_dm[df_acwi_dm["Segment"].isin(["Large Cap","Mid Cap"])]
            _dm_seg_rows.append({"Segment":"── World Index Total","# Stocks":len(_dm_tot),
                "FF MCap":format_bn(_dm_tot["Free Float MCap Y2025"].sum()),
                "Adj. FF MCap":format_bn(_dm_tot["Adjusted FF MCap"].sum()),
                "Weight %":f"{_dm_tot['Adjusted FF MCap'].sum()/_total_adj_acwi*100:.2f}%" if _total_adj_acwi>0 else "—"})
            st.dataframe(style_segment_table(pd.DataFrame(_dm_seg_rows),["Large Cap","Mid Cap","── World Index Total"]),
                use_container_width=True, hide_index=True)

        with _col_em_t:
            st.markdown("**EM Segments**")
            _em_seg_rows = [{"Segment":"EM Included","# Stocks":len(df_acwi_em),
                "FF MCap":format_bn(df_acwi_em["Free Float MCap Y2025"].sum()),
                "Adj. FF MCap":format_bn(df_acwi_em["Adjusted FF MCap"].sum()),
                "Weight %":f"{_em_adj_acwi/_total_adj_acwi*100:.2f}%" if _total_adj_acwi>0 else "—"}]
            st.dataframe(pd.DataFrame(_em_seg_rows), use_container_width=True, hide_index=True)

        # Country breakdown
        _ccol1, _ccol2 = st.columns(2)
        with _ccol1:
            st.markdown(f"**DM — Country Breakdown ({len(df_acwi_dm):,} Stocks)**")
            _dm_ct = df_acwi_dm.groupby("Exchange Country Name").agg(
                Stocks=("Symbol","count"), FF_MCap=("Free Float MCap Y2025","sum"),
                Adj=("Adjusted FF MCap","sum"), Avg=("Total MCap Y2025","mean")).reset_index()
            _dm_ct["FF MCap"] = _dm_ct["FF_MCap"].apply(format_bn)
            _dm_ct["Avg MCap"] = _dm_ct["Avg"].apply(format_bn)
            _dm_ct["Weight %"] = (_dm_ct["Adj"]/_total_adj_acwi*100).round(2)
            st.dataframe(_dm_ct.sort_values("Adj",ascending=False)[["Exchange Country Name","Stocks","FF MCap","Avg MCap","Weight %"]].rename(columns={"Exchange Country Name":"Land"}),
                use_container_width=True, hide_index=True)
        with _ccol2:
            st.markdown(f"**EM — Country Breakdown ({len(df_acwi_em):,} Stocks)**")
            _em_ct = df_acwi_em.groupby("Exchange Country Name").agg(
                Stocks=("Symbol","count"), FF_MCap=("Free Float MCap Y2025","sum"),
                Adj=("Adjusted FF MCap","sum"), Avg=("Total MCap Y2025","mean")).reset_index()
            _em_ct["FF MCap"] = _em_ct["FF_MCap"].apply(format_bn)
            _em_ct["Avg MCap"] = _em_ct["Avg"].apply(format_bn)
            _em_ct["Weight %"] = (_em_ct["Adj"]/_total_adj_acwi*100).round(2)
            st.dataframe(_em_ct.sort_values("Adj",ascending=False)[["Exchange Country Name","Stocks","FF MCap","Avg MCap","Weight %"]].rename(columns={"Exchange Country Name":"Land"}),
                use_container_width=True, hide_index=True)

        # Country Charts
        _acwi_all2 = pd.concat([df_acwi_dm, df_acwi_em], ignore_index=True)
        _total_adj2 = _acwi_all2["Adjusted FF MCap"].sum()
        _by_w2 = _acwi_all2.groupby("Exchange Country Name").agg(
            Stocks=("Symbol","count"), Adj=("Adjusted FF MCap","sum")).reset_index()
        _by_w2["Weight%"] = (_by_w2["Adj"]/_total_adj2*100).round(2)
        _top30 = _by_w2.sort_values("Adj",ascending=False).head(30)
        _rest2 = _by_w2.sort_values("Adj",ascending=False).iloc[30:]
        if len(_rest2):
            _top30 = pd.concat([pd.DataFrame([{"Exchange Country Name":f"Others ({len(_rest2)})","Stocks":_rest2["Stocks"].sum(),"Adj":_rest2["Adj"].sum(),"Weight%":_rest2["Weight%"].sum()}]),_top30])
        _top30 = _top30.sort_values("Adj",ascending=True)

        _by_s2 = _acwi_all2.groupby("Exchange Country Name").agg(Stocks=("Symbol","count")).reset_index()
        _by_s2["Pct"] = (_by_s2["Stocks"]/len(_acwi_all2)*100).round(2)
        _tops = _by_s2.sort_values("Stocks",ascending=False).head(30)
        _rests = _by_s2.sort_values("Stocks",ascending=False).iloc[30:]
        if len(_rests):
            _tops = pd.concat([pd.DataFrame([{"Exchange Country Name":f"Others ({len(_rests)})","Stocks":_rests["Stocks"].sum(),"Pct":_rests["Pct"].sum()}]),_tops])
        _tops = _tops.sort_values("Stocks",ascending=True)

        _ch_col1, _ch_col2 = st.columns(2)
        with _ch_col1:
            st.markdown("**Nach Anzahl Stocks (%)**")
            fig_s2 = go.Figure(go.Bar(x=_tops["Pct"],y=_tops["Exchange Country Name"],orientation="h",
                marker_color="#2979ff",text=_tops["Pct"].apply(lambda x:f"{x:.2f}%"),textposition="outside"))
            fig_s2.update_layout(template="plotly_dark",paper_bgcolor="#0f1117",plot_bgcolor="#161b27",
                height=700,margin=dict(t=10,b=10,l=10,r=60),xaxis=dict(showgrid=False))
            st.plotly_chart(fig_s2,use_container_width=True)
        with _ch_col2:
            st.markdown("**Nach Gewicht (Adj. FF MCap %)**")
            fig_w2 = go.Figure(go.Bar(x=_top30["Weight%"],y=_top30["Exchange Country Name"],orientation="h",
                marker_color="#ce93d8",text=_top30["Weight%"].apply(lambda x:f"{x:.2f}%"),textposition="outside"))
            fig_w2.update_layout(template="plotly_dark",paper_bgcolor="#0f1117",plot_bgcolor="#161b27",
                height=700,margin=dict(t=10,b=10,l=10,r=60),xaxis=dict(showgrid=False))
            st.plotly_chart(fig_w2,use_container_width=True)

        # Donut
        _donut2 = pd.DataFrame([
            {"Label":"DM","FF MCap":df_acwi_dm["Adjusted FF MCap"].sum()},
            {"Label":"EM","FF MCap":df_acwi_em["Adjusted FF MCap"].sum()},
        ])
        _dc1,_dc2 = st.columns([1,1])
        with _dc1:
            st.markdown("**ACWI Composition (DM vs EM)**")
            fig_d2 = px.pie(_donut2,names="Label",values="FF MCap",
                color="Label",color_discrete_map={"DM":"#2979ff","EM":"#ce93d8"},
                template="plotly_dark",hole=0.45)
            fig_d2.update_layout(paper_bgcolor="#0f1117",height=350,margin=dict(t=10,b=10))
            st.plotly_chart(fig_d2,use_container_width=True)

        # Download
        _acwi_dl2 = pd.concat([df_acwi_dm, df_acwi_em], ignore_index=True)
        st.download_button("⬇️ Download ACWI als Excel",
            data=to_excel_multi({"World Index (DM)":df_acwi_dm,"EM Index":df_acwi_em,"ACWI":_acwi_dl2,
                "Parameter":pd.DataFrame([{"Parameter":"Methodik","Wert":"V1 Global Sort + EM Threshold"},
                    {"Parameter":"Listing","Wert":"All (Primary + Secondary)"},
                    {"Parameter":"Filter","Wert":"Post"},
                    {"Parameter":"EM Threshold %","Wert":f"{em_threshold_pct:.1f}%"},
                    {"Parameter":"DM ADTV","Wert":f"{new_adtv_dm:,.0f}"},
                    {"Parameter":"EM ADTV","Wert":f"{new_adtv_em:,.0f}"}])}),
            file_name="NaroIX_ACWI_V1_Threshold.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.warning("Kein DM Grenzstock gefunden — bitte Daten prüfen.")

# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPER — New Universe Builder (Primary only + Thailand NVDR exception)
# ══════════════════════════════════════════════════════════════════════════════
def build_new_universe(df_raw_orig, country_cls, thailand_mode, max_price,
                       excl_hk_cny, excl_cor_na, excl_naics, excl_euro, excl_etf,
                       china_if, india_if, vietnam_if, saudi_if):
    """Build clean Primary-only universe with Thailand NVDR exception."""
    import re as _re
    df = df_raw_orig.copy()
    for col in ["Total MCap Y2025","Free Float MCap Y2025","Free Float Percent",
                "1M ADTV Y2025","3M ADTV Y2025","6M ADTV Y2025","12M ADTV Y2025","Closing Price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Step 1: Thailand NVDR exception (before Primary filter)
    _th = df["Exchange Name"].fillna("").str.upper() == "THAILAND"
    if thailand_mode == "NVDR":
        # Remove SHARE, keep NVDR (even though Secondary)
        df = df[~(_th & (df["Sec Type"].fillna("") == "SHARE"))].copy()
    else:
        # Remove NVDR, keep SHARE
        df = df[~(_th & (df["Sec Type"].fillna("") == "NVDR"))].copy()

    # Step 2: Primary only (Thailand NVDRs already handled)
    _th_nvdr = df["Exchange Name"].fillna("").str.upper() == "THAILAND"  # remaining Thai stocks
    df = df[(df["Listing"].fillna("") == "Primary") | _th_nvdr].copy()

    # Step 3: Exclusions
    df = df[df["Free Float MCap Y2025"] > 0].copy()
    if max_price:
        df = df[df["Closing Price"].fillna(0) < max_price].copy()
    if excl_hk_cny:
        df = df[~(df["Exchange Ticker"].str.contains("HKG", na=False) & (df["Trading Currency"] == "CNY"))].copy()
    if excl_cor_na:
        df = df[df["Country of Risk"].fillna("") != "@NA"].copy()
    if excl_naics:
        df = df[~df["NAICS"].fillna("").str.contains("Open-End Investment Fund", case=False, na=False)].copy()
    if excl_euro:
        df = df[~df["Exchange Name"].fillna("").isin(["Euro MTF", "@NA"])].copy()
    if excl_etf:
        df = df[~df["Name"].fillna("").str.contains(_re.compile(r'\bETF\b|\bSICAV\b|%', _re.IGNORECASE))].copy()

    # Step 4: Classification
    df["Mapping Country"] = df.apply(
        lambda r: r["Exchange Country Name"] if r.get("Exchange Country Name","") == r.get("Country of Incorp","")
                  else r.get("Country of Risk",""), axis=1)
    df["Classification"] = df["Mapping Country"].map(country_cls)
    df = df[df["Classification"].notna()].copy()

    # Step 5: Inclusion Factors
    ecn = df["Exchange Country Name"].fillna("")
    df["IF"] = np.where(ecn.str.upper()=="CHINA",        china_if,
               np.where(ecn.str.upper()=="INDIA",        india_if,
               np.where(ecn.str.upper()=="VIETNAM",      vietnam_if,
               np.where(ecn.str.upper()=="SAUDI ARABIA", saudi_if, 1.0))))
    df["Adj_FF_MCap"] = df["Free Float MCap Y2025"] * df["IF"]

    # ADTV best for ATVR
    df["ADTV_Best"] = df["12M ADTV Y2025"].where(df["12M ADTV Y2025"]>0,
                      df["6M ADTV Y2025"].where(df["6M ADTV Y2025"]>0,
                      df["3M ADTV Y2025"].where(df["3M ADTV Y2025"]>0,
                      df["1M ADTV Y2025"])))
    df["ATVR"] = np.where(df["Total MCap Y2025"]>0,
                          df["ADTV_Best"]*252/df["Total MCap Y2025"], 0)
    return df


def apply_liquidity_new(df, adtv_dm, adtv_em, atvr_dm, atvr_em):
    """Apply ADTV + ATVR filter."""
    mask = ((df["Classification"]=="DM") &
            (df["3M ADTV Y2025"]>=adtv_dm) &
            (df["6M ADTV Y2025"]>=adtv_dm) &
            (df["ATVR"]>=atvr_dm)) | \
           ((df["Classification"]=="EM") &
            (df["3M ADTV Y2025"]>=adtv_em) &
            (df["6M ADTV Y2025"]>=adtv_em) &
            (df["ATVR"]>=atvr_em))
    return df[mask].copy()


def assign_segments_new(df, large_pct, mid_pct, small_pct, group_col="Mapping Country", sort_col="Adj_FF_MCap"):
    """Assign Large/Mid/Small/Micro Cap segments per group."""
    results = []
    for grp_val, grp in df.groupby(group_col):
        grp = grp.sort_values(sort_col, ascending=False).copy()
        total = grp[sort_col].sum()
        if total == 0:
            grp["Segment_New"] = "Micro Cap"
            results.append(grp)
            continue
        grp["_cum_pct"] = grp[sort_col].cumsum() / total * 100
        grp["Segment_New"] = np.where(grp["_cum_pct"] <= large_pct, "Large Cap",
                             np.where(grp["_cum_pct"] <= mid_pct,   "Mid Cap",
                             np.where(grp["_cum_pct"] <= small_pct, "Small Cap", "Micro Cap")))
        results.append(grp)
    return pd.concat(results, ignore_index=True)


def add_step8_secondary_new(df_selected, df_raw_orig, adtv_dm, adtv_em, atvr_dm, atvr_em,
                             max_price, thailand_mode, china_if, india_if, vietnam_if, saudi_if):
    """Step 8: Add secondary share classes for selected entities."""
    selected_entities = set(df_selected["Entity ID"].dropna().unique())
    df_sec = df_raw_orig[
        (df_raw_orig["Listing"].fillna("") == "Secondary") &
        (df_raw_orig["Entity ID"].isin(selected_entities))
    ].copy()
    # Exclude Thailand NVDRs/SHAREs already handled
    _th = df_sec["Exchange Name"].fillna("").str.upper() == "THAILAND"
    if thailand_mode == "NVDR":
        df_sec = df_sec[~(_th & (df_sec["Sec Type"].fillna("") == "SHARE"))].copy()
    else:
        df_sec = df_sec[~(_th & (df_sec["Sec Type"].fillna("") == "NVDR"))].copy()

    if len(df_sec) == 0:
        return df_selected

    for col in ["Total MCap Y2025","Free Float MCap Y2025","Free Float Percent",
                "1M ADTV Y2025","3M ADTV Y2025","6M ADTV Y2025","12M ADTV Y2025","Closing Price"]:
        df_sec[col] = pd.to_numeric(df_sec[col], errors="coerce").fillna(0)

    df_sec = df_sec[df_sec["Free Float MCap Y2025"] > 0].copy()
    if max_price:
        df_sec = df_sec[df_sec["Closing Price"].fillna(0) < max_price].copy()

    ecn = df_sec["Exchange Country Name"].fillna("")
    df_sec["IF"] = np.where(ecn.str.upper()=="CHINA",        china_if,
                   np.where(ecn.str.upper()=="INDIA",        india_if,
                   np.where(ecn.str.upper()=="VIETNAM",      vietnam_if,
                   np.where(ecn.str.upper()=="SAUDI ARABIA", saudi_if, 1.0))))
    df_sec["Adj_FF_MCap"] = df_sec["Free Float MCap Y2025"] * df_sec["IF"]
    df_sec["ADTV_Best"] = df_sec["12M ADTV Y2025"].where(df_sec["12M ADTV Y2025"]>0,
                          df_sec["6M ADTV Y2025"].where(df_sec["6M ADTV Y2025"]>0,
                          df_sec["3M ADTV Y2025"].where(df_sec["3M ADTV Y2025"]>0,
                          df_sec["1M ADTV Y2025"])))
    df_sec["ATVR"] = np.where(df_sec["Total MCap Y2025"]>0,
                              df_sec["ADTV_Best"]*252/df_sec["Total MCap Y2025"], 0)

    # Remove already included symbols
    existing_symbols = set(df_selected["Symbol"].unique())
    df_sec = df_sec[~df_sec["Symbol"].isin(existing_symbols)].copy()

    if len(df_sec) == 0:
        return df_selected
    return pd.concat([df_selected, df_sec], ignore_index=True)


def render_new_tab(tab_name, df_included, large_pct, mid_pct,
                   china_if, india_if, vietnam_if, saudi_if,
                   params_dict):
    """Render standard visuals for a new index tab."""

    df_dm = df_included[df_included["Classification"]=="DM"].copy()
    df_em = df_included[df_included["Classification"]=="EM"].copy()

    seg_order = ["Large Cap","Mid Cap","Small Cap","Micro Cap"]

    # ── Top metrics ──────────────────────────────────────────────────────────
    total_adj = df_included["Adj_FF_MCap"].sum()
    em_adj    = df_em["Adj_FF_MCap"].sum()
    em_w      = em_adj/total_adj*100 if total_adj > 0 else 0

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Total Stocks",  f"{len(df_included):,}")
    m2.metric("DM Stocks",     f"{len(df_dm):,}")
    m3.metric("EM Stocks",     f"{len(df_em):,}")
    m4.metric("DM FF MCap",    format_bn(df_dm["Free Float MCap Y2025"].sum()))
    m5.metric("EM FF MCap",    format_bn(df_em["Free Float MCap Y2025"].sum()))
    m6.metric("EM Adj. Weight",f"{em_w:.2f}%")

    # ── 5 Index Products ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Index-Produkte**")
    _world_dm  = df_dm[df_dm["Segment_New"].isin(["Large Cap","Mid Cap"])]
    _world_em  = df_em[df_em["Segment_New"].isin(["Large Cap","Mid Cap"])]
    _imi_dm_s  = df_dm[df_dm["Segment_New"]=="Small Cap"]
    _imi_em_s  = df_em[df_em["Segment_New"]=="Small Cap"]

    _idx_rows = [
        {"Index": "🌍 World Index",  "DM": len(_world_dm), "EM": "—", "Total": len(_world_dm),
         "DM FF MCap": format_bn(_world_dm["Free Float MCap Y2025"].sum()),
         "Adj. Weight DM": f"{_world_dm['Adj_FF_MCap'].sum()/total_adj*100:.2f}%" if total_adj>0 else "—"},
        {"Index": "🌏 EM Index",     "DM": "—", "EM": len(_world_em), "Total": len(_world_em),
         "DM FF MCap": "—",
         "Adj. Weight DM": f"{_world_em['Adj_FF_MCap'].sum()/total_adj*100:.2f}%" if total_adj>0 else "—"},
        {"Index": "🌐 ACWI Index",   "DM": len(_world_dm), "EM": len(_world_em), "Total": len(_world_dm)+len(_world_em),
         "DM FF MCap": format_bn((_world_dm["Free Float MCap Y2025"].sum()+_world_em["Free Float MCap Y2025"].sum())),
         "Adj. Weight DM": "100.00%"},
        {"Index": "🌍+ World IMI",   "DM": len(_world_dm)+len(_imi_dm_s), "EM": "—", "Total": len(_world_dm)+len(_imi_dm_s),
         "DM FF MCap": format_bn((_world_dm["Free Float MCap Y2025"].sum()+_imi_dm_s["Free Float MCap Y2025"].sum())),
         "Adj. Weight DM": "—"},
        {"Index": "🌐+ ACWI IMI",    "DM": len(_world_dm)+len(_imi_dm_s), "EM": len(_world_em)+len(_imi_em_s),
         "Total": len(_world_dm)+len(_imi_dm_s)+len(_world_em)+len(_imi_em_s),
         "DM FF MCap": "—", "Adj. Weight DM": "—"},
    ]
    st.dataframe(pd.DataFrame(_idx_rows), use_container_width=True, hide_index=True)

    # ── Segment Tables ────────────────────────────────────────────────────────
    st.markdown("---")
    _sc1, _sc2 = st.columns(2)

    def seg_table(df_cls, label):
        rows = []
        total_ff = df_cls["Free Float MCap Y2025"].sum()
        total_adj_cls = df_cls["Adj_FF_MCap"].sum()
        for seg in seg_order:
            s = df_cls[df_cls["Segment_New"]==seg]
            rows.append({
                "Segment": seg,
                "Stocks": len(s),
                "FF MCap": format_bn(s["Free Float MCap Y2025"].sum()) if len(s)>0 else "—",
                "Adj. FF MCap": format_bn(s["Adj_FF_MCap"].sum()) if len(s)>0 else "—",
                "Weight %": f"{s['Adj_FF_MCap'].sum()/total_adj*100:.2f}%" if total_adj>0 and len(s)>0 else "—",
            })
        # Standard Index subtotal
        std = df_cls[df_cls["Segment_New"].isin(["Large Cap","Mid Cap"])]
        rows.insert(2, {
            "Segment": f"── {label} Index (Large+Mid)",
            "Stocks": len(std),
            "FF MCap": format_bn(std["Free Float MCap Y2025"].sum()),
            "Adj. FF MCap": format_bn(std["Adj_FF_MCap"].sum()),
            "Weight %": f"{std['Adj_FF_MCap'].sum()/total_adj*100:.2f}%" if total_adj>0 else "—",
        })
        return pd.DataFrame(rows)

    with _sc1:
        st.markdown("**DM Segmente**")
        _dm_seg = seg_table(df_dm, "World")
        def _style_dm_seg(df):
            def rs(row):
                if "World Index" in row["Segment"]: return ["background-color:#1a3a5c;font-weight:700;"]*len(row)
                return [""]*len(row)
            return df.style.apply(rs, axis=1)
        st.dataframe(_style_dm_seg(_dm_seg), use_container_width=True, hide_index=True)

    with _sc2:
        st.markdown("**EM Segmente**")
        _em_seg = seg_table(df_em, "EM")
        def _style_em_seg(df):
            def rs(row):
                if "EM Index" in row["Segment"]: return ["background-color:#1a2a1a;font-weight:700;"]*len(row)
                return [""]*len(row)
            return df.style.apply(rs, axis=1)
        st.dataframe(_style_em_seg(_em_seg), use_container_width=True, hide_index=True)

    # ── Country Breakdown ─────────────────────────────────────────────────────
    st.markdown("---")
    _cc1, _cc2 = st.columns(2)

    def country_table(df_cls):
        ct = df_cls.groupby("Mapping Country").agg(
            Stocks=("Symbol","count"),
            FF_MCap=("Free Float MCap Y2025","sum"),
            Adj_MCap=("Adj_FF_MCap","sum"),
            Avg_MCap=("Total MCap Y2025","mean"),
        ).reset_index().sort_values("Adj_MCap", ascending=False)
        ct["FF MCap"] = ct["FF_MCap"].apply(format_bn)
        ct["Avg MCap"] = ct["Avg_MCap"].apply(format_bn)
        ct["Weight %"] = (ct["Adj_MCap"]/total_adj*100).round(2)
        return ct[["Mapping Country","Stocks","FF MCap","Avg MCap","Weight %"]].rename(columns={"Mapping Country":"Land"})

    with _cc1:
        st.markdown(f"**DM Country Breakdown ({len(df_dm):,} Stocks)**")
        st.dataframe(country_table(df_dm[df_dm["Segment_New"].isin(["Large Cap","Mid Cap"])]),
                     use_container_width=True, hide_index=True)
    with _cc2:
        st.markdown(f"**EM Country Breakdown ({len(df_em):,} Stocks)**")
        st.dataframe(country_table(df_em[df_em["Segment_New"].isin(["Large Cap","Mid Cap"])]),
                     use_container_width=True, hide_index=True)

    # ── Country Charts ────────────────────────────────────────────────────────
    st.markdown("---")
    _acwi_std = df_included[df_included["Segment_New"].isin(["Large Cap","Mid Cap"])].copy()
    _by_w = _acwi_std.groupby("Mapping Country").agg(
        Stocks=("Symbol","count"), Adj=("Adj_FF_MCap","sum")).reset_index()
    _by_w["Weight%"] = (_by_w["Adj"]/_acwi_std["Adj_FF_MCap"].sum()*100).round(2)
    _by_w = _by_w.sort_values("Adj", ascending=False)
    _top30 = _by_w.head(30)
    _rest  = _by_w.iloc[30:]
    if len(_rest):
        _top30 = pd.concat([pd.DataFrame([{"Mapping Country":f"Others ({len(_rest)})", "Stocks":_rest["Stocks"].sum(), "Adj":_rest["Adj"].sum(), "Weight%":_rest["Weight%"].sum()}]), _top30])
    _top30 = _top30.sort_values("Adj", ascending=True)

    _ch1, _ch2 = st.columns(2)
    with _ch1:
        st.markdown("**Nach Anzahl Stocks (%)**")
        _by_s2 = _acwi_std.groupby("Mapping Country").agg(Stocks=("Symbol","count")).reset_index()
        _by_s2["Pct"] = (_by_s2["Stocks"]/len(_acwi_std)*100).round(2)
        _by_s2 = _by_s2.sort_values("Stocks", ascending=False)
        _top30s = _by_s2.head(30)
        _rests  = _by_s2.iloc[30:]
        if len(_rests):
            _top30s = pd.concat([pd.DataFrame([{"Mapping Country":f"Others ({len(_rests)})", "Stocks":_rests["Stocks"].sum(), "Pct":_rests["Pct"].sum()}]), _top30s])
        _top30s = _top30s.sort_values("Stocks", ascending=True)
        fig_s = go.Figure(go.Bar(x=_top30s["Pct"], y=_top30s["Mapping Country"],
            orientation="h", marker_color="#2979ff",
            text=_top30s["Pct"].apply(lambda x: f"{x:.2f}%"), textposition="outside"))
        fig_s.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
            height=700, margin=dict(t=10,b=10,l=10,r=60), xaxis=dict(showgrid=False))
        st.plotly_chart(fig_s, use_container_width=True)

    with _ch2:
        st.markdown("**Nach Gewicht (Adj. FF MCap %)**")
        fig_w = go.Figure(go.Bar(x=_top30["Weight%"], y=_top30["Mapping Country"],
            orientation="h", marker_color="#ce93d8",
            text=_top30["Weight%"].apply(lambda x: f"{x:.2f}%"), textposition="outside"))
        fig_w.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
            height=700, margin=dict(t=10,b=10,l=10,r=60), xaxis=dict(showgrid=False))
        st.plotly_chart(fig_w, use_container_width=True)

    # ── Donut + IF Impact ────────────────────────────────────────────────────
    st.markdown("---")
    _d1, _d2 = st.columns([1,1])
    with _d1:
        st.markdown("**ACWI Composition (DM vs EM)**")
        _donut = pd.DataFrame([
            {"Label":"DM","FF MCap":df_dm[df_dm["Segment_New"].isin(["Large Cap","Mid Cap"])]["Adj_FF_MCap"].sum()},
            {"Label":"EM","FF MCap":df_em[df_em["Segment_New"].isin(["Large Cap","Mid Cap"])]["Adj_FF_MCap"].sum()},
        ])
        fig_d = px.pie(_donut, names="Label", values="FF MCap",
            color="Label", color_discrete_map={"DM":"#2979ff","EM":"#ce93d8"},
            template="plotly_dark", hole=0.45)
        fig_d.update_layout(paper_bgcolor="#0f1117", height=350, margin=dict(t=10,b=10))
        st.plotly_chart(fig_d, use_container_width=True)

    with _d2:
        st.markdown("**Inclusion Factor Impact**")
        _if_entries = [
            ("China A-Shares (IF 20%)",
             df_included["Exchange Country Name"].fillna("").str.upper()=="CHINA", china_if),
            ("China H-Shares / Red Chips (IF 100%)",
             (df_included["Mapping Country"].fillna("").str.upper()=="CHINA") &
             (df_included["Exchange Country Name"].fillna("").str.upper()!="CHINA"), 1.0),
            ("Indien", df_included["Exchange Country Name"].fillna("").str.upper()=="INDIA", india_if),
            ("Vietnam", df_included["Exchange Country Name"].fillna("").str.upper()=="VIETNAM", vietnam_if),
            ("Saudi-Arabien", df_included["Exchange Country Name"].fillna("").str.upper()=="SAUDI ARABIA", saudi_if),
        ]
        _if_rows = []
        _tot_ff = df_included["Free Float MCap Y2025"].sum()
        _tot_adj2 = df_included["Adj_FF_MCap"].sum()
        for _nm, _msk, _fac in _if_entries:
            _ff = df_included.loc[_msk,"Free Float MCap Y2025"].sum()
            _adj = df_included.loc[_msk,"Adj_FF_MCap"].sum()
            if _ff > 0:
                _if_rows.append({"Land":_nm,"Factor":f"{_fac*100:.0f}%",
                    "Weight (vor)": round(_ff/_tot_ff*100,4) if _tot_ff>0 else 0,
                    "Weight (nach)": round(_adj/_tot_adj2*100,4) if _tot_adj2>0 else 0,
                    "Δ": round(_adj/_tot_adj2*100-_ff/_tot_ff*100,4) if _tot_ff>0 and _tot_adj2>0 else 0})
        if _if_rows:
            _if_df = pd.DataFrame(_if_rows)
            _if_df = pd.concat([_if_df, pd.DataFrame([{
                "Land":"Total","Factor":"—",
                "Weight (vor)": round(_if_df["Weight (vor)"].sum(),4),
                "Weight (nach)": round(_if_df["Weight (nach)"].sum(),4),
                "Δ": round(_if_df["Δ"].sum(),4)}])], ignore_index=True)
            def _sif(df):
                def rs(row):
                    if row["Land"]=="Total": return ["background-color:#1a2a4a;font-weight:600;"]*len(row)
                    return [""]*len(row)
                return df.style.apply(rs, axis=1)
            st.dataframe(_sif(_if_df), use_container_width=True, hide_index=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    _drop = ["_cum_pct","ADTV_Best","ATVR"]
    _world_dl = df_included[df_included["Segment_New"].isin(["Large Cap","Mid Cap","Small Cap","Micro Cap"])].copy()
    _acwi_dl  = df_included[df_included["Segment_New"].isin(["Large Cap","Mid Cap"])].copy()
    _params_dl = pd.DataFrame([{"Parameter":k,"Wert":v} for k,v in params_dict.items()])

    st.download_button(
        f"⬇️ Download {tab_name} als Excel",
        data=to_excel_multi({
            "Universe":         df_included[[c for c in df_included.columns if c not in _drop]],
            "World Index (DM)": df_included[(df_included["Classification"]=="DM") & df_included["Segment_New"].isin(["Large Cap","Mid Cap"])][[c for c in df_included.columns if c not in _drop]],
            "EM Index":         df_included[(df_included["Classification"]=="EM") & df_included["Segment_New"].isin(["Large Cap","Mid Cap"])][[c for c in df_included.columns if c not in _drop]],
            "ACWI Index":       _acwi_dl[[c for c in _acwi_dl.columns if c not in _drop]],
            "World IMI":        df_included[(df_included["Classification"]=="DM") & df_included["Segment_New"].isin(["Large Cap","Mid Cap","Small Cap"])][[c for c in df_included.columns if c not in _drop]],
            "ACWI IMI":         df_included[df_included["Segment_New"].isin(["Large Cap","Mid Cap","Small Cap"])][[c for c in df_included.columns if c not in _drop]],
            "Parameter Settings": _params_dl,
        }),
        file_name=f"NaroIX_{tab_name.replace(' ','_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: Global Sort (Threshold)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Global Sort (Threshold)
# ══════════════════════════════════════════════════════════════════════════════
with tab_gs:
    st.markdown("## 📊 Global Sort (Threshold)")
    st.caption("Primary only + Secondary Schritt 8 | Post-Liquiditätsfilter | Globale Sortierung nach Total MCap")

    _gs_u = build_new_universe(df_raw_original, country_cls, thailand_sec_type, max_closing_price,
        exclude_hk_cny, exclude_country_risk_na, exclude_naics_funds, exclude_euro_mtf, exclude_etf_sicav,
        china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)

    _gs_liq = apply_liquidity_new(_gs_u, new_adtv_dm, new_adtv_em, new_atvr_dm, new_atvr_em)

    # Global sort → segments
    _gs_sorted = _gs_liq.sort_values("Total MCap Y2025", ascending=False).copy()
    _gs_ff_tot = _gs_sorted["Free Float MCap Y2025"].sum()
    _gs_sorted["_cum_pct"] = _gs_sorted["Free Float MCap Y2025"].cumsum() / _gs_ff_tot * 100 if _gs_ff_tot>0 else 0
    _gs_sorted["Segment_New"] = np.where(_gs_sorted["_cum_pct"] <= large_thr, "Large Cap",
                                np.where(_gs_sorted["_cum_pct"] <= mid_thr,   "Mid Cap",
                                np.where(_gs_sorted["_cum_pct"] <= small_thr, "Small Cap", "Micro Cap")))

    _gs_final = add_step8_secondary_new(_gs_sorted, df_raw_original, new_adtv_dm, new_adtv_em,
        new_atvr_dm, new_atvr_em, max_closing_price, thailand_sec_type,
        china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)

    _gs_tot_adj = _gs_final["Adj_FF_MCap"].sum()
    _gs_final["Index_Weight"] = _gs_final["Adj_FF_MCap"]/_gs_tot_adj*100 if _gs_tot_adj>0 else 0

    _gs_params = {"Methodik":"Global Sort (Threshold)","Listing":"Primary only + Secondary Schritt 8",
        "Filter":"Post","Large Cap (%)":large_thr,"Mid Cap (%)":mid_thr,"Small Cap (%)":small_thr,
        "DM ADTV (USD)":f"{new_adtv_dm:,.0f}","EM ADTV (USD)":f"{new_adtv_em:,.0f}",
        "DM ATVR (%)":f"{new_atvr_dm*100:.0f}%","EM ATVR (%)":f"{new_atvr_em*100:.0f}%",
        "Max Price (USD)":f"{max_closing_price:,.0f}" if max_closing_price else "—"}

    render_new_tab("Global Sort", _gs_final, large_thr, mid_thr,
        china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor, _gs_params)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Per Country (Solactive)
# ══════════════════════════════════════════════════════════════════════════════
with tab_pc:
    st.markdown("## 🗺️ Per Country (Solactive)")
    st.caption("Primary only + Secondary Schritt 8 | Post-Liquiditätsfilter | Sortierung per Mapping Country")

    _pc_u = build_new_universe(df_raw_original, country_cls, thailand_sec_type, max_closing_price,
        exclude_hk_cny, exclude_country_risk_na, exclude_naics_funds, exclude_euro_mtf, exclude_etf_sicav,
        china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)

    _pc_liq = apply_liquidity_new(_pc_u, new_adtv_dm, new_adtv_em, new_atvr_dm, new_atvr_em)

    _pc_seg = assign_segments_new(_pc_liq, large_thr, mid_thr, small_thr,
        group_col="Mapping Country", sort_col="Adj_FF_MCap")

    _pc_final = add_step8_secondary_new(_pc_seg, df_raw_original, new_adtv_dm, new_adtv_em,
        new_atvr_dm, new_atvr_em, max_closing_price, thailand_sec_type,
        china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)

    _pc_tot_adj = _pc_final["Adj_FF_MCap"].sum()
    _pc_final["Index_Weight"] = _pc_final["Adj_FF_MCap"]/_pc_tot_adj*100 if _pc_tot_adj>0 else 0

    _pc_params = {"Methodik":"Per Country (Solactive)","Listing":"Primary only + Secondary Schritt 8",
        "Filter":"Post","Large Cap (%)":large_thr,"Mid Cap (%)":mid_thr,"Small Cap (%)":small_thr,
        "DM ADTV (USD)":f"{new_adtv_dm:,.0f}","EM ADTV (USD)":f"{new_adtv_em:,.0f}",
        "DM ATVR (%)":f"{new_atvr_dm*100:.0f}%","EM ATVR (%)":f"{new_atvr_em*100:.0f}%",
        "Max Price (USD)":f"{max_closing_price:,.0f}" if max_closing_price else "—"}

    render_new_tab("Per Country (Solactive)", _pc_final, large_thr, mid_thr,
        china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor, _pc_params)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: GIMI Method
# ══════════════════════════════════════════════════════════════════════════════
with tab_gimi:
    st.markdown("## ⚡ GIMI Method")
    st.caption("Primary only + Secondary Schritt 8 | EUMSS Pre-Filter | Liquidität Pre-Filter | Coverage per Land auf Adj_FF_MCap")

    _gm_u = build_new_universe(df_raw_original, country_cls, thailand_sec_type, max_closing_price,
        exclude_hk_cny, exclude_country_risk_na, exclude_naics_funds, exclude_euro_mtf, exclude_etf_sicav,
        china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)

    # EUMSS calibration on DM (using small_thr = 99%)
    _gm_dm_all = _gm_u[_gm_u["Classification"]=="DM"].sort_values("Total MCap Y2025", ascending=False).copy()
    _gm_ff_tot = _gm_dm_all["Free Float MCap Y2025"].sum()
    if _gm_ff_tot > 0:
        _gm_dm_all["_cp"] = _gm_dm_all["Free Float MCap Y2025"].cumsum() / _gm_ff_tot * 100
        _gm_eumss_rows = _gm_dm_all[_gm_dm_all["_cp"] >= small_thr]
        if len(_gm_eumss_rows) == 0:
            st.error("EUMSS konnte nicht kalibriert werden.")
            st.stop()
        _gm_eumss_full = _gm_eumss_rows.iloc[0]["Total MCap Y2025"]
        _gm_eumss_ff   = _gm_eumss_full * new_eumss_ff_ratio

        # EUMSS filter
        _gm_mask_eumss = ((_gm_u["Total MCap Y2025"] >= _gm_eumss_full) &
                          (_gm_u["Free Float MCap Y2025"] >= _gm_eumss_ff) &
                          (_gm_u["Free Float Percent"] >= min_ff_pct))
        _gm_eumss = _gm_u[_gm_mask_eumss].copy()

        # Pre-liquidity filter
        _gm_liq = apply_liquidity_new(_gm_eumss, new_adtv_dm, new_adtv_em, new_atvr_dm, new_atvr_em)

        # Coverage per country on Adj_FF_MCap → Standard Index
        _gm_results = []
        for _ctry, _grp in _gm_liq.groupby("Mapping Country"):
            _grp = _grp.sort_values("Adj_FF_MCap", ascending=False).copy()
            _tot = _grp["Adj_FF_MCap"].sum()
            if _tot == 0: continue
            _grp["_c"] = _grp["Adj_FF_MCap"].cumsum() / _tot * 100
            _cut = _grp[_grp["_c"] >= mid_thr].index
            _inc = _grp.loc[:_cut[0]] if len(_cut)>0 else _grp
            _tot_inc = _inc["Adj_FF_MCap"].sum()
            _inc = _inc.copy()
            _inc["_cp2"] = _inc["Adj_FF_MCap"].cumsum() / _tot_inc * 100 if _tot_inc>0 else 0
            _inc["Segment_New"] = np.where(_inc["_cp2"] <= large_thr, "Large Cap", "Mid Cap")
            _gm_results.append(_inc)

        _gm_std = pd.concat(_gm_results, ignore_index=True) if _gm_results else pd.DataFrame(columns=_gm_liq.columns.tolist()+["Segment_New"])

        # Small Cap = passed EUMSS but not in liquidity filter or not in 85%
        _gm_small = _gm_eumss[~_gm_eumss.index.isin(_gm_liq.index)].copy()
        _gm_small["Segment_New"] = "Small Cap"
        # Also add stocks in liquidity but above 85% cutoff
        _gm_above85 = _gm_liq[~_gm_liq.index.isin(_gm_std.index)].copy()
        _gm_above85["Segment_New"] = "Small Cap"

        # Micro Cap = below EUMSS
        _gm_micro = _gm_u[~_gm_u.index.isin(_gm_eumss.index)].copy()
        _gm_micro["Segment_New"] = "Micro Cap"

        # Step 8: Secondaries for standard index only
        _gm_final = add_step8_secondary_new(_gm_std, df_raw_original, new_adtv_dm, new_adtv_em,
            new_atvr_dm, new_atvr_em, max_closing_price, thailand_sec_type,
            china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor)

        _gm_complete = pd.concat([_gm_final, _gm_small, _gm_above85, _gm_micro], ignore_index=True)
        _gm_complete = _gm_complete.drop_duplicates(subset=["Symbol"]).copy()

        _gm_tot_adj = _gm_complete["Adj_FF_MCap"].sum()
        _gm_complete["Index_Weight"] = _gm_complete["Adj_FF_MCap"]/_gm_tot_adj*100 if _gm_tot_adj>0 else 0

        # Pipeline Diagnostik
        with st.expander("🔍 Pipeline Diagnostik", expanded=False):
            _gm_diag = [
                {"Schritt":"1 — Universe (Primary only)","DM":(_gm_u["Classification"]=="DM").sum(),"EM":(_gm_u["Classification"]=="EM").sum(),"Total":len(_gm_u),"Exkl.":"—"},
                {"Schritt":f"2 — EUMSS Filter ({_gm_eumss_full/1e6:.0f}M)","DM":(_gm_eumss["Classification"]=="DM").sum(),"EM":(_gm_eumss["Classification"]=="EM").sum(),"Total":len(_gm_eumss),"Exkl.":f"-{len(_gm_u)-len(_gm_eumss):,}"},
                {"Schritt":"3 — Liquidität (Pre)","DM":(_gm_liq["Classification"]=="DM").sum(),"EM":(_gm_liq["Classification"]=="EM").sum(),"Total":len(_gm_liq),"Exkl.":f"-{len(_gm_eumss)-len(_gm_liq):,}"},
                {"Schritt":f"4 — {mid_thr}% Coverage","DM":(_gm_std["Classification"]=="DM").sum(),"EM":(_gm_std["Classification"]=="EM").sum(),"Total":len(_gm_std),"Exkl.":f"-{len(_gm_liq)-len(_gm_std):,}"},
                {"Schritt":"5 — Secondary Schritt 8 (+)","DM":(_gm_final["Classification"]=="DM").sum(),"EM":(_gm_final["Classification"]=="EM").sum(),"Total":len(_gm_final),"Exkl.":f"+{len(_gm_final)-len(_gm_std):,}"},
            ]
            st.dataframe(pd.DataFrame(_gm_diag), use_container_width=True, hide_index=True)
            st.caption(f"EUMSS_FULL: {format_bn(_gm_eumss_full)} | EUMSS_FF: {format_bn(_gm_eumss_ff)} | FF Ratio: {new_eumss_ff_ratio*100:.0f}% | Min FF%: {min_ff_pct*100:.0f}%")

        _gm_params = {"Methodik":"GIMI Method","Listing":"Primary only + Secondary Schritt 8",
            "Filter":"Pre (nach EUMSS)","EUMSS Kalibrierung (%)":f"{small_thr}%",
            "EUMSS_FULL (USD)":format_bn(_gm_eumss_full),"EUMSS FF Ratio (%)":f"{new_eumss_ff_ratio*100:.0f}%",
            "EUMSS_FF (USD)":format_bn(_gm_eumss_ff),"Min FF%":f"{min_ff_pct*100:.0f}%",
            "Coverage (%)":f"{mid_thr}%","Large Cap (%)":large_thr,
            "DM ADTV (USD)":f"{new_adtv_dm:,.0f}","EM ADTV (USD)":f"{new_adtv_em:,.0f}",
            "DM ATVR (%)":f"{new_atvr_dm*100:.0f}%","EM ATVR (%)":f"{new_atvr_em*100:.0f}%",
            "Max Price (USD)":f"{max_closing_price:,.0f}" if max_closing_price else "—"}

        render_new_tab("GIMI Method", _gm_complete, large_thr, mid_thr,
            china_inclusion_factor, india_inclusion_factor, vietnam_inclusion_factor, saudi_inclusion_factor, _gm_params)
    else:
        st.error("Keine DM Stocks gefunden.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: ACWI & Varianten Vergleich
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("## 🔀 ACWI & Varianten Vergleich")
    st.caption("Vergleich aller vier Methoden: ACWI (V1+Threshold) | Global Sort | Per Country | GIMI")

    try:
        def _vstats(df, name, listing, filt):
            std = df[df["Segment_New"].isin(["Large Cap","Mid Cap"])] if "Segment_New" in df.columns else df[df["Status"]=="Included"] if "Status" in df.columns else df
            if "Segment_New" not in df.columns and "Status" in df.columns:
                # Tab 2 ACWI format
                dm_std = df[df["Segment"].isin(["Large Cap","Mid Cap"])] if "Segment" in df.columns else pd.DataFrame()
                em_std = df_acwi_em
                std = pd.concat([dm_std, em_std], ignore_index=True)
            dm_s = std[std["Classification"]=="DM"] if "Classification" in std.columns else pd.DataFrame()
            em_s = std[std["Classification"]=="EM"] if "Classification" in std.columns else pd.DataFrame()

            adj_col = "Adj_FF_MCap" if "Adj_FF_MCap" in std.columns else "Adjusted FF MCap"
            tot_adj = std[adj_col].sum() if adj_col in std.columns else 0
            em_adj  = em_s[adj_col].sum() if adj_col in em_s.columns and len(em_s)>0 else 0
            em_w    = em_adj/tot_adj*100 if tot_adj>0 else 0

            ecn_col = "Exchange Country Name"
            india_mask = std[ecn_col].fillna("").str.upper()=="INDIA" if ecn_col in std.columns else pd.Series([False]*len(std))
            india = std[india_mask]
            india_adj = india[adj_col].sum() if adj_col in india.columns and len(india)>0 else 0
            india_em_w = india_adj/em_adj*100 if em_adj>0 else 0

            # IMI
            if "Segment_New" in df.columns:
                dm_imi = df[(df["Classification"]=="DM") & df["Segment_New"].isin(["Large Cap","Mid Cap","Small Cap"])]
                acwi_imi = df[df["Segment_New"].isin(["Large Cap","Mid Cap","Small Cap"])]
            else:
                dm_imi = dm_s; acwi_imi = std

            return {
                "Methodik": name, "Listing": listing, "Filter": filt,
                "World Index": len(dm_s),
                "EM Index": len(em_s),
                "ACWI": len(std),
                "World IMI": len(dm_imi),
                "ACWI IMI": len(acwi_imi),
                "EM Weight %": f"{em_w:.2f}%",
                "Indien Stocks": len(india),
                "Indien EM %": f"{india_em_w:.2f}%",
            }

        # Build ACWI row from Tab 2
        _acwi_combined = pd.concat([df_acwi_dm, df_acwi_em], ignore_index=True)
        if "Adjusted FF MCap" in _acwi_combined.columns:
            _acwi_combined = _acwi_combined.rename(columns={"Adjusted FF MCap":"Adj_FF_MCap"})
        _acwi_combined["Segment_New"] = np.where(
            _acwi_combined.get("Segment","") == "Large Cap", "Large Cap",
            np.where(_acwi_combined.get("Segment","") == "Mid Cap", "Mid Cap", "Large Cap"))

        _cmp_rows = [
            _vstats(_acwi_combined,  "ACWI (V1 + Threshold)", "All", "Post"),
            _vstats(_gs_final,  "Global Sort (Threshold)",  "Primary only", "Post"),
            _vstats(_pc_final,  "Per Country (Solactive)",  "Primary only", "Post"),
            _vstats(_gm_complete, "GIMI Method",            "Primary only", "Pre"),
        ]
        _cmp_df = pd.DataFrame(_cmp_rows)
        st.dataframe(_cmp_df, use_container_width=True, hide_index=True)

        # Country weight comparison
        st.markdown("---")
        st.markdown("**Top 15 Länder — ACWI Gewicht % im Vergleich (Tabs 3–5)**")

        def _cw(df, name):
            std = df[df["Segment_New"].isin(["Large Cap","Mid Cap"])] if "Segment_New" in df.columns else df
            adj = "Adj_FF_MCap" if "Adj_FF_MCap" in std.columns else "Adjusted FF MCap"
            tot = std[adj].sum()
            ct  = std.groupby("Mapping Country")[adj].sum().reset_index()
            ct["Weight%"] = ct[adj]/tot*100 if tot>0 else 0
            ct["Methodik"] = name
            return ct.rename(columns={adj:"Adj"})

        _all_cw = pd.concat([
            _cw(_gs_final,    "Global Sort"),
            _cw(_pc_final,    "Per Country"),
            _cw(_gm_complete, "GIMI"),
        ])
        _top15 = _all_cw.groupby("Mapping Country")["Adj"].sum().nlargest(15).index.tolist()
        _all_cw_top = _all_cw[_all_cw["Mapping Country"].isin(_top15)]

        fig_cmp = px.bar(_all_cw_top, x="Mapping Country", y="Weight%", color="Methodik",
            barmode="group", template="plotly_dark",
            color_discrete_map={"Global Sort":"#2979ff","Per Country":"#66bb6a","GIMI":"#ce93d8"})
        fig_cmp.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#161b27",
            height=450, margin=dict(t=10,b=10))
        st.plotly_chart(fig_cmp, use_container_width=True)

    except Exception as _e:
        st.warning(f"⚠️ Bitte zuerst alle Tabs aufrufen damit alle Varianten berechnet werden. ({_e})")
