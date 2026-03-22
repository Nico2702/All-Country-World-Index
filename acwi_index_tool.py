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
    """Global DM threshold approach."""
    df = df_dm.sort_values("Free Float MCap Y2025", ascending=False).copy()
    total_ff = df["Free Float MCap Y2025"].sum()
    df["cum_ff"] = df["Free Float MCap Y2025"].cumsum()
    df["cum_pct"] = df["cum_ff"] / total_ff * 100

    # Shift: segment is determined by where cumulative pct crosses the threshold
    # A stock belongs to Large if its ENDING cumulative % is <= large_pct threshold
    # Mid: ending cum% between large and mid thresholds
    # Small: ending cum% between mid and small thresholds
    df["prev_cum_pct"] = df["cum_pct"].shift(1, fill_value=0)

    def assign_segment(row):
        if row["prev_cum_pct"] < large_pct:
            return "Large Cap"
        elif row["prev_cum_pct"] < mid_pct:
            return "Mid Cap"
        elif row["prev_cum_pct"] < small_pct:
            return "Small Cap"
        else:
            return "Micro / Excluded"

    df["Segment"] = df.apply(assign_segment, axis=1)
    return df


def compute_variant2(df_dm, large_pct, mid_pct, small_pct):
    """Per-country threshold approach."""
    results = []
    for country, grp in df_dm.groupby("Exchange Country Name"):
        grp = grp.sort_values("Free Float MCap Y2025", ascending=False).copy()
        total_ff = grp["Free Float MCap Y2025"].sum()
        if total_ff == 0:
            grp["Segment"] = "Micro / Excluded"
            grp["cum_ff"] = 0
            grp["cum_pct"] = 0
            results.append(grp)
            continue
        grp["cum_ff"] = grp["Free Float MCap Y2025"].cumsum()
        grp["cum_pct"] = grp["cum_ff"] / total_ff * 100
        grp["prev_cum_pct"] = grp["cum_pct"].shift(1, fill_value=0)

        def assign_segment(row):
            if row["prev_cum_pct"] < large_pct:
                return "Large Cap"
            elif row["prev_cum_pct"] < mid_pct:
                return "Mid Cap"
            elif row["prev_cum_pct"] < small_pct:
                return "Small Cap"
            else:
                return "Micro / Excluded"

        grp["Segment"] = grp.apply(assign_segment, axis=1)
        results.append(grp)
    return pd.concat(results, ignore_index=True)


def build_em_universe(df_em, large_pct, mid_pct, small_pct, mode="global"):
    """Build EM segments (same logic, global or per-country)."""
    if mode == "global":
        return compute_variant1(df_em, large_pct, mid_pct, small_pct)
    else:
        return compute_variant2(df_em, large_pct, mid_pct, small_pct)


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
    df["Free Float MCap Y2025"] = pd.to_numeric(df["Free Float MCap Y2025"], errors="coerce").fillna(0)
    df["Total MCap Y2025"] = pd.to_numeric(df["Total MCap Y2025"], errors="coerce").fillna(0)
    return df


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Data Source")
    uploaded = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    use_default = st.checkbox("Use default file (Framework_NaroIX)", value=True)

    st.markdown("---")
    st.markdown("### ⚙️ Index Parameters")

    listing_filter = st.radio("Listing Type", ["Primary only", "All (Primary + Secondary)"], index=0)

    st.markdown("**DM Percentile Thresholds**")
    large_thr = st.slider("Large Cap cutoff (%)", 50, 80, 70, 1,
        help="Cumulative FF MCap threshold for Large Cap inclusion")
    mid_thr   = st.slider("Mid Cap cutoff (%)",   large_thr+1, 95, 85, 1,
        help="Cumulative FF MCap threshold for Mid Cap inclusion (= World Index boundary)")
    small_thr = st.slider("Small Cap cutoff (%)", mid_thr+1, 100, 99, 1,
        help="Cumulative FF MCap threshold for Small Cap inclusion")

    st.markdown("**EM Parameters**")
    em_mode = st.radio("EM Segment Method", ["Global (like Variant 1)", "Per-Country (like Variant 2)"], index=0)
    em_large  = st.slider("EM Large Cap cutoff (%)", 50, 80, 70, 1)
    em_mid    = st.slider("EM Mid Cap cutoff (%)", em_large+1, 95, 85, 1)
    em_small  = st.slider("EM Small Cap cutoff (%)", em_mid+1, 100, 99, 1)

    st.markdown("---")
    st.markdown("<div style='color:#8892b0;font-size:11px;'>NaroIX Index Construction Tool<br/>© 2025 NaroIX</div>", unsafe_allow_html=True)


# ─── Load Data ─────────────────────────────────────────────────────────────────
if uploaded:
    df_raw = load_excel(uploaded)
elif use_default:
    try:
        df_raw = load_excel("/mnt/user-data/uploads/Framework_NaroIX_Year-by-Year_2.xlsx")
    except:
        st.error("Default file not found. Please upload an Excel file.")
        st.stop()
else:
    st.info("Please upload an Excel file or check 'Use default file'.")
    st.stop()

# Apply listing filter
if listing_filter == "Primary only":
    df_raw = df_raw[df_raw["Listing"] == "Primary"].copy()

df_dm = df_raw[df_raw["Classification"] == "DM"].copy()
df_em = df_raw[df_raw["Classification"] == "EM"].copy()


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

st.markdown("---")

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_v1, tab_v2, tab_acwi, tab_compare = st.tabs([
    "🌍 Universe Overview",
    "📐 Variant 1 — Global DM Thresholds",
    "🗺️ Variant 2 — Per-Country DM Thresholds",
    "🌐 ACWI (DM + EM)",
    "⚖️ Variant Comparison",
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: VARIANT 1 — GLOBAL DM THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════
with tab_v1:
    st.subheader("Variant 1 — Global DM Thresholds")
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

    df_v1 = compute_variant1(df_dm, large_thr, mid_thr, small_thr)

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

        world_idx = df_v1[df_v1["Segment"].isin(["Large Cap","Mid Cap"])]
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
    export_v1 = df_v1[df_v1["Segment"] != "Micro / Excluded"].copy()
    export_v1["cum_pct"] = export_v1["cum_pct"].round(4)
    st.download_button(
        "⬇️ Download World Index (Variant 1) as Excel",
        data=to_excel_download(export_v1.drop(columns=["cum_ff","prev_cum_pct"]), "World_V1"),
        file_name="NaroIX_World_Variant1.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: VARIANT 2 — PER-COUNTRY DM THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════
with tab_v2:
    st.subheader("Variant 2 — Per-Country DM Thresholds")
    st.markdown(f"""
    <div class="info-box">
    <b>Methodik:</b> Das Prozedere aus Variant 1 wird für <b>jedes DM-Land separat</b> angewendet.
    Innerhalb jedes Landes werden die Aktien nach FF MCap sortiert und die Segmente auf Basis der
    länderspezifischen kumulierten FF MCap-Anteile bestimmt.<br>
    • <b>World Index (DM)</b> = Large Cap + Mid Cap je Land (bis {mid_thr}% kumuliert pro Land)
    </div>
    """, unsafe_allow_html=True)

    df_v2 = compute_variant2(df_dm, large_thr, mid_thr, small_thr)

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

        world_v2 = df_v2[df_v2["Segment"].isin(["Large Cap","Mid Cap"])]
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

    export_v2 = df_v2[df_v2["Segment"] != "Micro / Excluded"].copy()
    export_v2["cum_pct"] = export_v2["cum_pct"].round(4)
    st.download_button(
        "⬇️ Download World Index (Variant 2) as Excel",
        data=to_excel_download(export_v2.drop(columns=["cum_ff","prev_cum_pct"]), "World_V2"),
        file_name="NaroIX_World_Variant2.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: ACWI (DM + EM)
# ══════════════════════════════════════════════════════════════════════════════
with tab_acwi:
    st.subheader("ACWI — All Country World Index (DM + EM)")

    acwi_variant = st.radio("DM Methodik für ACWI:", ["Variant 1 (Global)", "Variant 2 (Per-Country)"], horizontal=True)
    em_method_str = "global" if "Global" in em_mode else "per-country"

    if acwi_variant == "Variant 1 (Global)":
        df_dm_seg = compute_variant1(df_dm, large_thr, mid_thr, small_thr)
    else:
        df_dm_seg = compute_variant2(df_dm, large_thr, mid_thr, small_thr)

    df_em_seg = build_em_universe(df_em, em_large, em_mid, em_small, mode=em_method_str)

    df_acwi_dm = df_dm_seg[df_dm_seg["Segment"] != "Micro / Excluded"].copy()
    df_acwi_em = df_em_seg[df_em_seg["Segment"] != "Micro / Excluded"].copy()

    total_acwi = len(df_acwi_dm) + len(df_acwi_em)
    total_ff_acwi = df_acwi_dm["Free Float MCap Y2025"].sum() + df_acwi_em["Free Float MCap Y2025"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ACWI Total Stocks", f"{total_acwi:,}")
    c2.metric("ACWI FF MCap", format_bn(total_ff_acwi))
    c3.metric("DM Weight", f"{df_acwi_dm['Free Float MCap Y2025'].sum()/total_ff_acwi*100:.1f}%")
    c4.metric("EM Weight", f"{df_acwi_em['Free Float MCap Y2025'].sum()/total_ff_acwi*100:.1f}%")

    col_dm, col_em = st.columns(2)

    with col_dm:
        st.markdown("**DM Segments**")
        s = segment_summary(df_dm_seg)
        s["FF MCap (USD)"] = s["FF MCap (USD)"].apply(format_bn)
        s["Avg FF MCap (USD)"] = s["Avg FF MCap (USD)"].apply(format_bn)
        st.dataframe(s, use_container_width=True, hide_index=True)

    with col_em:
        st.markdown("**EM Segments**")
        s2 = segment_summary(df_em_seg)
        s2["FF MCap (USD)"] = s2["FF MCap (USD)"].apply(format_bn)
        s2["Avg FF MCap (USD)"] = s2["Avg FF MCap (USD)"].apply(format_bn)
        st.dataframe(s2, use_container_width=True, hide_index=True)

    # Combined donut
    st.markdown("**ACWI Composition**")
    acwi_summary = []
    for seg in ["Large Cap","Mid Cap","Small Cap"]:
        dm_ff = df_acwi_dm[df_acwi_dm["Segment"]==seg]["Free Float MCap Y2025"].sum()
        em_ff = df_acwi_em[df_acwi_em["Segment"]==seg]["Free Float MCap Y2025"].sum()
        if dm_ff > 0: acwi_summary.append({"Label":f"DM {seg}","FF MCap":dm_ff,"Type":"DM"})
        if em_ff > 0: acwi_summary.append({"Label":f"EM {seg}","FF MCap":em_ff,"Type":"EM"})
    acwi_df_summary = pd.DataFrame(acwi_summary)

    fig_acwi = px.pie(
        acwi_df_summary, names="Label", values="FF MCap",
        color="Type", color_discrete_map={"DM":"#2979ff","EM":"#ce93d8"},
        template="plotly_dark", hole=0.45,
    )
    fig_acwi.update_layout(paper_bgcolor="#0f1117", height=350, margin=dict(t=10,b=10))
    st.plotly_chart(fig_acwi, use_container_width=True)

    # Full ACWI download
    df_acwi_export = pd.concat([df_acwi_dm, df_acwi_em], ignore_index=True)
    st.download_button(
        "⬇️ Download Full ACWI Constituents as Excel",
        data=to_excel_download(
            df_acwi_export[[c for c in df_acwi_export.columns if c not in ["cum_ff","prev_cum_pct"]]], "ACWI"),
        file_name="NaroIX_ACWI.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("⚖️ Variant 1 vs. Variant 2 — Direct Comparison")

    df_v1_c = compute_variant1(df_dm, large_thr, mid_thr, small_thr)
    df_v2_c = compute_variant2(df_dm, large_thr, mid_thr, small_thr)

    world_v1 = df_v1_c[df_v1_c["Segment"].isin(["Large Cap","Mid Cap"])]
    world_v2 = df_v2_c[df_v2_c["Segment"].isin(["Large Cap","Mid Cap"])]

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
