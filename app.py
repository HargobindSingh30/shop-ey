import streamlit as st
import json
import numpy as np
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Market Intelligence and Investment Prioritization Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.main .block-container { padding-top: 2rem; max-width: 1200px; }

.hero-header {
    background: linear-gradient(135deg, #1C2340 0%, #2C3E6B 100%);
    padding: 2.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.hero-header h1 { color: white; font-size: 2.2rem; font-weight: 700; margin-bottom: 0.3rem; }
.hero-header p { color: #8899AA; font-size: 1rem; margin: 0; }
.gold-line { width: 60px; height: 3px; background: #C5A55A; margin: 1rem 0; }

.score-card {
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    border: 2px solid;
}
.score-invest { background: #EBF5F0; border-color: #0B6E4F; }
.score-monitor { background: #EAF0F6; border-color: #1A5276; }
.score-exit { background: #F5EAEA; border-color: #922B21; }

.score-number { font-size: 3.5rem; font-weight: 700; line-height: 1; }
.score-label { font-size: 1rem; color: #5D6D7E; margin-top: 0.3rem; }
.rec-badge {
    display: inline-block;
    padding: 0.3rem 1.2rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

.metric-card {
    background: #F8F9FA;
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid #D5D8DC;
    margin-bottom: 0.5rem;
}
.metric-title { font-weight: 600; color: #1C2340; font-size: 0.85rem; }
.metric-value { color: #5D6D7E; font-size: 0.8rem; }

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1C2340;
    border-bottom: 2px solid #C5A55A;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    display: inline-block;
}

.driver-positive { color: #0B6E4F; font-weight: 600; }
.driver-negative { color: #922B21; font-weight: 600; }

.company-local { 
    display: inline-block; 
    width: 8px; height: 8px; 
    background: #C5A55A; 
    border-radius: 50%; 
    margin-right: 6px; 
}

.footer-bar {
    background: #1C2340;
    padding: 0.8rem 1.5rem;
    border-radius: 8px;
    color: #667788;
    font-size: 0.75rem;
    text-align: center;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL DATA
# ============================================================
@st.cache_data
def load_model():
    data_path = os.path.join(os.path.dirname(__file__), "model_data.json")
    with open(data_path) as f:
        return json.load(f)

MODEL = load_model()

FIELDS = [
    ("internet_users_pct_latest", "Internet Users (%)", "IT.NET.USER.ZS"),
    ("internet_users_pct_trend", "Internet Growth Rate", "Trend"),
    ("account_ownership_pct_adult_latest", "Account Ownership (%)", "FX.OWN.TOTL.ZS"),
    ("account_ownership_pct_adult_trend", "Account Ownership Growth", "Trend"),
    ("mobile_subs_per_100_latest", "Mobile Subs per 100", "IT.CEL.SETS.P2"),
    ("mobile_subs_per_100_trend", "Mobile Growth Rate", "Trend"),
    ("gdp_per_capita_usd_latest", "GDP per Capita (USD)", "NY.GDP.PCAP.CD"),
    ("gdp_per_capita_usd_trend", "GDP per Capita Growth", "Trend"),
    ("urban_population_pct_latest", "Urbanization (%)", "SP.URB.TOTL.IN.ZS"),
    ("urban_population_pct_trend", "Urbanization Growth", "Trend"),
    ("population_total_latest", "Population", "SP.POP.TOTL"),
    ("population_total_trend", "Population Growth", "Trend"),
    ("gdp_total_usd_latest", "GDP Total (USD)", "NY.GDP.MKTP.CD"),
    ("gdp_total_usd_trend", "GDP Growth Rate", "Trend"),
    ("fdi_inflow_pct_gdp_latest", "FDI Inflow (% GDP)", "BX.KLT.DINV.WD.GD.ZS"),
    ("fdi_inflow_pct_gdp_trend", "FDI Growth Rate", "Trend"),
]

MEXICO_PRESET = {
    "internet_users_pct_latest": 75.0, "internet_users_pct_trend": 0.08,
    "account_ownership_pct_adult_latest": 49.0, "account_ownership_pct_adult_trend": 0.12,
    "mobile_subs_per_100_latest": 95.0, "mobile_subs_per_100_trend": 0.01,
    "gdp_per_capita_usd_latest": 11500.0, "gdp_per_capita_usd_trend": 0.03,
    "urban_population_pct_latest": 81.0, "urban_population_pct_trend": 0.005,
    "population_total_latest": 130000000.0, "population_total_trend": 0.01,
    "gdp_total_usd_latest": 1.5e12, "gdp_total_usd_trend": 0.04,
    "fdi_inflow_pct_gdp_latest": 2.5, "fdi_inflow_pct_gdp_trend": 0.01,
}

# ============================================================
# PREDICTION ENGINE
# ============================================================
def predict_market(macro_input):
    # Normalize input
    input_normed = {}
    for col in MODEL["macro_features"]:
        val = macro_input.get(col, 0)
        r = MODEL["norm_ranges"].get(col, {})
        mn, mx = r.get("min", 0), r.get("max", 1)
        if mx != mn:
            input_normed[col] = max(0, min(1, (val - mn) / (mx - mn)))
        else:
            input_normed[col] = 0.5

    input_vector = [input_normed[col] for col in MODEL["macro_features"]]

    # Find similar markets
    distances = []
    for c in MODEL["countries"]:
        cv = []
        for col in MODEL["macro_features"]:
            r = MODEL["norm_ranges"].get(col, {})
            mn, mx = r.get("min", 0), r.get("max", 1)
            if mx != mn:
                cv.append(max(0, min(1, (c["macro"][col] - mn) / (mx - mn))))
            else:
                cv.append(0.5)
        dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(input_vector, cv)))
        distances.append({
            "country": c["country"], "region": c["region"],
            "distance": round(dist, 4),
            "score": c["composite_score"],
            "recommendation": c["recommendation"]
        })
    distances.sort(key=lambda x: x["distance"])
    similar = distances[:3]
    sim_countries = [m["country"] for m in similar]

    # Score Consumer Adoption directly
    ca_weights = MODEL["weights"]["consumer_adoption"]
    ca_score = 0
    for col, w in ca_weights.items():
        if col in input_normed:
            nv = input_normed[col]
            if MODEL["feature_direction"].get(col, 1) == -1:
                nv = 1 - nv
            ca_score += nv * w

    # Estimate MD, OE, GO from similar markets
    total_inv = sum(1 / (m["distance"] + 0.001) for m in similar)
    wts = [1 / (m["distance"] + 0.001) / total_inv for m in similar]

    def wavg(field):
        vals = []
        for m in similar:
            c = next((x for x in MODEL["countries"] if x["country"] == m["country"]), None)
            vals.append(c[field] if c else 0)
        return sum(v * w for v, w in zip(vals, wts))

    scores = {
        "Consumer Adoption": round(ca_score * 10, 2),
        "Growth Opportunity": round(wavg("growth_opportunity_score"), 2),
        "Market Dynamics": round(wavg("market_dynamics_score"), 2),
        "Operational Efficiency": round(wavg("operational_efficiency_score"), 2),
    }

    composite = round(np.mean(list(scores.values())), 2)

    if composite >= MODEL["thresholds"]["invest"]:
        rec = "Invest"
    elif composite >= MODEL["thresholds"]["exit"]:
        rec = "Monitor"
    else:
        rec = "Exit"

    # Key drivers
    drivers = []
    for col in MODEL["macro_features"]:
        nv = input_normed[col]
        d = MODEL["feature_direction"].get(col, 1)
        if d == -1:
            nv = 1 - nv
        max_w = max(
            MODEL["weights"]["consumer_adoption"].get(col, 0),
            MODEL["weights"]["growth_opportunity"].get(col, 0),
            MODEL["weights"]["market_dynamics"].get(col, 0),
            MODEL["weights"]["operational_efficiency"].get(col, 0),
        )
        drivers.append({
            "feature": MODEL["macro_labels"].get(col, col),
            "contribution": round(nv * max_w, 4),
            "normalized": nv,
        })
    drivers.sort(key=lambda x: -x["contribution"])
    pos_drivers = [d for d in drivers if d["normalized"] > 0.5][:4]
    neg_drivers = [d for d in drivers if d["normalized"] <= 0.5][:3]

    # Reference companies
    refs = [c for c in MODEL["companies"] if c["country"] in sim_countries]
    refs.sort(key=lambda x: -x["revenue_cagr"])
    refs = refs[:10]
    if len(refs) < 5:
        extra = [c for c in MODEL["companies"] if c["country"] not in sim_countries]
        extra.sort(key=lambda x: -x["revenue_cagr"])
        refs.extend(extra[:8 - len(refs)])

    # Expected performance
    cagrs = sorted([c["revenue_cagr"] for c in refs])
    margins = sorted([c["net_margin"] for c in refs])
    vols = sorted([c["volatility"] for c in refs])
    q = lambda arr, p: arr[int(len(arr) * p)] if arr else 0

    perf = {
        "Revenue CAGR": f"{q(cagrs, 0.25)*100:.1f}% to {q(cagrs, 0.75)*100:.1f}%",
        "Net Margin": f"{q(margins, 0.25)*100:.1f}% to {q(margins, 0.75)*100:.1f}%",
        "Volatility": f"{q(vols, 0.5)*100:.1f}%",
        "Profitable": f"{sum(1 for c in refs if c['net_margin'] > 0) / len(refs) * 100:.0f}%",
    }

    return {
        "scores": scores, "composite": composite, "recommendation": rec,
        "similar": similar, "pos_drivers": pos_drivers, "neg_drivers": neg_drivers,
        "refs": refs, "perf": perf, "sim_countries": sim_countries, "macro": macro_input,
    }


def get_expert_advice(result, api_key):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        pos_str = ", ".join([f"{d['feature']} ({d['contribution']:.3f})" for d in result["pos_drivers"]])
        neg_str = ", ".join([f"{d['feature']} ({d['contribution']:.3f})" for d in result["neg_drivers"]])
        ref_str = json.dumps([{
            "name": c["company_name"], "country": c["country"],
            "revenue_B": round(c["revenue_usd"] / 1e9, 1),
            "cagr": f"{c['revenue_cagr']*100:.1f}%",
            "margin": f"{c['net_margin']*100:.1f}%",
            "archetype": c["archetype"],
            "local": c["country"] in result["sim_countries"]
        } for c in result["refs"]])

        prompt = (
            "You are a senior strategy consultant at EY Data & AI. "
            "Shop-EY is a global retailer (household/everyday consumer products) evaluating a new market.\n\n"
            f"SCORE: {result['composite']}/10 | REC: {result['recommendation']}\n"
            f"LENS: {json.dumps(result['scores'])}\n"
            f"Invest threshold: {MODEL['thresholds']['invest']:.2f} | Exit threshold: {MODEL['thresholds']['exit']:.2f}\n"
            f"POSITIVE DRIVERS: {pos_str}\n"
            f"CONSTRAINTS: {neg_str}\n"
            f"PERFORMANCE: {json.dumps(result['perf'])}\n"
            f"SIMILAR MARKETS: {json.dumps(result['similar'])}\n"
            f"REFERENCE COMPANIES: {ref_str}\n"
            f"CONTEXT:\n{MODEL['knowledge_base']}\n\n"
            "Write a 300-400 word strategic brief in plain paragraphs (no markdown/bullets/formatting).\n"
            "Sections: ASSESSMENT (compare to best/worst markets), KEY DRIVERS (practical implications with numbers), "
            "COMPARABLE EVIDENCE (name companies), RISKS (specific, data-grounded), RECOMMENDED ACTIONS (reference specific companies). "
            "Every claim must cite a number, company, or market."
        )

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# UI
# ============================================================

# Hero
st.markdown("""
<div class="hero-header">
    <h1>Market Intelligence and Investment Prioritization Engine</h1>
    <div class="gold-line"></div>
    <p>Input the macro indicators of any market to receive an investment assessment,<br>
    powered by analysis of 43 e-commerce companies across 13 markets worldwide.</p>
</div>
""", unsafe_allow_html=True)

# Presets
st.markdown('<div class="section-header">Select a Market or Enter Custom Data</div>', unsafe_allow_html=True)

preset_options = ["Custom (enter manually)"]
preset_options += [f"{c['country']} ({c['region']})" for c in MODEL["countries"]]
preset_options.append("MX (New Market)")

selected_preset = st.selectbox("Preset", preset_options, label_visibility="collapsed")

# Determine values
preset_values = {}
if selected_preset == "Custom (enter manually)":
    preset_values = {key: 0.0 for key, _, _ in FIELDS}
elif selected_preset == "MX (New Market)":
    preset_values = MEXICO_PRESET
else:
    country_code = selected_preset.split(" ")[0]
    country_data = next((c for c in MODEL["countries"] if c["country"] == country_code), None)
    if country_data:
        preset_values = country_data["macro"]

# Input form
st.markdown('<div class="section-header">Macro Indicators</div>', unsafe_allow_html=True)
st.caption("All indicators available from [World Bank Open Data](https://data.worldbank.org)")

macro_input = {}
cols = st.columns(4)
for i, (key, label, wb) in enumerate(FIELDS):
    with cols[i % 4]:
        default = float(preset_values.get(key, 0.0))
        macro_input[key] = st.number_input(
            f"{label}",
            value=default,
            format="%.4f" if "trend" in key else "%.2f",
            help=f"World Bank: {wb}",
            key=f"input_{key}",
        )

# Analyze button
col_btn1, col_btn2, _ = st.columns([1, 1, 4])
with col_btn1:
    analyze = st.button("🔍 Analyze Market", type="primary", use_container_width=True)

# ============================================================
# RESULTS
# ============================================================
if analyze:
    if all(v == 0 for v in macro_input.values()):
        st.warning("Please select a preset market or enter values for the macro indicators before analyzing.")
    else:
        result = predict_market(macro_input)
        st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]
    rec = result["recommendation"]

    colors = {"Invest": "#0B6E4F", "Monitor": "#1A5276", "Exit": "#922B21"}
    bgs = {"Invest": "#EBF5F0", "Monitor": "#EAF0F6", "Exit": "#F5EAEA"}
    css_class = {"Invest": "score-invest", "Monitor": "score-monitor", "Exit": "score-exit"}

    st.markdown("---")

    # Score hero
    col_score, col_meta = st.columns([1, 2])
    with col_score:
        st.markdown(f"""
        <div class="score-card {css_class[rec]}">
            <div class="score-number" style="color: {colors[rec]}">{result['composite']}</div>
            <div class="score-label">out of 10</div>
            <div class="rec-badge" style="background: {bgs[rec]}; color: {colors[rec]}">{rec}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_meta:
        st.markdown("### Market Assessment")
        st.write(f"Invest threshold: {MODEL['thresholds']['invest']:.2f} | Exit threshold: {MODEL['thresholds']['exit']:.2f}")

        # Lens scores
        for lens, score in result["scores"].items():
            bar_color = "#0B6E4F" if score >= 6 else "#1A5276" if score >= 4 else "#922B21"
            tag = "direct" if lens == "Consumer Adoption" else "estimated"
            pct = min(score / 10 * 100, 100)
            st.markdown(f"""
            <div style="display:flex;align-items:center;margin-bottom:6px;">
                <span style="width:180px;font-size:0.85rem;color:#1C2340;font-weight:500;">{lens}</span>
                <div style="flex:1;height:8px;background:#E8E8E8;border-radius:4px;margin:0 10px;">
                    <div style="width:{pct}%;height:8px;background:{bar_color};border-radius:4px;"></div>
                </div>
                <span style="font-weight:700;color:{bar_color};width:40px;">{score}</span>
                <span style="font-size:0.7rem;color:#999;width:60px;">{tag}</span>
            </div>
            """, unsafe_allow_html=True)

    # Details grid
    col1, col2 = st.columns(2)

    with col1:
        # Key Drivers
        st.markdown('<div class="section-header">Key Drivers</div>', unsafe_allow_html=True)
        st.caption("Positive")
        for d in result["pos_drivers"]:
            st.markdown(f'<div class="metric-card"><span class="metric-title">{d["feature"]}</span> '
                        f'<span class="driver-positive" style="float:right;">+{d["contribution"]:.3f}</span></div>',
                        unsafe_allow_html=True)
        st.caption("Constraints")
        for d in result["neg_drivers"]:
            st.markdown(f'<div class="metric-card"><span class="metric-title">{d["feature"]}</span> '
                        f'<span class="driver-negative" style="float:right;">{d["contribution"]:.3f}</span></div>',
                        unsafe_allow_html=True)

    with col2:
        # Similar Markets
        st.markdown('<div class="section-header">Most Similar Markets</div>', unsafe_allow_html=True)
        for m in result["similar"]:
            badge_color = colors.get(m["recommendation"], "#666")
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-title">{m['country']}</span>
                <span style="color:#5D6D7E;font-size:0.8rem;"> ({m['region']})</span>
                <span style="float:right;background:{bgs.get(m['recommendation'],'#eee')};color:{badge_color};
                    padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600;">
                    {m['score']} {m['recommendation']}</span>
                <div style="font-size:0.75rem;color:#999;margin-top:4px;">Distance: {m['distance']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Expected Performance
        st.markdown('<div class="section-header">Expected Performance</div>', unsafe_allow_html=True)
        st.caption(f"Based on {len(result['refs'])} reference companies")
        for label, val in result["perf"].items():
            st.markdown(f'<div class="metric-card"><span class="metric-title">{label}</span> '
                        f'<span style="float:right;font-weight:600;color:#1C2340;">{val}</span></div>',
                        unsafe_allow_html=True)

    # Reference Companies
    st.markdown("---")
    st.markdown('<div class="section-header">Reference Companies</div>', unsafe_allow_html=True)

    arch_colors = {
        "High Growth, Profitable": "#0B6E4F",
        "Mature, Profitable": "#1A5276",
        "Struggling": "#922B21",
        "High Growth, Unprofitable": "#B7860B",
    }

    comp_data = []
    for c in result["refs"]:
        local = "●" if c["country"] in result["sim_countries"] else ""
        comp_data.append({
            "": local,
            "Company": c["company_name"],
            "Country": c["country"],
            "Segment": c["segment"],
            "Revenue": f"${c['revenue_usd']/1e9:.1f}B",
            "CAGR": f"{c['revenue_cagr']*100:.1f}%",
            "Margin": f"{c['net_margin']*100:.1f}%",
            "Archetype": c["archetype"],
        })

    st.dataframe(
        comp_data,
        use_container_width=True,
        hide_index=True,
    )
    st.caption("● = Operates in a similar market")

    # Expert Advice
    st.markdown("---")
    st.markdown('<div class="section-header">Expert Advice</div>', unsafe_allow_html=True)
    st.write("Generate a strategic brief synthesizing the full analysis context with AI.")

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    if st.button("🧠 Get Expert Advice", disabled=not api_key):
        with st.spinner("Generating strategic brief..."):
            brief = get_expert_advice(result, api_key)
        st.markdown(f"""
        <div style="background:#F8F9FA;border-radius:8px;padding:1.5rem;
            border:1px solid #D5D8DC;line-height:1.8;font-size:0.9rem;white-space:pre-wrap;">{brief}</div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer-bar">
        Shop-EY Market Intelligence Engine | EY Data & AI Canada | Built on analysis of 43 public e-commerce companies across 13 markets, 2015-2026
    </div>
    """, unsafe_allow_html=True)
