import streamlit as st
from PIL import Image
import tempfile
import os
import base64

from integration import (
    predict_disease, check_environmental_support, score_diseases,
    explain_disease_rule, TREATMENTS
)
from locations import TEA_PLANTATIONS
from weather import get_weather
from symptoms import score_symptoms, ALL_SYMPTOMS


@st.cache_data
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


BG_IMAGE_B64 = get_base64_image("assets/tea_bg.png")
LOGO_B64 = get_base64_image("assets/logo.png")
logo_img = Image.open("assets/logo.png")

st.set_page_config(
    page_title="TeaGuard",
    page_icon=logo_img,
    layout="centered",
)

# =========================================================
# THEME
# =========================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

LIGHT = {
    "bg": "#F7F5EF", "card": "#FFFFFF", "text": "#1F2A1E", "subtext": "#46523F",
    "accent": "#3E7C4A", "accent_soft": "#E4F0E2", "border": "#DCE6D8",
    "sidebar": "#EFF3EA", "overlay": "rgba(255,255,255,0.72)",
}
DARK = {
    "bg": "#121712", "card": "#1E251C", "text": "#F1F7EE", "subtext": "#C3D2BE",
    "accent": "#7BDB88", "accent_soft": "#26332A", "border": "#33422F",
    "sidebar": "#161C15", "overlay": "rgba(10,14,10,0.72)",
}
T = DARK if st.session_state.dark_mode else LIGHT

# =========================================================
# ONE consolidated stylesheet
# =========================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif;
}}

.stApp {{
    background-color: {T["bg"]};
    background-image: linear-gradient({T["overlay"]}, {T["overlay"]}),
                       url("data:image/png;base64,{BG_IMAGE_B64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    color: {T["text"]};
}}

section[data-testid="stSidebar"] {{
    background-color: {T["sidebar"]};
    border-right: 1px solid {T["border"]};
}}

h1, h2, h3, h4 {{ color: {T["text"]} !important; }}
p, span, label, .stCaption, div[data-testid="stCaptionContainer"], li {{
    color: {T["subtext"]} !important;
}}
.stMarkdown strong, .stMarkdown b {{ color: {T["text"]} !important; }}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {T["card"]} !important;
    border: 1px solid {T["border"]} !important;
    border-radius: 14px !important;
    padding: 10px 6px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.12);
}}

div[data-testid="stMetric"] {{
    background-color: {T["card"]};
    border: 1px solid {T["border"]};
    border-radius: 14px;
    padding: 14px 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.10);
}}
div[data-testid="stMetricValue"] {{ color: {T["text"]} !important; }}
div[data-testid="stMetricLabel"] {{ color: {T["subtext"]} !important; }}

div[data-testid="stExpander"] {{
    background-color: {T["card"]};
    border: 1px solid {T["border"]};
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.10);
}}

div[data-testid="stAlert"] {{
    background-color: {T["card"]} !important;
    border: 1px solid {T["border"]} !important;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.12);
}}
div[data-testid="stAlert"] p {{ color: {T["text"]} !important; }}

section[data-testid="stFileUploaderDropzone"] {{
    background-color: {T["card"]} !important;
    border: 1.5px dashed {T["accent"]} !important;
    border-radius: 12px !important;
}}
section[data-testid="stFileUploaderDropzone"] * {{ color: {T["subtext"]} !important; }}
section[data-testid="stFileUploaderDropzone"] button {{
    background-color: {T["accent_soft"]} !important;
    color: {T["accent"]} !important;
    border: 1px solid {T["accent"]} !important;
}}

div[data-baseweb="select"] > div {{
    background-color: {T["card"]} !important;
    border: 1px solid {T["border"]} !important;
    color: {T["text"]} !important;
    border-radius: 10px !important;
}}
div[data-baseweb="select"] span {{ color: {T["text"]} !important; }}
ul[data-baseweb="menu"] {{ background-color: {T["card"]} !important; }}
ul[data-baseweb="menu"] li {{ color: {T["text"]} !important; }}
span[data-baseweb="tag"] {{ background-color: {T["accent"]} !important; }}
span[data-baseweb="tag"] span {{ color: #FFFFFF !important; }}

button[kind="primary"] {{
    background-color: {T["accent"]} !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}}
button[kind="primary"] p {{ color: #FFFFFF !important; }}

button[kind="secondary"] {{
    background-color: {T["card"]} !important;
    border: 1px solid {T["border"]} !important;
    border-radius: 10px !important;
}}
button[kind="secondary"] p {{ color: {T["text"]} !important; }}

hr {{ border-color: {T["border"]} !important; }}

/* ---- Sidebar brand block ---- */
.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: -8px;
    margin-bottom: 2px;
}}
.sidebar-brand img {{ width: 56px; height: 56px; object-fit: contain; }}
.sidebar-brand span {{
    font-size: 1.7rem;
    font-weight: 700;
    color: {T["text"]} !important;
    letter-spacing: -0.5px;
    line-height: 1;
}}
.sidebar-tagline {{
    font-size: 0.8rem;
    color: {T["subtext"]} !important;
    margin: 4px 0 14px 0;
}}

/* ---- Interactive step strip (replaces the old static hero) ---- */
.step-card {{
    background-color: {T["card"]};
    border: 1px solid {T["border"]};
    border-radius: 14px;
    padding: 16px 10px;
    text-align: center;
    height: 100%;
    transition: transform 0.15s ease, border-color 0.15s ease;
}}
.step-card:hover {{
    transform: translateY(-3px);
    border-color: {T["accent"]};
}}
.step-num {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background-color: {T["accent_soft"]};
    color: {T["accent"]} !important;
    font-weight: 700;
    font-size: 0.85rem;
    margin-bottom: 8px;
}}
.step-title {{
    font-weight: 600;
    font-size: 0.88rem;
    color: {T["text"]} !important;
    margin: 2px 0 2px 0;
}}
.step-desc {{
    font-size: 0.75rem;
    color: {T["subtext"]} !important;
}}

/* ---- Empty-state card (replaces the plain info box) ---- */
.empty-state {{
    border: 1.5px dashed {T["accent"]};
    border-radius: 14px;
    padding: 28px 20px;
    text-align: center;
    margin-top: 4px;
}}
.empty-state .icon {{ font-size: 28px; margin-bottom: 8px; }}
.empty-state .title {{ color: {T["text"]} !important; font-weight: 600; font-size: 1rem; }}
.empty-state .desc {{ color: {T["subtext"]} !important; font-size: 0.85rem; margin-top: 3px; }}

/* ---- Explanation cards ---- */
.reason-card {{
    background-color: {T["card"]};
    border: 1px solid {T["border"]};
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
}}
.reason-card.support {{ border-left: 4px solid {T["accent"]}; border-radius: 0 12px 12px 0; }}
.reason-card.contradict {{ border-left: 4px solid #E0A030; border-radius: 0 12px 12px 0; }}
.reason-title {{
    font-weight: 600;
    color: {T["text"]} !important;
    margin-bottom: 6px;
    font-size: 0.95rem;
}}
.reason-item {{
    color: {T["subtext"]} !important;
    font-size: 0.88rem;
    margin: 3px 0;
}}
.conclusion-box {{
    background-color: {T["accent_soft"]};
    border: 1px solid {T["accent"]};
    border-radius: 12px;
    padding: 14px 16px;
    margin-top: 8px;
    color: {T["text"]} !important;
    font-size: 0.92rem;
    line-height: 1.5;
}}
.symptom-pill {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    margin: 3px 4px 3px 0;
}}
.symptom-pill.matched {{
    background-color: {T["accent_soft"]};
    color: {T["accent"]} !important;
    border: 1px solid {T["accent"]};
}}
.symptom-pill.missing {{
    background-color: {T["border"]};
    color: {T["subtext"]} !important;
}}
</style>
""", unsafe_allow_html=True)


# =========================================================
# Explanation parser
# =========================================================
def render_explanation(explanation_text):
    lines = explanation_text.split("\n")
    supporting, contradicting = [], []
    matched_symptoms, missing_symptoms = [], []
    conclusion = ""
    section = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Supporting factors"):
            section = "support"
        elif stripped.startswith("Contradicting factors"):
            section = "contradict"
        elif stripped.startswith("Matched symptoms:"):
            matched_symptoms = [s.strip() for s in stripped.replace("Matched symptoms:", "").split(",") if s.strip()]
        elif stripped.startswith("Known symptoms NOT reported:"):
            missing_symptoms = [s.strip() for s in stripped.replace("Known symptoms NOT reported:", "").split(",") if s.strip()]
        elif stripped.startswith("CONCLUSION:"):
            conclusion = stripped.replace("CONCLUSION:", "").strip()
        elif stripped.startswith("+") and section == "support":
            supporting.append(stripped.lstrip("+ ").strip())
        elif stripped.startswith("-") and section == "contradict":
            contradicting.append(stripped.lstrip("- ").strip())

    col1, col2 = st.columns(2)

    with col1:
        card_html = '<div class="reason-card support"><div class="reason-title">✅ Supporting factors</div>'
        if supporting:
            for item in supporting:
                card_html += f'<div class="reason-item">• {item}</div>'
        else:
            card_html += '<div class="reason-item">None</div>'
        card_html += "</div>"
        st.markdown(card_html, unsafe_allow_html=True)

    with col2:
        card_html = '<div class="reason-card contradict"><div class="reason-title">⚠️ Contradicting factors</div>'
        if contradicting:
            for item in contradicting:
                card_html += f'<div class="reason-item">• {item}</div>'
        else:
            card_html += '<div class="reason-item">None</div>'
        card_html += "</div>"
        st.markdown(card_html, unsafe_allow_html=True)

    if matched_symptoms or missing_symptoms:
        pills_html = '<div style="margin: 6px 0 4px 0;">'
        for s in matched_symptoms:
            pills_html += f'<span class="symptom-pill matched">✓ {s}</span>'
        for s in missing_symptoms:
            pills_html += f'<span class="symptom-pill missing">{s}</span>'
        pills_html += "</div>"
        st.markdown(pills_html, unsafe_allow_html=True)

    if conclusion:
        st.markdown(f'<div class="conclusion-box">🧭 {conclusion}</div>', unsafe_allow_html=True)


# =========================================================
# SIDEBAR — brand now lives here, top-left
# =========================================================
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand">
        <img src="data:image/png;base64,{LOGO_B64}">
        <span>TeaGuard</span>
    </div>
    <div class="sidebar-tagline">Tea leaf disease diagnosis</div>
    """, unsafe_allow_html=True)

    st.toggle("🌙 Dark mode", key="dark_mode")
    st.divider()

    st.header("📋 Diagnosis inputs")

    st.subheader("1. Leaf image")
    uploaded_file = st.file_uploader("Upload a tea leaf photo", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Preview", width="stretch")

    st.subheader("2. Plantation location")
    location_name = st.selectbox("Location", list(p["name"] for p in TEA_PLANTATIONS.values()))
    selected_site = next(p for p in TEA_PLANTATIONS.values() if p["name"] == location_name)

    st.subheader("3. Observed symptoms")
    st.caption("Optional — select any that match what you see")
    user_symptoms = st.multiselect("Symptoms", ALL_SYMPTOMS, label_visibility="collapsed")

    st.divider()
    run_button = st.button("🔍 Run diagnosis", type="primary", use_container_width=True)

# =========================================================
# MAIN AREA — interactive step strip instead of a static hero
# =========================================================
steps = [
    ("1", "📷 Upload", "A clear photo of the affected leaf"),
    ("2", "📍 Locate", "Select the plantation region"),
    ("3", "🩺 Symptoms", "Tick what you visually observe"),
    ("4", "🔍 Diagnose", "Get a ranked, explained result"),
]
cols = st.columns(4)
for col, (num, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-num">{num}</div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# =========================================================
# RUN PIPELINE
# =========================================================
if run_button:
    if uploaded_file is None:
        st.error("⚠️ Please upload a leaf image in the sidebar before running a diagnosis.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getvalue())
        image_path = tmp.name

    progress = st.progress(0, text="Starting diagnosis...")
    progress.progress(20, text="Fetching live weather...")
    humidity, temperature, sunshine_hours = get_weather(selected_site["lat"], selected_site["lon"])

    progress.progress(50, text="Running CNN image analysis...")
    cnn_results = predict_disease(image_path)

    progress.progress(75, text="Checking environmental conditions against domain knowledge...")
    ontology_support = check_environmental_support(humidity, temperature, sunshine_hours)
    symptom_matches = score_symptoms(user_symptoms)
    ranked = score_diseases(cnn_results, ontology_support, symptom_matches)

    progress.progress(100, text="Done!")
    progress.empty()
    os.unlink(image_path)

    top_disease = next(iter(ranked))
    top_info = ranked[top_disease]
    top_disease_class = top_disease + "_Disease"

    st.subheader("📊 Diagnosis snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Location", location_name)
    c2.metric("Humidity", f"{humidity}%")
    c3.metric("Temperature", f"{temperature}°C")

    st.divider()

    st.subheader("🧠 CNN prediction")
    with st.container(border=True):
        for disease, prob in sorted(cnn_results.items(), key=lambda x: -x[1]):
            st.markdown(f"""
            <div style="margin-bottom: 14px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="font-weight:600; color:{T["text"]};">{disease}</span>
                    <span style="color:{T["subtext"]};">{prob:.1%}</span>
                </div>
                <div style="background-color:{T["border"]}; border-radius:8px; height:10px; overflow:hidden;">
                    <div style="background-color:{T["accent"]}; width:{prob*100}%; height:100%; border-radius:8px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.subheader("🏆 Final ranked diagnosis")
    with st.container(border=True):
        for disease, info in ranked.items():
            is_top = disease == top_disease
            icon = "✅" if info["environmental_support"] else "⚠️"
            label = f"{'⭐ ' if is_top else ''}**{disease}** — {info['final_score']:.1%} overall"
            st.write(f"{label}  \n{icon} CNN {info['cnn_confidence']:.1%} · "
                     f"{'environmentally supported' if info['environmental_support'] else 'no environmental support'} · "
                     f"symptom match {info['symptom_score']:.0%}")

    st.divider()

    st.subheader("🔎 Why this diagnosis?")
    explanation = explain_disease_rule(
        top_disease_class, humidity, temperature, sunshine_hours,
        maturity="YoungLeaf", reported_symptoms=user_symptoms
    )
    with st.container(border=True):
        render_explanation(explanation)
        with st.expander("View raw reasoning text"):
            st.code(explanation, language=None)

    st.divider()

    st.subheader("📝 Result")

    if top_info["cnn_confidence"] > 0.85:
        st.success(f"**Diagnosis confirmed: {top_disease}**", icon="✅")
        if not top_info["environmental_support"]:
            st.info(
                "Current weather doesn't match this disease's typical favorable conditions, "
                "but that doesn't lower confidence here — visible symptoms often reflect "
                "conditions from 1–3 weeks earlier, before weather changed.",
                icon="💡",
            )
        with st.container(border=True):
            st.markdown("**Recommended treatment**")
            for step in TREATMENTS.get(top_disease, ["No treatment data available."]):
                st.write(f"- {step}")

    elif top_info["environmental_support"] and top_info["cnn_confidence"] > 0.5:
        st.success(f"**Diagnosis confirmed: {top_disease}**", icon="✅")
        with st.container(border=True):
            st.markdown("**Recommended treatment**")
            for step in TREATMENTS.get(top_disease, ["No treatment data available."]):
                st.write(f"- {step}")

    elif top_info["cnn_confidence"] > 0.5:
        if top_info["symptom_score"] > 0:
            st.warning(
                f"**Possible {top_disease}** ({top_info['cnn_confidence']:.1%} confidence) — "
                f"partially supported by reported symptoms ({top_info['symptom_score']:.0%} match), "
                f"but current weather doesn't support it and CNN confidence is only moderate. "
                f"Consider professional confirmation.",
                icon="⚠️",
            )
        else:
            st.warning(
                f"**Possible {top_disease}** ({top_info['cnn_confidence']:.1%} confidence) — "
                f"neither environmental conditions nor reported symptoms support this diagnosis. "
                f"Recommend professional inspection before treatment.",
                icon="⚠️",
            )
        st.caption("Treatment guidance withheld pending symptom or expert confirmation.")

    else:
        st.warning(
            f"**Diagnosis uncertain** (top candidate: {top_disease}, confidence "
            f"{top_info['cnn_confidence']:.1%}). Treatment guidance withheld — "
            f"recommend professional inspection.",
            icon="❓",
        )

else:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🍃</div>
        <div class="title">Ready when you are</div>
        <div class="desc">Fill in the sidebar and click Run diagnosis to begin</div>
    </div>
    """, unsafe_allow_html=True)