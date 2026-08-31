import streamlit as st
from PIL import Image
import tempfile
import os

from integration import (
    predict_disease, check_environmental_support, score_diseases,
    explain_disease_rule, TREATMENTS
)
from locations import TEA_PLANTATIONS
from weather import get_weather
from symptoms import score_symptoms, ALL_SYMPTOMS

st.set_page_config(
    page_title="Tea Leaf Disease Diagnosis",
    page_icon="🍃",
    layout="centered",
)

# ---------- Header ----------
st.title("🍃 Tea Leaf Disease Diagnosis")
st.caption("Hybrid CNN + Ontology-based diagnostic system for Blister Blight, Grey Blight, and Red Rust")

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown(
        "1. **Upload** a photo of the affected tea leaf.\n"
        "2. **Select** the plantation location so we can check live weather.\n"
        "3. **Tick any symptoms** you can see on the leaf (optional, but improves accuracy).\n"
        "4. Click **Run Diagnosis** — the system combines CNN image analysis, current "
        "weather, and your symptom checklist to produce a ranked diagnosis with an "
        "explanation of *why*, plus treatment guidance where appropriate."
    )

st.divider()

# ---------- Sidebar: all inputs live here ----------
with st.sidebar:
    st.header("📋 Diagnosis Inputs")

    st.subheader("1. Leaf Image")
    uploaded_file = st.file_uploader("Upload a tea leaf photo", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Preview", use_container_width=True)

    st.subheader("2. Plantation Location")
    location_name = st.selectbox("Location", list(p["name"] for p in TEA_PLANTATIONS.values()))
    selected_site = next(p for p in TEA_PLANTATIONS.values() if p["name"] == location_name)

    st.subheader("3. Observed Symptoms")
    st.caption("Optional — select any that match what you see")
    user_symptoms = st.multiselect("Symptoms", ALL_SYMPTOMS, label_visibility="collapsed")

    st.divider()
    run_button = st.button("🔍 Run Diagnosis", type="primary", use_container_width=True)

# ---------- Run pipeline ----------
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

    # ---------- Snapshot row ----------
    st.subheader("📊 Diagnosis Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Location", location_name)
    c2.metric("Humidity", f"{humidity}%")
    c3.metric("Temperature", f"{temperature}°C")

    st.divider()

    # ---------- CNN results ----------
    st.subheader("🧠 CNN Prediction")
    for disease, prob in sorted(cnn_results.items(), key=lambda x: -x[1]):
        col_label, col_bar = st.columns([1, 3])
        col_label.write(f"**{disease}**")
        col_bar.progress(prob, text=f"{prob:.1%}")

    st.divider()

    # ---------- Final ranked diagnosis ----------
    st.subheader("🏆 Final Ranked Diagnosis")
    for disease, info in ranked.items():
        is_top = disease == top_disease
        icon = "✅" if info["environmental_support"] else "⚠️"
        label = f"{'⭐ ' if is_top else ''}**{disease}** — {info['final_score']:.1%} overall"
        st.write(f"{label}  \n{icon} CNN {info['cnn_confidence']:.1%} · "
                 f"{'environmentally supported' if info['environmental_support'] else 'no environmental support'} · "
                 f"symptom match {info['symptom_score']:.0%}")

    st.divider()

    # ---------- Explanation ----------
    st.subheader("🔎 Why this diagnosis?")
    explanation = explain_disease_rule(
        top_disease_class, humidity, temperature, sunshine_hours,
        maturity="YoungLeaf", reported_symptoms=user_symptoms
    )
    with st.expander("View detailed reasoning", expanded=True):
        st.code(explanation, language=None)

    st.divider()

    # ---------- Verdict + treatment ----------
    st.subheader("📝 Result")

    if top_info["cnn_confidence"] > 0.85:
        st.success(f"**Diagnosis Confirmed: {top_disease}**", icon="✅")
        if not top_info["environmental_support"]:
            st.info(
                "Current weather doesn't match this disease's typical favorable conditions, "
                "but that doesn't lower confidence here — visible symptoms often reflect "
                "conditions from 1–3 weeks earlier, before weather changed.",
                icon="💡",
            )
        st.markdown("**Recommended Treatment**")
        for step in TREATMENTS.get(top_disease, ["No treatment data available."]):
            st.write(f"- {step}")

    elif top_info["environmental_support"] and top_info["cnn_confidence"] > 0.5:
        st.success(f"**Diagnosis Confirmed: {top_disease}**", icon="✅")
        st.markdown("**Recommended Treatment**")
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
    st.info("👈 Fill in the details in the sidebar and click **Run Diagnosis** to get started.")