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

st.set_page_config(page_title="Tea Leaf Disease Diagnosis", layout="centered")
st.title("🍃 Tea Leaf Disease Diagnosis")
st.caption("Hybrid CNN + Ontology-based diagnostic system")

# ---------- Inputs ----------
st.header("1. Upload Leaf Image")
uploaded_file = st.file_uploader("Choose a tea leaf image", type=["jpg", "jpeg", "png"])

st.header("2. Select Plantation Location")
location_name = st.selectbox("Location", list(p["name"] for p in TEA_PLANTATIONS.values()))
selected_site = next(p for p in TEA_PLANTATIONS.values() if p["name"] == location_name)

st.header("3. Observed Symptoms (optional)")
user_symptoms = st.multiselect("Select any symptoms you observe", ALL_SYMPTOMS)

run_button = st.button("Run Diagnosis", type="primary")

# ---------- Run pipeline ----------
if run_button:
    if uploaded_file is None:
        st.error("Please upload a leaf image first.")
        st.stop()

    # Save uploaded image to a temp file so predict_disease() can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    st.image(Image.open(image_path), caption="Uploaded leaf", width=300)

    with st.spinner("Fetching weather..."):
        humidity, temperature, sunshine_hours = get_weather(selected_site["lat"], selected_site["lon"])
    st.info(f"Weather at {location_name}: {humidity}% humidity, {temperature}°C")

    with st.spinner("Running CNN prediction..."):
        cnn_results = predict_disease(image_path)

    ontology_support = check_environmental_support(humidity, temperature, sunshine_hours)
    symptom_matches = score_symptoms(user_symptoms)
    ranked = score_diseases(cnn_results, ontology_support, symptom_matches)

    os.unlink(image_path)  # clean up temp file

    # ---------- CNN results ----------
    st.header("CNN Prediction")
    for disease, prob in sorted(cnn_results.items(), key=lambda x: -x[1]):
        st.write(f"**{disease}**: {prob:.2%}")
        st.progress(prob)

    # ---------- Final ranked diagnosis ----------
    st.header("Final Ranked Diagnosis")
    top_disease = next(iter(ranked))
    top_info = ranked[top_disease]
    top_disease_class = top_disease + "_Disease"

    for disease, info in ranked.items():
        flag = "✅ supported" if info["environmental_support"] else "⚠️ no environmental support"
        st.write(f"**{disease}**: {info['final_score']:.1%} overall (CNN: {info['cnn_confidence']:.1%}, {flag})")

    # ---------- Explanation ----------
    st.header("Explanation")
    explanation = explain_disease_rule(
        top_disease_class, humidity, temperature, sunshine_hours,
        maturity="YoungLeaf", reported_symptoms=user_symptoms
    )
    st.code(explanation, language=None)

    # ---------- Verdict + treatment ----------
    st.header("Result")
    if top_info["environmental_support"] and top_info["cnn_confidence"] > 0.5:
        st.success(f"Diagnosis Confirmed: {top_disease}")
        st.subheader("Recommended Treatment")
        for step in TREATMENTS.get(top_disease, ["No treatment data available."]):
            st.write(f"- {step}")
    elif top_info["symptom_score"] > 0:
        st.warning(f"Possible {top_disease}, but environmental conditions do not fully support this diagnosis. "
                   f"Reported symptoms partially align ({top_info['symptom_score']:.0%} match). "
                   f"Treatment guidance withheld — consider professional confirmation.")
    else:
        st.warning(f"Possible {top_disease}, but environmental conditions do not support this diagnosis, "
                   f"and no matching symptoms were reported. "
                   f"Treatment guidance withheld — recommend professional inspection.")