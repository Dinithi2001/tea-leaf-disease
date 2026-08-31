import json
import numpy as np
import keras
import keras.src.ops.numpy as npops
from keras.utils import load_img, img_to_array
from owlready2 import *
from locations import select_plantation
from weather import get_weather
from symptoms import score_symptoms, ALL_SYMPTOMS, DISEASE_SYMPTOMS

# ---------- Load CNN ----------
model = keras.models.load_model(
    "model/tea_model.h5",
    custom_objects={"TrueDivide": npops.TrueDivide, "Subtract": npops.Subtract},
)

with open("model/tea_model.classes.json") as f:
    class_list = json.load(f)

full_names = {"BB": "BlisterBlight", "RR": "RedRust", "Healthy": "Healthy"}

TREATMENTS = {
    "BlisterBlight": [
        "Avoid dense shade, particularly in mist pockets — sunshine is detrimental to the fungus.",
        "No resistant cultivars currently exist, so cultural control matters most.",
        "Apply recommended copper fungicide.",
    ],
    "RedRust": [
        "Use rust-resistant plant varieties where available.",
        "Pluck off and burn infected leaves; rake and remove fallen debris regularly.",
        "Obstruct direct sunshine with shade trees.",
        "Skillful pruning and plucking of bud leaf.",
        "Ensure adequate spacing between plants for air circulation.",
        "Spray copper oxychloride 0.25% or Bordeaux mixture on infected/injured plant portions.",
        "Direct fungicide spraying toward young stems and laterals bearing rusty fructifications.",
        "In severely affected sections, apply 2% MoP and Urea as a rehabilitative measure.",
        "Avoid plant stress and poorly drained sites; improve soil nutrient status (N, P, K).",
        "Burn infected plants after the growing season — do not compost them.",
    ],
    "GreyBlight": [
        "Monitor incidence through and after the harvesting round, not only in the wet season.",
        "Use sharp, well-maintained pruning shears and harvester blades — clean cuts resist infection better.",
        "Schedule pruning so cut surfaces close under favourable conditions; cut below affected wood on badly hit bushes.",
        "Remove severely blighted prunings and heavy litter from the rows to reduce spore reservoir.",
        "Manage drainage, shade, and airflow to shorten leaf wetness periods.",
        "Maintain balanced nutrition — poor nutrition is consistently linked to heavier disease.",
        "No cure exists; this fungus is wound-dependent and requires ongoing management, not elimination.",
    ],
}

def predict_disease(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    results = {}
    for idx, prob in enumerate(predictions):
        code = class_list[idx]
        name = full_names.get(code, code)
        results[name] = float(prob)
    return results

# ---------- Load ontology ----------
onto = get_ontology("ontology/tea-leaf.owl").load()

def get_conditions(humidity, temperature, sunshine_hours):
    conditions = []
    if humidity > 80:
        conditions.append(onto.HighHumidity_Instance)
    if temperature < 25:
        conditions.append(onto.LowTemperature_Instance)
    elif temperature > 28:
        conditions.append(onto.HighTemperature_Instance)
    if sunshine_hours > 6:
        conditions.append(onto.BrightSunshine_Instance)
    return conditions

# Mirrors the SWRL rules (RedRustRule, BlisterBlightRule, GreyBlightRule) defined in
# tea-leaf.ttl. Evaluated directly in Python rather than via sync_reasoner_pellet(),
# since Pellet's bundled Jena jars require a JDK newer than what's installed here.
DISEASE_RULES = [
    {"disease": "RedRust_Disease", "conditions": ["BrightSunshine", "HighTemperature"], "maturity": "YoungLeaf"},
    {"disease": "BlisterBlight_Disease", "conditions": ["HighHumidity", "LowTemperature"], "maturity": "YoungLeaf"},
    {"disease": "GreyBlight_Disease", "conditions": ["HighHumidity", "PlantStress"], "maturity": None},
]

CONDITION_LABELS = {
    "HighHumidity": "high humidity (>80%)",
    "LowTemperature": "low temperature (<25°C)",
    "HighTemperature": "high temperature (>28°C)",
    "BrightSunshine": "bright sunshine (>6 sunlight hours)",
    "PlantStress": "plant stress indicators",
}

def explain_disease_rule(disease_key, humidity, temperature, sunshine_hours,
                          maturity="YoungLeaf", reported_symptoms=None):
    """
    Produces a specific, human-readable explanation of why a disease's
    environmental rule did or did not fire, naming the exact condition(s)
    responsible — not just a generic conflict message.
    """
    if reported_symptoms is None:
        reported_symptoms = []

    rule = next((r for r in DISEASE_RULES if r["disease"] == disease_key), None)
    if rule is None:
        return f"No environmental rule defined for {disease_key}."

    disease_name = disease_key.replace("_Disease", "")
    reasons_met = []
    reasons_failed = []

    # Check each required condition individually against actual values
    for cond in rule["conditions"]:
        if cond == "HighHumidity":
            if humidity > 80:
                reasons_met.append(f"humidity is {humidity}%, which satisfies high humidity (>80%)")
            else:
                reasons_failed.append(f"humidity is only {humidity}%, below the 80% needed for high humidity")
        elif cond == "LowTemperature":
            if temperature < 25:
                reasons_met.append(f"temperature is {temperature}°C, which satisfies low temperature (<25°C)")
            else:
                reasons_failed.append(f"temperature is {temperature}°C, too high for the low-temperature requirement (<25°C)")
        elif cond == "HighTemperature":
            if temperature > 28:
                reasons_met.append(f"temperature is {temperature}°C, which satisfies high temperature (>28°C)")
            else:
                reasons_failed.append(f"temperature is {temperature}°C, below the 28°C needed for high temperature")
        elif cond == "BrightSunshine":
            if sunshine_hours > 6:
                reasons_met.append(f"sunshine is estimated at {sunshine_hours}h, which satisfies bright sunshine (>6h)")
            else:
                reasons_failed.append(f"sunshine is estimated at only {sunshine_hours}h, well below the >6h needed for bright sunshine — current conditions are cloudy or rainy")
        elif cond == "PlantStress":
            reasons_failed.append("no plant stress indicators (drought, waterlogging, pest damage, poor nutrition) were reported")

    # Maturity check
    if rule["maturity"]:
        if maturity == rule["maturity"]:
            reasons_met.append(f"leaf maturity ({maturity}) matches the required stage")
        else:
            reasons_failed.append(f"leaf maturity is {maturity}, but this disease typically requires {rule['maturity']}")

    # Symptom check
    known_symptoms = DISEASE_SYMPTOMS.get(disease_name, [])
    matched_symptoms = [s for s in reported_symptoms if s in known_symptoms]
    missing_symptoms = [s for s in known_symptoms if s not in reported_symptoms]

    lines = [f"--- Explanation for {disease_name} ---"]

    if reasons_met:
        lines.append("Supporting factors:")
        for r in reasons_met:
            lines.append(f"  + {r}")
    if reasons_failed:
        lines.append("Contradicting factors:")
        for r in reasons_failed:
            lines.append(f"  - {r}")

    if matched_symptoms:
        lines.append(f"Matched symptoms: {', '.join(matched_symptoms)}")
    if missing_symptoms:
        lines.append(f"Known symptoms NOT reported: {', '.join(missing_symptoms)}")

    if not reasons_failed:
        lines.append(f"CONCLUSION: Current conditions are consistent with active {disease_name} risk.")
    else:
        lines.append(f"CONCLUSION: Current conditions do not match {disease_name}'s typical "
                      f"favorable conditions ({reasons_failed[0]}). Note: this does not rule out "
                      f"{disease_name} as the correct diagnosis — visible symptoms often appear "
                      f"1-3 weeks after the actual infection, by which point weather conditions "
                      f"may have changed considerably.")

    return "\n".join(lines)
def check_environmental_support(humidity, temperature, sunshine_hours, maturity="YoungLeaf_Instance"):
    """
    Checks whether CURRENT environmental conditions are consistent with active
    disease-favorable conditions at this location right now.

    Important: this does NOT validate whether the CNN's diagnosed disease was
    correctly caused by past weather. Since diseases like Blister Blight have
    an incubation period of 1-3 weeks between infection and visible symptoms,
    a visibly diseased leaf photographed today may have been infected under
    very different weather than what is measured now. This function should be
    interpreted as a "current disease risk / ongoing favorable conditions"
    check, not a historical infection-cause validator.
    """
    leaf = onto.LeafInstance("leaf_test_run")
    for cond in get_conditions(humidity, temperature, sunshine_hours):
        leaf.hasCondition.append(cond)
    maturity_individual = getattr(onto, maturity)
    leaf.hasMaturity.append(maturity_individual)

    condition_classes = {cls for cond in leaf.hasCondition for cls in cond.is_a}
    maturity_classes = set(maturity_individual.is_a)

    supported = []
    for rule in DISEASE_RULES:
        if not all(getattr(onto, name) in condition_classes for name in rule["conditions"]):
            continue
        if rule["maturity"] and getattr(onto, rule["maturity"]) not in maturity_classes:
            continue
        supported.append(rule["disease"])
    return supported

def score_diseases(cnn_results, ontology_support, symptom_matches=None,
                    w_cnn=0.6, w_environment=0.25, w_symptom=0.15):
    """
    Combines CNN probability, environmental support, and symptom overlap
    into a single ranked score per disease.
    """
    if symptom_matches is None:
        symptom_matches = {}

    scores = {}
    for disease, cnn_prob in cnn_results.items():
        if disease == "Healthy":
            continue

        disease_class = disease + "_Disease"
        env_score = 1.0 if disease_class in ontology_support else 0.0
        symptom_score = symptom_matches.get(disease, 0.0)  # expects 0.0-1.0

        final_score = (w_cnn * cnn_prob) + (w_environment * env_score) + (w_symptom * symptom_score)

        scores[disease] = {
            "final_score": final_score,
            "cnn_confidence": cnn_prob,
            "environmental_support": env_score == 1.0,
            "symptom_score": symptom_score,
        }

    return dict(sorted(scores.items(), key=lambda x: -x[1]["final_score"]))

# ---------- Run everything and print final result ----------
if __name__ == "__main__":
    image_path = "test_leaf.jpg"

    lat, lon, location_name = select_plantation()
    humidity, temperature, sunshine_hours = get_weather(lat, lon)
    print(f"\nUsing weather for {location_name}: humidity={humidity}%, temp={temperature}°C\n")


    print("\nAvailable symptoms:")
    for i, s in enumerate(ALL_SYMPTOMS, 1):
        print(f"  {i}. {s}")
    choice = input("Enter symptom numbers you observe (comma-separated), or press Enter to skip: ")

    if choice.strip():
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        user_symptoms = [ALL_SYMPTOMS[i] for i in indices]
    else:
        user_symptoms = []

    symptom_matches = score_symptoms(user_symptoms)
    print(f"Reported symptoms: {user_symptoms}\n")

    cnn_results = predict_disease(image_path)
    ontology_support = check_environmental_support(humidity, temperature, sunshine_hours)
    ranked = score_diseases(cnn_results, ontology_support, symptom_matches)

    print("=== CNN Prediction ===")
    for disease, prob in sorted(cnn_results.items(), key=lambda x: -x[1]):
        print(f"{disease}: {prob:.2%}")

    print()
    print("=== Ontology Environmental Support ===")
    print(ontology_support if ontology_support else "No disease environmentally supported")

    print()
    print("=== Final Ranked Diagnosis ===")
    for disease, info in ranked.items():
        flag = "supported" if info["environmental_support"] else "no environmental support"
        print(f"{disease}: {info['final_score']:.1%} overall "
              f"(CNN: {info['cnn_confidence']:.1%}, {flag})")

    print()

    print()
    top_disease = next(iter(ranked))
    top_info = ranked[top_disease]
    top_disease_class = top_disease + "_Disease"

    explanation = explain_disease_rule(
        top_disease_class, humidity, temperature, sunshine_hours,
        maturity="YoungLeaf", reported_symptoms=user_symptoms
    )
    print(explanation)
    print()

    if top_info["cnn_confidence"] > 0.85:
        # High CNN confidence stands on its own. Environmental mismatch does not
        # invalidate a strong visual diagnosis, since symptoms can lag infection
        # by 1-3 weeks — current weather may simply differ from infection-time weather.
        print(f"=== Diagnosis Confirmed: {top_disease} ===")
        if not top_info["environmental_support"]:
            print("Note: current weather does not match this disease's typical favorable "
                  "conditions, but this does not affect confidence in the diagnosis — visible "
                  "symptoms often reflect conditions from 1-3 weeks earlier.")
        print("Recommended treatment:")
        for step in TREATMENTS.get(top_disease, ["No treatment data available."]):
            print(f"  - {step}")

    elif top_info["environmental_support"] and top_info["cnn_confidence"] > 0.5:
        # Moderate CNN confidence, but corroborated by current environmental conditions.
        print(f"=== Diagnosis Confirmed: {top_disease} ===")
        print("Recommended treatment:")
        for step in TREATMENTS.get(top_disease, ["No treatment data available."]):
            print(f"  - {step}")

    elif top_info["cnn_confidence"] > 0.5:
        # Moderate confidence, no environmental corroboration — this is where
        # environment/symptom evidence should genuinely matter, since the CNN
        # itself isn't confident enough to stand alone.
        if top_info["symptom_score"] > 0:
            print(f"NOTE: Possible {top_disease} ({top_info['cnn_confidence']:.1%} confidence), "
                  f"partially supported by reported symptoms ({top_info['symptom_score']:.0%} match), "
                  f"but current weather does not support it and CNN confidence is only moderate. "
                  f"Consider professional confirmation.")
        else:
            print(f"NOTE: Possible {top_disease} ({top_info['cnn_confidence']:.1%} confidence), but "
                  f"neither environmental conditions nor reported symptoms support this diagnosis. "
                  f"Recommend professional inspection before treatment.")
        print("Treatment guidance withheld pending symptom or expert confirmation.")

    else:
        print(f"Diagnosis uncertain (top candidate: {top_disease}, confidence "
              f"{top_info['cnn_confidence']:.1%}). Treatment guidance withheld — "
              f"recommend professional inspection.")