import json
import numpy as np
import keras
import keras.src.ops.numpy as npops
from keras.utils import load_img, img_to_array
from locations import select_plantation
from weather import get_weather
from symptoms import score_symptoms, ALL_SYMPTOMS, DISEASE_SYMPTOMS
from ontology_queries import (
    instance_of, temp_leaf, get_rule_conditions, infer_supported_diseases,
    evaluate_weather, get_condition_thresholds, get_treatments, get_maturity_stages,
    get_incubation_periods,
    ENVIRONMENT, MATURITY,
)

# ---------- Load CNN ----------
model = keras.models.load_model(
    "model/tea_model.h5",
    custom_objects={"TrueDivide": npops.TrueDivide, "Subtract": npops.Subtract},
)

with open("model/tea_model.classes.json") as f:
    class_list = json.load(f)

# Maps CNN class codes (from tea_model.classes.json) to the disease names used by the
# ontology and symptoms.py. Keys are normalised (lowercase, letters/digits only) so that
# "GB", "Grey_Blight", "grey blight" etc. all resolve to the same disease.
full_names = {
    "bb": "BlisterBlight", "blisterblight": "BlisterBlight",
    "rr": "RedRust", "redrust": "RedRust",
    "gb": "GreyBlight", "greyblight": "GreyBlight", "grayblight": "GreyBlight",
    "cl": "Cercosporella", "cs": "Cercosporella", "cercosporella": "Cercosporella",
    "cercosporellaleafdisease": "Cercosporella",
    "healthy": "Healthy", "h": "Healthy",
}

def class_name(code):
    key = "".join(ch for ch in code.lower() if ch.isalnum())
    return full_names.get(key, code)

# Treatment steps per disease, read from the ontology's hasTreatment links via SPARQL.
TREATMENTS = get_treatments()

def predict_disease(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    results = {}
    for idx, prob in enumerate(predictions):
        code = class_list[idx]
        name = class_name(code)
        results[name] = float(prob)
    return results

# ---------- Ontology ----------
def weather_readings(humidity, temperature, sunshine_hours):
    """Weather values keyed by the leaf data properties they are stored under in the ontology."""
    return {"humidityValue": humidity, "temperatureValue": temperature, "sunshineHours": sunshine_hours}

def get_conditions(humidity, temperature, sunshine_hours):
    """
    Environmental condition individuals that hold for these readings. The thresholds
    (e.g. HighHumidity: minHumidity 80) are stored in the ontology and read by SPARQL.
    """
    conditions, _ = evaluate_weather(weather_readings(humidity, temperature, sunshine_hours))
    return conditions

# Body conditions of the environmental and maturity SWRL rules, and the numeric thresholds
# of each environmental condition, read from the ontology via SPARQL. Used to explain a
# diagnosis; the inference itself is done by the check_*_support() functions below.
DISEASE_RULES = get_rule_conditions(ENVIRONMENT)
DISEASE_MATURITY = get_rule_conditions(MATURITY)
CONDITION_THRESHOLDS = get_condition_thresholds()

# Leaf maturity stages a user can choose from (subclasses of LeafMaturityStage), via SPARQL.
MATURITY_STAGES = get_maturity_stages()

# Time from infection to visible symptoms per disease (incubationPeriod), via SPARQL.
INCUBATION_PERIODS = get_incubation_periods()

def readable(concept):
    """HighHumidity -> high humidity"""
    return "".join(f" {ch.lower()}" if ch.isupper() else ch for ch in concept).strip()

def symptom_lag_note(disease):
    """Why current weather may not match a visible infection, using the disease's
    incubationPeriod from the ontology when it has one."""
    period = INCUBATION_PERIODS.get(disease)
    if period:
        return (f"visible symptoms appear some time after the actual infection "
                f"({readable(disease)} incubation: {period}), by which point weather "
                f"conditions may have changed considerably.")
    return ("visible symptoms appear some time after the actual infection, by which "
            "point weather conditions may have changed considerably.")

def explain_disease_rule(disease_key, humidity, temperature, sunshine_hours,
                          maturity, reported_symptoms=None):
    """
    Produces a specific, human-readable explanation of why a disease's
    environmental rule did or did not fire, naming the exact condition(s)
    responsible — not just a generic conflict message.
    """
    if reported_symptoms is None:
        reported_symptoms = []

    rule_conditions = DISEASE_RULES.get(disease_key)
    if rule_conditions is None:
        return f"No environmental rule defined for {disease_key}."

    disease_name = disease_key.replace("_Disease", "")
    reasons_met = []
    reasons_failed = []

    readings = weather_readings(humidity, temperature, sunshine_hours)
    _, met_thresholds = evaluate_weather(readings)

    # Check each threshold of each required condition against the actual reading
    for cond in rule_conditions:
        thresholds = CONDITION_THRESHOLDS.get(cond)
        if not thresholds:
            reasons_failed.append(f"no {readable(cond)} indicators were reported, and this "
                                  f"condition cannot be measured from weather data")
            continue
        for th in thresholds:
            value = readings.get(th["reading"])
            requirement = f"{th['label']} {th['op']} {th['bound']:g}{th['unit']}"
            if value is None:
                reasons_failed.append(f"{readable(cond)} needs {requirement}, but {th['label']} "
                                      f"is not available from weather data")
            elif th["threshold"] in met_thresholds:
                reasons_met.append(f"{th['label']} is {value}{th['unit']}, which satisfies "
                                   f"{readable(cond)} ({requirement})")
            else:
                reasons_failed.append(f"{th['label']} is {value}{th['unit']}, but "
                                      f"{readable(cond)} needs {requirement}")

    # Maturity check
    required_maturity = DISEASE_MATURITY.get(disease_key)
    if required_maturity:
        if maturity in required_maturity:
            reasons_met.append(f"leaf maturity ({maturity}) matches the required stage")
        else:
            reasons_failed.append(f"leaf maturity is {maturity}, but this disease typically requires "
                                  f"{' or '.join(required_maturity)}")

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
                      f"{disease_name} as the correct diagnosis — {symptom_lag_note(disease_name)}")

    return "\n".join(lines)
def check_environmental_support(humidity, temperature, sunshine_hours):
    """
    Checks whether CURRENT environmental conditions are consistent with active
    disease-favorable conditions at this location right now.

    Important: this does NOT validate whether the CNN's diagnosed disease was
    correctly caused by past weather. Since diseases have an incubation period
    between infection and visible symptoms (incubationPeriod in the ontology),
    a visibly diseased leaf photographed today may have been infected under
    very different weather than what is measured now. This function should be
    interpreted as a "current disease risk / ongoing favorable conditions"
    check, not a historical infection-cause validator.
    """
    conditions = get_conditions(humidity, temperature, sunshine_hours)
    with temp_leaf(hasCondition=conditions) as leaf:
        return infer_supported_diseases(leaf, ENVIRONMENT)

def check_maturity_support(maturity):
    """
    Checks which diseases are supported by the given leaf maturity stage
    ("YoungLeaf" or "MatureLeaf"), mirroring check_environmental_support()
    but for the independent maturity evidence track.
    """
    maturity_individual = instance_of(maturity)
    with temp_leaf(hasMaturity=[maturity_individual]) as leaf:
        return infer_supported_diseases(leaf, MATURITY)

def score_diseases(cnn_results, ontology_support, symptom_matches=None, maturity_support=None,
                    w_cnn=0.5, w_environment=0.2, w_symptom=0.15, w_maturity=0.15):
    """
    Combines CNN probability, environmental support, symptom overlap, and
    leaf maturity support into a single ranked score per disease.
    """
    if symptom_matches is None:
        symptom_matches = {}
    if maturity_support is None:
        maturity_support = []

    # Diseases known to the ontology but not to the CNN (e.g. Cercosporella) are still
    # ranked, scored on environment/symptom/maturity evidence with a CNN probability of 0.
    candidates = dict(cnn_results)
    for disease in DISEASE_SYMPTOMS:
        candidates.setdefault(disease, 0.0)

    scores = {}
    for disease, cnn_prob in candidates.items():
        if disease == "Healthy":
            continue

        disease_class = disease + "_Disease"
        env_score = 1.0 if disease_class in ontology_support else 0.0
        symptom_score = symptom_matches.get(disease, 0.0)  # expects 0.0-1.0
        maturity_score = 1.0 if disease_class in maturity_support else 0.0

        final_score = (w_cnn * cnn_prob) + (w_environment * env_score) \
            + (w_symptom * symptom_score) + (w_maturity * maturity_score)

        scores[disease] = {
            "final_score": final_score,
            "cnn_confidence": cnn_prob,
            "environmental_support": env_score == 1.0,
            "symptom_score": symptom_score,
            "maturity_support": maturity_score == 1.0,
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

    print("Leaf maturity:")
    for i, stage in enumerate(MATURITY_STAGES, 1):
        print(f"  {i}. {readable(stage).capitalize()}")
    maturity_choice = input(f"Select leaf maturity (1-{len(MATURITY_STAGES)}, default 1): ").strip()
    maturity = MATURITY_STAGES[int(maturity_choice) - 1] if maturity_choice else MATURITY_STAGES[0]
    print(f"Selected maturity: {maturity}\n")

    cnn_results = predict_disease(image_path)
    ontology_support = check_environmental_support(humidity, temperature, sunshine_hours)
    maturity_support = check_maturity_support(maturity)
    ranked = score_diseases(cnn_results, ontology_support, symptom_matches, maturity_support)

    print("=== CNN Prediction ===")
    for disease, prob in sorted(cnn_results.items(), key=lambda x: -x[1]):
        print(f"{disease}: {prob:.2%}")

    print()
    print("=== Ontology Environmental Support ===")
    print(ontology_support if ontology_support else "No disease environmentally supported")

    print()
    print("=== Ontology Maturity Support ===")
    print(maturity_support if maturity_support else "No disease maturity-supported")

    print()
    print("=== Final Ranked Diagnosis ===")
    for disease, info in ranked.items():
        flag = "supported" if info["environmental_support"] else "no environmental support"
        maturity_flag = "supported" if info["maturity_support"] else "no maturity support"
        print(f"{disease}: {info['final_score']:.1%} overall "
              f"(CNN: {info['cnn_confidence']:.1%}, {flag}, {maturity_flag})")

    print()

    print()
    top_disease = next(iter(ranked))
    top_info = ranked[top_disease]
    top_disease_class = top_disease + "_Disease"

    explanation = explain_disease_rule(
        top_disease_class, humidity, temperature, sunshine_hours,
        maturity=maturity, reported_symptoms=user_symptoms
    )
    print(explanation)
    print()

    if top_info["cnn_confidence"] > 0.85:
        # High CNN confidence stands on its own. Environmental mismatch does not
        # invalidate a strong visual diagnosis, since symptoms can lag infection
        # (incubationPeriod) — current weather may simply differ from infection-time weather.
        print(f"=== Diagnosis Confirmed: {top_disease} ===")
        if not top_info["environmental_support"]:
            print("Note: current weather does not match this disease's typical favorable "
                  "conditions, but this does not affect confidence in the diagnosis — "
                  f"{symptom_lag_note(top_disease)}")
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