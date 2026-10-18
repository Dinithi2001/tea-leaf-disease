from ontology_queries import get_disease_symptoms, get_matched_symptoms, get_symptom_info

# Disease -> symptom classes, read from the bodies of the symptomSupported SWRL rules via SPARQL.
DISEASE_SYMPTOMS = get_disease_symptoms()

ALL_SYMPTOMS = sorted({s for symptoms in DISEASE_SYMPTOMS.values() for s in symptoms})

# Symptom class -> (label, description), from rdfs:label / rdfs:comment via SPARQL.
SYMPTOM_INFO = get_symptom_info()

def symptom_label(key):
    label, _ = SYMPTOM_INFO.get(key, (key, ""))
    return label

def score_symptoms(user_reported_symptoms):
    """
    Returns a 0.0-1.0 overlap score per disease based on how many of that
    disease's symptom SWRL rules fire for a leaf individual carrying the
    reported symptoms (one rule per symptom), evaluated with SPARQL.
    """
    matched = get_matched_symptoms(user_reported_symptoms)
    scores = {}
    for disease, known_symptoms in DISEASE_SYMPTOMS.items():
        scores[disease] = len(matched.get(disease, [])) / len(known_symptoms)
    return scores

if __name__ == "__main__":
    print("Available symptoms:")
    for i, s in enumerate(ALL_SYMPTOMS, 1):
        print(f"  {i}. {s}")

    print()
    choice = input("Enter symptom numbers you observe (comma-separated), or press Enter to skip: ")
    if choice.strip():
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        selected = [ALL_SYMPTOMS[i] for i in indices]
    else:
        selected = []

    print()
    print("Selected symptoms:", selected)
    print("Symptom match scores:", score_symptoms(selected))