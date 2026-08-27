DISEASE_SYMPTOMS = {
    "RedRust": ["RustySpot", "ReddishHairySpot", "YellowPatchVariegation", "StemDieback"],
    "BlisterBlight": ["TranslucentSpot", "WhitePowderyProtrusion"],
    "GreyBlight": ["ConcentricRingPatch", "GreySilveredCenter", "BlackFruitingBodyDots"],
}

ALL_SYMPTOMS = sorted({s for symptoms in DISEASE_SYMPTOMS.values() for s in symptoms})

def score_symptoms(user_reported_symptoms):
    """
    Returns a 0.0-1.0 overlap score per disease based on how many of that
    disease's known symptoms the user reported.
    """
    scores = {}
    for disease, known_symptoms in DISEASE_SYMPTOMS.items():
        if not known_symptoms:
            scores[disease] = 0.0
            continue
        overlap = len(set(user_reported_symptoms) & set(known_symptoms))
        scores[disease] = overlap / len(known_symptoms)
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