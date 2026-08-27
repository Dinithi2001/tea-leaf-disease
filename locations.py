TEA_PLANTATIONS = {
    "1": {"name": "Nuwara Eliya", "lat": 6.9497, "lon": 80.7891},
    "2": {"name": "Kandy", "lat": 7.2906, "lon": 80.6337},
    "3": {"name": "Ratnapura", "lat": 6.6828, "lon": 80.3992},
    "4": {"name": "Badulla", "lat": 6.9934, "lon": 81.0550},
    "5": {"name": "Galle", "lat": 6.0535, "lon": 80.2210},
    "6": {"name": "Hedigalla", "lat": 6.4679318, "lon": 80.2960776},
}

def select_plantation():
    print("Select plantation location:")
    for key, site in TEA_PLANTATIONS.items():
        print(f"  {key}. {site['name']}")

    choice = input("Enter number: ").strip()
    if choice not in TEA_PLANTATIONS:
        print("Invalid selection, defaulting to Nuwara Eliya.")
        choice = "1"

    selected = TEA_PLANTATIONS[choice]
    return selected["lat"], selected["lon"], selected["name"]

if __name__ == "__main__":
    lat, lon, name = select_plantation()
    print(f"Selected: {name} ({lat}, {lon})")