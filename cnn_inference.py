import json
import numpy as np
import keras
import keras.src.ops.numpy as npops
from keras.utils import load_img, img_to_array

model = keras.models.load_model(
    "model/tea_model.h5",
    custom_objects={"TrueDivide": npops.TrueDivide, "Subtract": npops.Subtract},
)

with open("model/tea_model.classes.json") as f:
    class_list = json.load(f)

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

if __name__ == "__main__":
    results = predict_disease("test_leaf.jpg")
    for disease, prob in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{disease}: {prob:.2%}")
