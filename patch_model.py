import h5py
import json
import shutil

# Make a safe copy first
shutil.copy("model/tea_model.h5", "model/tea_model_patched.h5")

def fix_config(obj):
    """Recursively rename batch_shape -> batch_input_shape in InputLayer configs."""
    if isinstance(obj, dict):
        if obj.get("class_name") == "InputLayer" and "batch_shape" in obj.get("config", {}):
            obj["config"]["batch_input_shape"] = obj["config"].pop("batch_shape")
        for v in obj.values():
            fix_config(v)
    elif isinstance(obj, list):
        for item in obj:
            fix_config(item)

with h5py.File("model/tea_model_patched.h5", "r+") as f:
    raw_config = f.attrs["model_config"]
    if isinstance(raw_config, bytes):
        raw_config = raw_config.decode("utf-8")
    config = json.loads(raw_config)
    fix_config(config)
    f.attrs["model_config"] = json.dumps(config)

print("Patched model saved as model/tea_model_patched.h5")
