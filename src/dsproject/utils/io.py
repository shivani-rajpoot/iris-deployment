import json
from pathlib import Path
import joblib

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_joblib(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)

def load_joblib(path):
    return joblib.load(path)

def save_json(d, path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(d, f, indent=2)

def load_json(path):
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)