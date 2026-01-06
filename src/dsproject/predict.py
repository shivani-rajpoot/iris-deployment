import numpy as np
from .utils.io import load_joblib
from .utils.logger_utils import get_logger

logger = get_logger(__name__)

def load_model(path: str):
    logger.info(f'Loading model: {path}')
    return load_joblib(path)

def predict_single(model, features):
    X = np.array(features, dtype=float).reshape(1, -1)
    pred = model.predict(X)[0]
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X).tolist()[0]
    else:
        proba = []
    return {'prediction': int(pred), 'proba': proba}