from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from .config import Config
from .data import load_data
from .features import build_preprocess_pipeline
from .model import make_model
from .utils.io import save_joblib, save_json, ensure_dir
from .utils.logger_utils import get_logger

logger = get_logger(__name__)

def train(cfg: Config):
    logger.info('Loading data...')
    X, y = load_data(cfg.data.source, cfg.data.path)

    logger.info('Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    logger.info('Building pipelines...')
    pre = build_preprocess_pipeline(cfg.preprocessing.scaler)
    model = make_model(cfg.model.type, **cfg.model.params)

    if pre is not None:
        pipe = Pipeline([('preprocess', pre), ('model', model)])
    else:
        pipe = Pipeline([('model', model)])

    logger.info('Training...')
    pipe.fit(X_train, y_train)

    logger.info('Evaluating...')
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info(f'Accuracy: {acc:.4f}')

    out_dir = ensure_dir(Path(cfg.output.model_dir))
    model_path = out_dir / cfg.output.model_filename
    meta_path = out_dir / cfg.output.metadata_filename

    save_joblib(pipe, model_path)
    meta = {
        'project_name': cfg.project_name,
        'model_type': cfg.model.type,
        'params': cfg.model.params,
        'accuracy': float(acc),
        'classes_': [int(c) for c in np.unique(y)],
    }
    save_json(meta, meta_path)
    logger.info(f'Saved model to {model_path}')
    logger.info(f'Saved metadata to {meta_path}')
    return {'accuracy': acc, 'model_path': str(model_path), 'meta_path': str(meta_path)}