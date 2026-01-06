from dataclasses import dataclass
from typing import Any, Dict
import yaml

@dataclass
class DataConfig:
    source: str
    path: str

@dataclass
class PreprocessConfig:
    scaler: str

@dataclass
class ModelConfig:
    type: str
    params: Dict[str, Any]

@dataclass
class OutputConfig:
    model_dir: str
    model_filename: str
    metadata_filename: str

@dataclass
class Config:
    project_name: str
    random_state: int
    test_size: float
    data: DataConfig
    preprocessing: PreprocessConfig
    model: ModelConfig
    output: OutputConfig

def load_config(path: str) -> 'Config':
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return Config(
        project_name=cfg['project_name'],
        random_state=cfg['random_state'],
        test_size=cfg['test_size'],
        data=DataConfig(**cfg['data']),
        preprocessing=PreprocessConfig(**cfg['preprocessing']),
        model=ModelConfig(**cfg['model']),
        output=OutputConfig(**cfg['output']),
    )