from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def build_preprocess_pipeline(strategy: str = 'standard'):
    if strategy == 'standard':
        return Pipeline([('scaler', StandardScaler())])
    if strategy == 'minmax':
        return Pipeline([('scaler', MinMaxScaler())])
    if strategy == 'none':
        return None
    raise ValueError(f'Unknown scaler strategy: {strategy}')