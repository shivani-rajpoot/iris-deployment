from typing import Tuple
import pandas as pd
from sklearn.datasets import load_iris

def load_data(source: str, path: str = '') -> Tuple[pd.DataFrame, pd.Series]:
    if source == 'sklearn_iris':
        iris = load_iris(as_frame=True)
        return iris.data, iris.target
    if source == 'csv':
        df = pd.read_csv(path)
        X = df.drop(columns=['target'])
        y = df['target']
        return X, y
    raise ValueError(f'Unsupported data source: {source}')