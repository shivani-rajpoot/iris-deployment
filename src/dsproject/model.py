from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def make_model(model_type: str, **params):
    if model_type == 'logistic_regression':
        return LogisticRegression(**params)
    if model_type == 'random_forest':
        return RandomForestClassifier(**params)
    if model_type == 'svc':
        return SVC(probability=True, **params)
    raise ValueError(f'Unknown model type: {model_type}')