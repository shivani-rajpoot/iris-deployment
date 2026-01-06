from src.dsproject.predict import load_model, predict_single

MODEL_PATH = 'artifacts/model/model.joblib'

def prompt_float(name, default=None):
    while True:
        raw = input(f"Enter {name}" + (f" (default {default})" if default is not None else "") + ": ").strip()
        if not raw and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print('Please enter a valid number.')

if __name__ == '__main__':
    model = load_model(MODEL_PATH)
    print('Provide four features for Iris prediction (in cm).')
    sl = prompt_float('sepal_length', 5.1)
    sw = prompt_float('sepal_width', 3.5)
    pl = prompt_float('petal_length', 1.4)
    pw = prompt_float('petal_width', 0.2)
    out = predict_single(model, [sl, sw, pl, pw])
    label_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    print('Prediction:', label_map[out['prediction']])
    if out['proba']:
        print('Probabilities:', out['proba'])