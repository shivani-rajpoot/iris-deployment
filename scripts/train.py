from src.dsproject.config import load_config
from src.dsproject.train import train

if __name__ == '__main__':
    cfg = load_config('config/config.yaml')
    res = train(cfg)
    print('Training complete.')
    print('Accuracy:', round(res['accuracy'], 4))
    print('Model saved at:', res['model_path'])