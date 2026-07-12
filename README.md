# Bank Customer Churn Prediction using ANN

This project predicts whether a bank customer is likely to churn using an Artificial Neural Network (TensorFlow/Keras).

## Files required (besides your `.ipynb` notebook)

- `requirements.txt` - Python dependencies
- `predict.py` - CLI inference script for saved model artifacts
- `.gitignore` - Prevents committing virtual env, checkpoints, and model artifacts
- `README.md` - Setup and usage instructions

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected artifacts from notebook training

Save these from your notebook after training:

- Keras model: `model.keras` (or `model.h5`)
- Optional preprocessing object: `preprocessor.pkl` (or `.joblib`)

## Run prediction

Create an input JSON file with one customer object (or a list of customer objects), then run:

```bash
python predict.py --model model.keras --preprocessor preprocessor.pkl --input input.json
```

If your model does not require external preprocessing:

```bash
python predict.py --model model.keras --input input.json
```
