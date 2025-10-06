# Character Recognition Web App

This project provides a minimal Flask web app that lets you upload an image of a single character and returns a prediction using the included Keras model file `emnist_cnnmodel87.keras`.

Prerequisites
- Python 3.8+ (3.10/3.11 recommended)
- Windows PowerShell (instructions below)

Quick setup (PowerShell)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000 in your browser, upload an image, and the app will show predicted label and confidence.

Notes
- The mapping of model output indices to characters is defined in `predict_utils.py` as `CLASS_MAP`. Update it if your model uses a different label encoding.
- If you don't have GPU support, TensorFlow will run on CPU; installation may take a while.
