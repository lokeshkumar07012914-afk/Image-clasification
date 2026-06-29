# Image-Classification

A professional, modular TensorFlow 2.x project for CIFAR-10 image classification. This repository refactors a single script project into a reusable package with training, evaluation, prediction, and an interactive Streamlit demo.

## Project overview

This project trains a convolutional neural network (CNN) on the CIFAR-10 dataset to classify 32x32 RGB images into 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

Key goals of this refactor:
- Convert a single-script prototype into a maintainable, testable Python package.
- Add modern training best-practices: BatchNormalization, Dropout, data augmentation, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
- Save the best model using the modern `.keras` format.
- Provide evaluation utilities (accuracy, precision, recall, F1, classification report, confusion matrix) and plotting helpers.
- Ship a Streamlit application for quick interactive predictions.
- Follow industry best practices: type hints, docstrings, logging, exception handling, and PEP 8 style.

## Features

- Modular layout under `src/`:
  - `config.py` — central constants and paths
  - `data_loader.py` — CIFAR-10 loader and tf.data dataset utilities with optional augmentation
  - `preprocess.py` — image loading and preprocessing helpers
  - `model.py` — improved CNN architecture (BatchNorm, Dropout)
  - `train.py` — training loop with callbacks and history/plot saving
  - `evaluate.py` — compute and save metrics and confusion matrix
  - `predict.py` — single-image prediction utility
  - `utils.py` — logging and IO helpers
- Streamlit demo at `app/app.py` for uploading images and viewing predictions and probabilities.
- Lightweight pytest tests (in `tests/`) to validate core functionality.
- CI workflow (suggested) to run tests on push/PR.

## Quickstart

1. Setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate    # Windows (PowerShell)
pip install -r requirements.txt
```

2. Train a model (example)

```bash
python main.py train --epochs 20 --batch-size 64 --augment
```

This will train the CNN on CIFAR-10 and save the best model in `saved_model/cifar10_classifier.keras`.

3. Evaluate a saved model

```bash
python main.py evaluate saved_model/cifar10_classifier.keras
```

Evaluation outputs (classification report, confusion matrix) will be stored in the `images/` directory.

4. Predict a single local image

```bash
python main.py predict saved_model/cifar10_classifier.keras path/to/image.jpg --top-k 5
```

5. Run the Streamlit demo

```bash
python main.py serve
# or
streamlit run app/app.py
```

Open the app in your browser (default http://localhost:8501) to upload an image, see predictions and probabilities.

## Repository layout

```
Image-Classification/
│
├── app/
│   └── app.py                 # Streamlit application
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── dataset/
├── saved_model/
├── images/
├── notebooks/
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
```

## Best practices and notes

- The model uses softmax outputs (probabilities). Predictions and evaluation rely on the highest probability class.
- The training pipeline includes data augmentation and callbacks to prevent overfitting and to save the best model automatically.
- For reproducibility, random seeds are set in the training entrypoint.
- The default configuration assumes CIFAR-10 images (32x32). If you adapt the repo for other datasets, update `src/config.py` accordingly.

## Contributing

Contributions are welcome. Suggested workflow:
1. Create a feature branch: `git checkout -b feat/your-feature`
2. Run tests and linters locally
3. Open a pull request against `main`

If you add code that requires heavy computation (training), include a small smoke test so CI can run quickly.

## License

This repository is released under the MIT License — see the included `LICENSE` file for details.
