# 🏅 Sports Image Classification
---

## 📁 Project Structure

```
ahksase2002-sports-image-classification/
├── app.py                      # Streamlit web app
├── requirments.txt             # Python dependencies
├── scratch_models128.py        # Custom models for 128x128 input
├── scratch_models224.py        # Custom models for 224x224 input
├── utils.py                    # Utility functions for model loading, prediction, etc.
├── models/                     # Pretrained model weights (.pth)
│   ├── densenet.pth
│   ├── resnet.pth
│   └── simple_net.pth
├── notebooks/                  # Jupyter notebooks for training, analysis, Optuna, etc.
│   └── *.ipynb
├── Sports-Image-Classification.rar  # Compressed project archive
└── README.md                   # This file
```

---

## 🧠 Classes

The model predicts one of the following sports:

- Badminton
- Cricket
- Tennis
- Swimming
- Soccer
- Wrestling
- Karate

---

## 🚀 Features

- **Pretrained Models:** Fine-tuned ResNet, VGG, and other architectures with center/random cropping.
- **Custom CNNs:** Scratch-trained networks for various input sizes (128 & 224).
- **Optuna Integration:** Automated hyperparameter tuning with visualization.
- **Streamlit UI:** Interactive web interface for model comparison and image predictions.
- **Image Transform Visualization:** View how images are preprocessed.
- **Comprehensive Analysis:** Dataset statistics, training metrics, and pixel distribution plots.

---

## 🖥️ Usage

### 1. Install Dependencies

```bash
pip install -r requirments.txt
```

### 2. Launch the Web App

```bash
streamlit run app.py
```

### 3. Navigate the App

- **Classification Dashboard:** Upload images, predict class, compare model performance.
- **Input Transform Showcase:** Visualize random vs center cropping.
- **Image Analysis:** View dataset distribution and pixel stats.
- **Optuna Analysis:** Analyze hyperparameter tuning and model trials.

---

## 🛠️ Models

The following model configurations are available:

- **Input Sizes:** 128x128 or 224x224
- **Training Types:** Scratch vs Pretrained
- **Augmentations:** Random Resized Crop or Center Crop
- **Decay:** With or without weight decay regularization

---

## 📊 Training & Evaluation

Use the Jupyter notebooks in `/notebooks/` for:

- Training models
- Comparing different architectures
- Plotting results
- Hyperparameter tuning via Optuna

Example:
- `main_224_pretrained_center_cropping.ipynb`
- `analyzing_optuna_results.ipynb`

---

## 📦 Models & Checkpoints

Pretrained and scratch-trained models are saved in subdirectories like:

```
logs_random_cropping/logs_128_pretrained/checkpoints/resnet50/
```

Each contains:
- `final_model.pt` – model weights
- `metrics.pt` – training/validation metrics
- `accuracy_plot.png`, `loss_plot.png` – training visualizations

---

## 📌 Notes

- `ImageDataset1` expects already-split folders (train/test).
- `ImageDataset2` allows for custom data splitting.
- CLIP-based zero-shot labeling used for test data without labels.

---
