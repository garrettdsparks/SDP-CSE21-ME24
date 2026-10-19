# Pressure Gauge Binary Classifier (`gauge_classifier`)

A transfer-learning computer vision pipeline built with PyTorch and `timm` (`efficientnet_b0`) to classify industrial boiler components as **YES (`gauge`)** or **NO (`not_gauge`)**.

---

## 1. Directory Structure

Because training images and model weights are excluded from Git via `.gitignore`, create the local `data/` subdirectories before running the notebook:

```text
gauge_classifier/
├── data/                        # Excluded from Git (create locally)
│   ├── gauge/                   # Positive samples (analog pressure gauges)
│   ├── not_gauge/               # Negative samples (boilers, pipes, valves, round lookalikes)
│   └── unseen_tests/            # Optional: drop standalone test images here
├── models/                      # Auto-generated during training (excluded from Git)
│   └── best_gauge_model.pt      # Best validation checkpoint weights
├── environment.yml              # Cross-platform Conda environment specification
├── gauge_classifier.ipynb       # End-to-end training, metrics, and inference notebook
└── README.md
```

---

## 2. Environment Setup (macOS / Windows / Linux)

### Option A: Using an Existing Anaconda Environment in VS Code / Jupyter
1. Open `gauge_classifier.ipynb` and select your preferred Conda kernel in the top-right kernel picker.
2. Run the installation cell at the top of the notebook to install the required dependencies directly into the active kernel:

```python
%pip install torch torchvision timm scikit-learn matplotlib seaborn pillow
```

### Option B: Creating a Fresh Conda Environment from `environment.yml`
From your terminal or Anaconda Prompt, navigate into `gauge_classifier/` and run:

```bash
conda env create -f environment.yml
conda activate gauge-env
```

Then select **`gauge-env`** as your kernel inside Jupyter / VS Code.

---

## 3. Preparing Local Training Data

1. Create `data/gauge/` and `data/not_gauge/` inside `gauge_classifier/`.
2. Populate both folders with `.jpg`, `.jpeg`, `.png`, or `.webp` images:
   * **`data/gauge/`**: ~25+ images of analog pressure gauges (various angles and lighting).
   * **`data/not_gauge/`**: ~25+ images of non-gauge mechanical room equipment (boiler vessels, piping, data plates, valve handwheels, sight glasses, clocks).
3. The notebook automatically performs a stratified **70% Train / 15% Validation / 15% Test** split in memory—no manual folder splitting required.

---

## 4. Training & Evaluation

Run all cells in `gauge_classifier.ipynb` sequentially:
* **Cell 1 (Setup & Hardware Detection):** Seeds random number generators and automatically selects Apple Silicon (`mps`), NVIDIA GPU (`cuda`), or `cpu`.
* **Cell 2 (Data Pipeline):** Loads images from `data/`, applies data augmentations, and builds PyTorch `DataLoader` objects.
* **Cell 3 (Fine-Tuning):** Fine-tunes a pretrained `efficientnet_b0` backbone using `BCEWithLogitsLoss` and saves the lowest validation loss checkpoint to `models/best_gauge_model.pt`.
* **Cell 4 (Performance Metrics Dashboard):** Evaluates the best checkpoint on the held-out Test set and outputs **Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC**, a **Confusion Matrix**, and **Loss/ROC Curves**.
* **Cell 5 (Inference):** Visualizes predictions on the unseen test split or runs single-image inference via `predict_image("path/to/image.jpg")`.
