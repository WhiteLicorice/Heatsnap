# Heatsnap

This is the repository for **Heatsnap**, a deep learning project that estimates the **Heat Index** (perceived temperature) of an outdoor environment using only a single photograph.

The model analyzes visual cues (sky color, haze, lighting conditions, shadows) to regress a continuous heat index value, useful for estimating thermal comfort from static outdoor cameras.

# Architecture

We have migrated from Keras to **Native PyTorch (v2.x)** to ensure:
1.  **Maximum Training Speed:** Direct control over the training loop and `DataLoader` avoids abstraction overhead.
2.  **Robustness:** Custom error handling for the Skyfinder dataset's corrupt/truncated JPEG files.
3.  **Deployment Readiness:** Native support for ONNX export, facilitating offline Android deployment (ExecuTorch).

# Development

### Prerequisites
* Windows 10/11 (Native)
* Python 3.11.7
* NVIDIA GPU (RTX 3060 or newer recommended) with CUDA 12.x support.

### Setup
Do **NOT** use `pip install -r requirements.txt`. We use a specific script to ensure the correct PyTorch-CUDA bindings are installed.

1.  Clone the repository.
2.  Run the setup script:
    ```cmd
    autosetup.bat
    ```
3.  This script will:
    * Create a virtual environment (`venv`).
    * Install **PyTorch 2.5.1** (Stable) with **CUDA 12.4**.
    * Install scientific dependencies (`pysolar`, `scikit-learn`, `pandas`).
    * Validate GPU acceleration and environment integrity.

# Dataset

We utilize the [Skyfinder](https://cs.valdosta.edu/~rpmihail/skyfinder/) dataset (~90,000 images from 53 outdoor weather cameras).
* See the `data/` directory for data structure.
* **Note:** The dataset contains corrupt/truncated JPEGs. The training pipeline includes these during training for model robustness.
 
# Roadmap & Status

### Phase 1: Data Preprocessing (Complete)
- [x] Build `extract_working_dataset.py` to parse raw Skyfinder metadata.
- [x] Implement robust MATLAB serial date conversion (w/ 366-day epoch offset fix and UTC normalization).
- [x] Filter night images using `pysolar` to calculate Solar Elevation Angle (< -6.0° Civil Twilight).
- [x] Implement **Anderson et al. (2013)** Heat Index algorithm:
    - [x] Handle cold domain (T < 40°F) continuity.
    - [x] Handle transitional domain (Steadman approximation).
    - [x] Handle hot domain (Rothfusz regression).

### Phase 2: Training Pipeline (In Progress)
- [x] **Migration:** Switch core backend from Keras 3 to Native PyTorch.
- [x] **Split Strategy:** Group by Camera ID (Leave-One-Site-Out) to prevent overfitting to background artifacts.
- [x] **Data Loading:**
    - [x] Implement custom PyTorch `Dataset` class.
    - [x] Handle corrupt JPEGs gracefully (try-catch in `__getitem__`).
    - [x] Resize to 224x224 and apply ImageNet Normalization.
- [ ] **Training Loop:**
    - [x] Implement EfficientNetV2 backbone (via `torchvision`).
    - [x] Implement Huber Loss (robust regression).
    - [ ] Optimize training with `torch.amp` (Automatic Mixed Precision).
    - [ ] Instead of regressing a continuous heat index, perhaps look into regressing qualitative heat index warning (range of heat index) to improve viability.

### Phase 3: Evaluation & Deployment (Pending)
- [ ] **Metrics:**
    - [ ] Compute MAE (Mean Absolute Error) on unseen camera sites.
    - [ ] Generate "Predicted vs. Actual" scatter plots to assess linearity.
- [ ] **Explainability:**
    - [ ] Generate Grad-CAM visualizations to verify the model attends to sky/haze features rather than ground objects.
- [ ] **Mobile Deployment:**
    - [ ] Export trained model to **ONNX**.
    - [ ] Convert to **TFLite** or **ExecuTorch** format for offline Android integration.
