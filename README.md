# Heatsnap

This is the repository for Heatsnap, a project that aims to train a deep learning model in determining the heat index of an outdoor environment from its photo.

# Development

We'll be using Keras with PyTorch in the backend. Run `autosetup.bat` (NOT `pip install -r requirements.txt`. To install manually, see the steps in `autosetup.bat`.

# Dataset

We'll be using the [Skyfinder](https://cs.valdosta.edu/~rpmihail/skyfinder/) dataset. See the `data` directory for further instructions.

# TODO

### Phase 1: Data Curation & Physics
- [x] Build `extract_working_dataset.py` to parse raw Skyfinder metadata.
- [x] Implement robust MATLAB serial date conversion (w/ 366-day epoch offset fix and UTC normalization).
- [x] Filter night images using `pysolar` to calculate Solar Elevation Angle (< -6.0° Civil Twilight).
- [x] Implement "Anderson et al. (2013)" Heat Index algorithm:
    - [x] Handle cold domain (T < 40°F) continuity.
    - [x] Handle transitional domain (Steadman approximation).
    - [x] Handle hot domain (Rothfusz regression).

### Phase 2: Data Loading & Split
- [x] Group by Camera ID to prevent "Clever Hans" overfitting to site-specific backgrounds.
- [ ] Preprocessing:
    - [x] Resize images to 224x224 (EfficientNetV2 standard).
    - [x] Apply ImageNet normalization (Mean/Std).
    - [ ] Augment carefully (avoid disrupting sky gradients/horizon lines).
- [x] Implement `SkyfinderDataset` class to serve image tensors paired with continuous `heat_index` targets.

### Phase 3: Modeling & Evaluation
- [x] Implement EfficientNetV2 backbone with a dense regression head.
- [ ] Optimize using Huber Loss (robust to outliers) or MSE.
- [ ] Evaluation:
    - [ ] Compute MAE (Mean Absolute Error) on the unseen camera sites.
    - [ ] Generate "Predicted vs. Actual" scatter plots to assess linearity.
    - [ ] Generate Class Activation Maps (CAM) to verify the model attends to sky/haze features rather than ground artifacts.
