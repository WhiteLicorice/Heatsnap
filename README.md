# Heatsnap

This is the repository for Heatsnap, a project that aims to train a deep learning model in determining the heat index of an outdoor environment from its photo.

# Development

We'll be using Keras with PyTorch in the backend. Run `autosetup.bat` (NOT `pip install -r requirements.txt`. To install manually, see the steps in `autosetup.bat`.

# Dataset

We'll be using the [Skyfinder](https://cs.valdosta.edu/~rpmihail/skyfinder/) dataset. See the `data` directory for further instructions.

# TODO

### Phase 1
- [x] Build `extract_working_dataset.py` to parse raw Skyfinder metadata.
- [x] Implement robust MATLAB serial date conversion (w/ 366-day epoch offset fix and UTC normalization).
- [x] Filter night images using `pysolar` to calculate Solar Elevation Angle (< -6.0° Civil Twilight).
- [x] Implement "Anderson et al. (2013)" Heat Index algorithm:
    - [x] Handle cold domain (T < 40°F) continuity.
    - [x] Handle transitional domain (Steadman approximation).
    - [x] Handle hot domain (Rothfusz regression).

### Phase 2
- [ ] Create Train/Val/Test splits.
    - *Note: Must split by Camera ID or Time Blocks to prevent data leakage (don't put 7:00 AM in train and 7:01 AM in test).*
- [ ] Preprocess:
    - [ ] Resize images (e.g., 224x224 for EfficientNetV2).
    - [ ] Normalization (Mean/Std).
    - [ ] Augmentation (Random crops/rotations? *Careful with sky gradients*).
- [ ] Implement `SkyfinderDataset` class to load images paired with the new `heat_index` targets.

### Phase 3
- [ ] Train a simple CNN or fine-tune ResNet-18 to predict Heat Index.
- [ ] Implement MSE (Mean Squared Error) or Huber Loss (robust to outliers).
- [ ] Evaluate:
    - [ ] Calculate MAE (Mean Absolute Error).
    - [ ] Visualize "Predicted vs. Actual" scatter plots.
    - [ ] Visualize Activation Maps (CAM) to see if the model looks at the sky or the ground.
