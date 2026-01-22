# ML Toy Examples

Small, self-contained scripts demonstrating basic machine learning workflows using **Keras** and **PyTorch**. Intended for learning and experimentation.

---

## Scripts

### `keras_basic.py`

* Binary classification on synthetic 2D data
* Simple MLP built with Keras
* Trains, predicts, and plots score distributions

### `torch_basic.py`

* Minimal PyTorch binary classifier
* Synthetic 2D dataset
* Manual training loop and accuracy computation

### `torch_fancy.py`

* More modular PyTorch classifier
* Custom `nn.Module` classes and training loop
* Accuracy and score histogram visualization

### `torch_autoencoder.py`

* Fully connected autoencoder in PyTorch
* Trained on MNIST digits
* Visualizes original vs reconstructed images

---

## Dependencies

```text
numpy
matplotlib
scikit-learn
torch
torchvision
tensorflow / keras
```

Recommended installation:

```bash
python -m venv venv
source venv/bin/activate
pip install numpy matplotlib scikit-learn torch torchvision tensorflow
```

---

## Notes

* Uses synthetic data or MNIST
* Focused on clarity, not production use
* Best run inside a virtual environment

