# t-loss-tf

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/andreped/t-loss-tf/workflows/tests/badge.svg)](https://github.com/andreped/t-loss-tf/actions)

This repository contains the implementation of the T-Loss designed for semantic segmentation with TensorFlow backend.

This implementation is a work in progress. Will make a release when it is finished. Stay tuned!

Currently: The implementation seems to work mechanically but the returned loss values are strange.

## Installation

```
pip install git+https://github.com/andreped/t-loss-tf.git
```

## Usage
As the t-loss contains a trainable parameter, in keras the loss needed to be implemented as a custom layer.
Hence, instead of setting the loss as normally through `model.compile(loss=[...])`, just add it to the model
at appropriate place.

```python
import tensorflow as tf
from t_loss import TLoss

model = tf.keras.Sequential()
model.add(TLoss(image_size=512))
[...]
```

## License
The code in this repository is released under [MIT License](https://github.com/andreped/t-loss-tf/blob/main/LICENSE).

## Citation
The implementation is based on the original torch implementation hosted [here](https://github.com/Digital-Dermatology/t-loss).

Hence, if this code is used, please cite the original research article:
```
@inproceedings{gonzalezjimenezRobustTLoss2023,
  title     = {Robust T-Loss for Medical Image Segmentation},
  author    = {Gonzalez-Jimenez, Alvaro and Lionetti, Simone and Gottfrois, Philippe and Gröger, Fabian and Pouly, Marc and Navarini, Alexander},
  journal   = {Medical {{Image Computing}} and {{Computer Assisted Intervention}} – {{MICCAI}} 2023},
  publisher = {{Springer International Publishing}},
  year      = {2023},
}
```
