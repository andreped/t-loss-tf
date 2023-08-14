# t-loss-tf

This repository contains the implementation of the T-Loss designed for semantic segmentation with TensorFlow backend.

This implementation is a work in progress. Will make a release when it is finished. Stay tuned!

## Installation

```
pip install git+https://github.com/andreped/t-loss-tf.git
```

## Usage
```python
import tensorflow as tf
from t_loss import TLoss

model = tf.keras.Model()
model.compile(loss=TLoss(image_size=512))
[...]
```

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

## License
The code in this repository is released under [MIT License](https://github.com/andreped/t-loss-tf/blob/main/LICENSE).
