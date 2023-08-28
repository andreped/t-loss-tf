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
at appropriate place. An example can be seen below:

```python
import tensorflow as tf
from t_loss import TLoss

input_shape = (16, 16, 1)
# create dummy inputs and GTs
x = tf.ones((32,) + input_shape, dtype="float32")
y = tf.ones((32,) + input_shape, dtype="float32")

input_x = tf.keras.Input(shape=input_shape)
input_y = tf.keras.Input(shape=input_shape)

z = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), activation="relu")(input_x)
z = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(z)
z = tf.keras.layers.UpSampling2D(size=(2, 2))(z)
z = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(z)
z = TLoss(tensor_shape=input_shape, image_size=input_shape[0])(z, input_y)
model = tf.keras.Model(inputs=[input_x, input_y], outputs=[z])
print(model.summary())

model.compile(optimizer="adam")
model.fit(x=[x, y], y=y, batch_size=2, epochs=1, verbose="auto")
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
