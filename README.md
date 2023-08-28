<div align="center">
<img src="./assets/tloss.gif" width="320">
<h1 align="center">t-loss-tf</h1>
<h3 align="center">Robust T-Loss for Medical Image Segmentation with TensorFlow backend</h3>

[![GitHub Downloads](https://img.shields.io/github/downloads/andreped/t-loss-tf/total?label=GitHub%20downloads&logo=github)](https://github.com/andreped/t-loss-tf/releases)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/andreped/t-loss-tf/workflows/tests/badge.svg)](https://github.com/andreped/t-loss-tf/actions)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

</div>

This T-loss implementation is an adaption of the original PyTorch code [Digital-Dermatology/t-loss-loss](https://github.com/Digital-Dermatology/t-loss).

More information about T-loss and the original paper can be found [here](https://robust-tloss.github.io/).

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

# create dummy inputs and GTs
input_shape = (16, 16, 1)
x = tf.ones((32,) + input_shape, dtype="float32")
y = tf.ones((32,) + input_shape, dtype="float32")

# define network
input_x = tf.keras.Input(shape=input_shape)
input_y = tf.keras.Input(shape=input_shape)
z = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), activation="relu")(input_x)
z = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(z)
z = tf.keras.layers.UpSampling2D(size=(2, 2))(z)
z = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(z)
z = TLoss(tensor_shape=input_shape, image_size=input_shape[0])(z, input_y)
model = tf.keras.Model(inputs=[input_x, input_y], outputs=[z])

# train model
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
