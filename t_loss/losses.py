import math

import tensorflow as tf


class TLoss(tf.keras.layers.Layer):
    """
    Implementation of T-Loss in TensorFlow.
    Args:
        image_size (float): Value of image/input size.
        nu (float): Value of nu.
        epsilon (float): Value of epsilon.
        reduction_mode (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed.
            'name': Name of the loss function.
    """

    def __init__(
        self,
        tensor_shape: float = None,
        image_size: float = None,
        nu: float = 1.0,
        epsilon: float = 1e-8,
        reduction_mode: str = "mean",
        name: str = "t_loss",
        **kwargs,
    ):
        super().__init__()
        self.tensor_shape = tensor_shape
        self.image_size = image_size
        self.D = tf.constant(image_size * image_size, dtype="float32")
        self.nu = nu
        self.epsilon = epsilon
        self.reduction_mode = reduction_mode

        self.lambdas = tf.ones((image_size, image_size), dtype="float32")
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.nu = self.add_weight(
            name="nu", shape=self.tensor_shape, initializer=tf.keras.initializers.Constant(self.nu), trainable=True
        )

    def call(self, y_pred, y_true):
        delta_i = y_pred - y_true
        sum_nu_epsilon = tf.math.exp(self.nu) + self.epsilon
        first_term = tf.math.lgamma((sum_nu_epsilon + self.D) / 2)
        second_term = tf.math.lgamma(sum_nu_epsilon / 2)
        third_term = -0.5 * tf.math.reduce_sum(self.lambdas + self.epsilon)
        fourth_term = (self.D / 2) * tf.math.log(
            tf.constant(math.pi)
        )  # tf.constant(tf.math.log(math.pi), dtype="float32")
        fifth_term = (self.D / 2) * (self.nu + self.epsilon)

        delta_squared = tf.math.pow(delta_i, 2)
        lambdas_exp = tf.math.exp(self.lambdas + self.epsilon)
        numerator = delta_squared * lambdas_exp
        numerator = tf.math.reduce_sum(numerator, axis=(1, 2, 3))

        fraction = numerator / sum_nu_epsilon
        sixth_term = ((sum_nu_epsilon + self.D) / 2) * tf.math.log(1 + fraction)

        total_losses = first_term + second_term + third_term + fourth_term + fifth_term + sixth_term

        if self.reduction_mode == "mean":
            output = tf.math.reduce_mean(total_losses)
        elif self.reduction_mode == "sum":
            output = tf.math.reduce_sum(total_losses)
        elif self.reduction_mode == "none":
            output = total_losses
        else:
            raise ValueError(f"The reduction method '{self.reduction_mode}' is not implemented.")

        # add to losses
        self.add_loss(output)

        # We won't actually use the output, but we need something for the TF graph
        return output
