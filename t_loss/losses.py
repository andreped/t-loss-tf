import tensorflow as tf


class TLoss(tf.keras.losses.Loss):
    """
    Implementation of T-Loss in TensorFlow.
    Args:
        image_size (float): Value of image/input size.
        nu (float): Value of nu.
        epsilon (float): Value of epsilon.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed.
            name: Name of the loss function.
    """

    def __init__(
        self,
        image_size: float = None,
        nu: float = 1.0,
        epsilon: float = 1e-8,
        reduction: str = "mean",
        name: str = "t_loss",
    ):
        super().__init__()
        self.image_size = image_size
        self.nu = nu
        self.epsilon = epsilon
        self.reduction = reduction

        self.lambdas = tf.math.ones((image_size, image_size), dtype="float32")

    def call(self, y_true, y_pred):
        delta_i = y_pred - y_true
        sum_nu_epsilon = tf.math.exp(self.nu) + self.epsilon
        first_term = tf.math.gamma((sum_nu_epsilon + self.image_size) / 2)
        second_term = tf.math.gamma(sum_nu_epsilon / 2)
        third_term = -0.5 * tf.math.reduce_sum(self.lambdas + self.epsilon)
        fourth_term = (self.image_size / 2) * tf.math.log(tf.math.pi)
        fifth_term = (self.image_size / 2) * (self.nu + self.epsilon)

        delta_squared = tf.math.pow(delta_i, 2)
        lambdas_exp = tf.math.exp(self.lambdas + self.epsilon)
        numerator = delta_squared * lambdas_exp
        numerator = tf.math.reduce_sum(numerator, axis=(1, 2))

        fraction = numerator / sum_nu_epsilon
        sixth_term = ((sum_nu_epsilon + self.image_size) / 2) * tf.math.log(1 + fraction)

        total_losses = (first_term, second_term, third_term, fourth_term, fifth_term, sixth_term)

        if self.reduction == "mean":
            return total_losses.mean()
        elif self.reduction == "sum":
            return total_losses.sum()
        elif self.reduction == "none":
            return total_losses
        else:
            raise ValueError(f"The reduction method '{self.reduction}' is not implemented.")
