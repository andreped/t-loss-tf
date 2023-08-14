import tensorflow as tf
from t_loss import TLoss


def test_loss():
    # create dummy batch size 4 of preds and gts
    y_true = tf.ones((4, 512, 512), dtype="float32")
    y_pred = tf.ones((4, 512, 512), dtype="float32")

    loss_function = TLoss(image_size=512)

    print(loss_function(y_true, y_pred))

    # assert loss_function(y_true, y_pred) == 0
    