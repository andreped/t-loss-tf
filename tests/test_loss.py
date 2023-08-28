import tensorflow as tf
from t_loss import TLoss


def test_loss():
    # create dummy batch size 4 of preds and gts
    y_true = tf.ones((4, 512, 512, 1), dtype="float32")
    y_pred = tf.ones((4, 512, 512, 1), dtype="float32")

    loss_function = TLoss(tensor_shape=(512, 512, 1), image_size=512)

    print(loss_function(y_true, y_pred))

    # assert loss_function(y_true, y_pred) == 0


def test_loss_with_network():
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

    model.save("test_model")
    del model
    loaded_model = tf.keras.models.load_model("test_model")


if __name__ == "__main__":
    test_loss()
    test_loss_with_network()
