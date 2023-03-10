import tensorflow as tf
import keras.backend as K

input_shape = (224, 224, 3)

base_model = tf.keras.applications.DenseNet121(
    input_shape=input_shape, weights="imagenet", include_top=False)
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

for layer in base_model.layers:
    layer.trainable = False
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)


optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999)


def weighted_binary_crossentropy(weights):

    def w_binary_crossentropy(y_true, y_pred):
        # Calculate the binary crossentropy
        binary_crossentropy = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weights_tensor = y_true * weights[1] + (1. - y_true) * weights[0]
        weighted_binary_crossentropy = weights_tensor * binary_crossentropy

        return K.mean(weighted_binary_crossentropy)

    return w_binary_crossentropy
loss = weighted_binary_crossentropy([0.07,0.93])
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

