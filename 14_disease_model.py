import tensorflow as tf

input_shape = (224, 224, 3)

base_model = tf.keras.applications.DenseNet121(
    input_shape=input_shape, weights="ImageNet", include_top=False)
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(14, activation='sigmoid')(x)

for layer in base_model.layers:
    layer.trainable = False
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999)
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['AUC'])
