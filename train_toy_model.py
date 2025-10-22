import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

# tf.config.run_functions_eagerly(True)

from adam import Adam
from nadam import Nadam
from muon import Muon

x = np.arange(-1000, 1000, 0.1)
y = 5*x + 10

def create_model():
    inputs = tf.keras.layers.Input(batch_shape=(None, 1))
    # x = tf.keras.layers.Dense(4, name="in")(inputs)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Activation("relu")(x)
    # x = tf.keras.layers.Dense(32)(x)
    # # x = tf.keras.layers.Dense(128)(x)
    # x = tf.keras.layers.Activation("relu")(x)
    # x = tf.keras.layers.Dense(128)(x)
    output = tf.keras.layers.Dense(1, use_bias=True, name="out")(inputs)

    return tf.keras.Model(inputs=inputs, outputs=output)

model = create_model()

optim = Muon(learning_rate=1e-2,
             adam_lr_ratio=1,
             use_nadam=True,
             nesterov=1,
             # exclude_layers=["out"],
             use_caution=0,
             weight_decay=0.004,
             )
# optim = Adam(learning_rate=1e-3)
# optim = Nadam(learning_rate=1e-3, weight_decay=0.04)
# model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-3), loss="mse")
model.compile(optimizer=optim, loss="mse")

model.fit(x, y, batch_size=64, epochs=100)

print(model.weights)


