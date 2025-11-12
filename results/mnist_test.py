import tensorflow as tf
import tensorflow_datasets as tfds

from Net.ResNet_Block import ResNet_Block
from adam import Adam
from nadam import Nadam
from muon import Muon
from grok_fast import Grokfast_EMA
from ortho_grad import Orthograd

# code from https://www.tensorflow.org/datasets/keras_example
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# written by me
def create_model():
    inputs = tf.keras.Input(batch_shape=(None, 28, 28))
    x = tf.keras.layers.Reshape((28, 28, 1))(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", name="in")(x)
    for _ in range(2):
        x = ResNet_Block(64, (3, 3))(x)

    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation("relu")(out)

    out = tf.keras.layers.Dense(32)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation("relu")(out)

    out = tf.keras.layers.Dense(10, name="out")(out)
    return tf.keras.Model(inputs=inputs, outputs=out)


model = create_model()
model.save_weights("test.weights.h5")
# commented out to ensure the same weights are used for different trials

model.load_weights("test.weights.h5")

optimizer = Orthograd(Muon(1e-2,
                           weight_decay=1e-3,
                           adam_lr_ratio=0.1,
                           use_nadam=False,
                   exclude_layers=["in", "out"]))
# optimizer = Orthograd(Grokfast_EMA(Muon(1e-3, weight_decay=None,
#                    exclude_layers=["in", "out"]))) # must be in this order.

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    # jit_compile=False,
)

model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test,
)