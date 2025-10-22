import tensorflow as tf
import tensorflow_datasets as tfds

from Net.ResNet_Block import ResNet_Block
from adam import Adam
from nadam import Nadam
from muon import Muon

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


# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(10)
# ])
def create_model():
    inputs = tf.keras.Input(batch_shape=(None, 28, 28))
    x = tf.keras.layers.Reshape((28, 28, 1))(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", name="in")(x)
    for _ in range(2):
        x = ResNet_Block(32, (3, 3))(x)

    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(10, name="out")(x)
    return tf.keras.Model(inputs=inputs, outputs=out)


model = create_model()
# model.save_weights("test.weights.h5")

model.load_weights("test.weights.h5")
model.compile(
    # optimizer=Adam(0.001, weight_decay=None, caution=False),
    # optimizer=Nadam(0.001, weight_decay=None, caution=False),
    optimizer=Muon(1e-3, weight_decay=0.0, caution=True,
                   exclude_layers=["in", "out"]),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)