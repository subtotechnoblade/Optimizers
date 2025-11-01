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


# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(10)
# ])

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

model.load_weights("test.weights.h5")

# optimizer = Grokfast_EMA(Orthograd(Muon(1e-3, weight_decay=1e-4, caution=False,
#                    exclude_layers=["in", "out"])))
optimizer = Orthograd(Grokfast_EMA(Muon(1e-3, weight_decay=1e-4, caution=False,
                   exclude_layers=["in", "out"])))

# optimizer = Muon(1e-3, weight_decay=0.0, caution=False,
#                    exclude_layers=["in", "out"])
# pure muon with no weight decay, an no caution
"""469/469 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step - loss: 0.5935 - sparse_categorical_accuracy: 0.85072025-11-01 14:34:39.900245: I external/local_xla/xla/service/gpu/autotuning/conv_algorithm_picker.cc:557] Omitted potentially buggy algorithm eng14{k25=0} for conv (f32[16,64,28,28]{3,2,1,0}, u8[0]{0}) custom-call(f32[16,64,28,28]{3,2,1,0}, f32[64,64,3,3]{3,2,1,0}, f32[64]{0}), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBiasActivationForward", backend_config={"cudnn_conv_backend_config":{"activation_mode":"kNone","conv_result_scale":1,"leakyrelu_alpha":0,"side_input_scale":0},"force_earliest_schedule":false,"operation_queue_id":"0","wait_on_operation_queues":[]}
2025-11-01 14:34:40.045790: I external/local_xla/xla/service/gpu/autotuning/conv_algorithm_picker.cc:557] Omitted potentially buggy algorithm eng14{k25=0} for conv (f32[16,64,28,28]{3,2,1,0}, u8[0]{0}) custom-call(f32[16,64,28,28]{3,2,1,0}, f32[64,64,3,3]{3,2,1,0}, f32[64]{0}, f32[16,64,28,28]{3,2,1,0}), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBiasActivationForward", backend_config={"cudnn_conv_backend_config":{"activation_mode":"kNone","conv_result_scale":1,"leakyrelu_alpha":0,"side_input_scale":1},"force_earliest_schedule":false,"operation_queue_id":"0","wait_on_operation_queues":[]}
469/469 ━━━━━━━━━━━━━━━━━━━━ 59s 67ms/step - loss: 0.5928 - sparse_categorical_accuracy: 0.8508 - val_loss: 1.0782 - val_sparse_categorical_accuracy: 0.6915
Epoch 2/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 24s 52ms/step - loss: 0.0524 - sparse_categorical_accuracy: 0.9881 - val_loss: 0.1114 - val_sparse_categorical_accuracy: 0.9681
Epoch 3/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 25s 52ms/step - loss: 0.0213 - sparse_categorical_accuracy: 0.9951 - val_loss: 0.0819 - val_sparse_categorical_accuracy: 0.9765
Epoch 4/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 24s 50ms/step - loss: 0.0096 - sparse_categorical_accuracy: 0.9982 - val_loss: 0.0976 - val_sparse_categorical_accuracy: 0.9719
Epoch 5/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 24s 50ms/step - loss: 0.0072 - sparse_categorical_accuracy: 0.9986 - val_loss: 0.1027 - val_sparse_categorical_accuracy: 0.9704
Epoch 6/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 24s 51ms/step - loss: 0.0055 - sparse_categorical_accuracy: 0.9990 - val_loss: 0.1182 - val_sparse_categorical_accuracy: 0.9670
Epoch 7/10
 52/469 ━━━━━━━━━━━━━━━━━━━━ 18s 45ms/step - loss: 0.0035 - sparse_categorical_accuracy: 0.9997"""

# muon with ortho grad, grokfast , and wd=1e-4
"""469/469 ━━━━━━━━━━━━━━━━━━━━ 31s 40ms/step - loss: 0.6322 - sparse_categorical_accuracy: 0.8507 - val_loss: 0.5546 - val_sparse_categorical_accuracy: 0.8949
Epoch 2/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 14s 29ms/step - loss: 0.1300 - sparse_categorical_accuracy: 0.9841 - val_loss: 0.1196 - val_sparse_categorical_accuracy: 0.9821
Epoch 3/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 14s 29ms/step - loss: 0.0939 - sparse_categorical_accuracy: 0.9925 - val_loss: 0.1404 - val_sparse_categorical_accuracy: 0.9809
Epoch 4/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 14s 29ms/step - loss: 0.0770 - sparse_categorical_accuracy: 0.9970 - val_loss: 0.1863 - val_sparse_categorical_accuracy: 0.9739
Epoch 5/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 14s 30ms/step - loss: 0.0720 - sparse_categorical_accuracy: 0.9987 - val_loss: 0.1594 - val_sparse_categorical_accuracy: 0.9839
Epoch 6/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 14s 29ms/step - loss: 0.0702 - sparse_categorical_accuracy: 0.9992 - val_loss: 0.1372 - val_sparse_categorical_accuracy: 0.9818
Epoch 7/10
469/469 ━━━━━━━━━━━━━━━━━━━━ 14s 29ms/step - loss: 0.0694 - sparse_categorical_accuracy: 0.9996 - val_loss: 0.1184 - val_sparse_categorical_accuracy: 0.9837
Epoch 8/10
289/469 ━━━━━━━━━━━━━━━━━━━━ 5s 28ms/step - loss: 0.0667 - sparse_categorical_accuracy: 0.9998"""
model.compile(
    # optimizer=Adam(0.001, weight_decay=None, caution=False),
    # optimizer=Nadam(0.001, weight_decay=None, caution=False),
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test,
)