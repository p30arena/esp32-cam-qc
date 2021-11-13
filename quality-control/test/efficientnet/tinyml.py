import os
import pathlib
import tensorflow as tf
import hexdump

PATH = os.path.join('out', 'products')
model_path = pathlib.Path('out/model')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 1
IMG_SIZE = (240, 240)
IMG_SHAPE = IMG_SIZE + (3,)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)


def representative_dataset():
    for input_value, output_value in train_dataset.batch(1):
        input_value = tf.squeeze(input_value)
        yield [input_value]


if model_path.exists():
    model = tf.keras.models.load_model(model_path)
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = model.layers[2](inputs)  # efficientnetb0
    x = model.layers[3](x)  # global_avg_pooling2d
    x = model.layers[4](x)  # batch_norm_2d
    outputs = model.layers[6](x)  # dense
    model = tf.keras.Model(inputs, outputs)
else:
    print("you must first run trian.py")
    exit(1)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

# Save the model to disk
with open("out/model/model.tflite", "wb") as file:
    file.write(tflite_model)
