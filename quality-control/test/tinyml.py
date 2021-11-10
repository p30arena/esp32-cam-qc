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

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE,
                                                            color_mode='grayscale',
                                                            )

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)


def representative_dataset():
    for input_value, output_value in train_dataset.batch(1):
        input_value = tf.squeeze(input_value, axis=0)
        yield [input_value]


if model_path.exists():
    model = tf.keras.models.load_model(model_path)
else:
    print("you must first run trian.py")
    exit(1)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()
bytes = hexdump.dump(tflite_model).split(' ')
c_array = ', '.join(['0x%02x' % int(byte, 16) for byte in bytes])
c = 'const unsigned char model_data[] __attribute__((aligned(8))) = {%s};' % (
    c_array)
c += '\nconst int model_data_len = %d;' % (len(bytes))
c_code = c

# Save the model to disk
with open("out/model/model.tflite", "wb") as file:
    file.write(tflite_model)
with open("out/model/model.c", "wb") as file:
    file.write(str.encode(c_code))
