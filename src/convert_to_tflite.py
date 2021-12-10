import tensorflow as tf

model_filename = "../data/model.h5"
tfmodel_filename = "../data/model.tflite"

converter = tf.lite.TFLiteConverter.from_saved_model(model_filename)
tflite_model = converter.convert()

# Save the model.
with open(tfmodel_filename, "wb") as f:
    f.write(tflite_model)
