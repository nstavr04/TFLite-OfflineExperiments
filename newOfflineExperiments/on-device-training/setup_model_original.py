import tensorflow as tf

IMG_SIZE = 224

class Model(tf.Module):

  def __init__(self):
    self.model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE), name='flatten'),
        tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(10, name='dense_2')
    ])

    self.model.compile(
        optimizer='sgd',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

    # The `train` function takes a batch of input images and labels.
    @tf.function(input_signature=[
         tf.TensorSpec([None, IMG_SIZE, IMG_SIZE], tf.float32),
         tf.TensorSpec([None, 10], tf.float32),
     ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            prediction = self.model(x)
            loss = self._LOSS_FN(prediction, y)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self._OPTIM.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        result = {"loss": loss}
        for grad in gradients:
            result[grad.name] = grad
        return result

    self.train = train

  @tf.function(input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE], tf.float32)])
  def predict(self, x):
     return {
         "output": self.model(x)
     }

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def save(self, checkpoint_path):
     tensor_names = [weight.name for weight in self.model.weights]
     tensors_to_save = [weight.read_value() for weight in self.model.weights]
     tf.raw_ops.Save(
         filename=checkpoint_path, tensor_names=tensor_names,
         data=tensors_to_save, name='save')
     return {
         "checkpoint_path": checkpoint_path
     }

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def restore(self, checkpoint_path):
      restored_tensors = {}
      for var in self.model.weights:
        restored = tf.raw_ops.Restore(
            file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
            name='restore')
        var.assign(restored)
        restored_tensors[var.name] = restored
      return restored_tensors

SAVED_MODEL_DIR = "saved_model"

m = Model()

tf.saved_model.save(
    m,
    SAVED_MODEL_DIR,
    signatures={
        'train':
            m.train().get_concrete_function(),
        'infer':
            m.predict.get_concrete_function(),
        'save':
            m.save.get_concrete_function(),
        'restore':
            m.restore.get_concrete_function(),
    })

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [
   tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
   tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()
