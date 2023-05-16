import tensorflow as tf

# Load the model
interpreter = tf.lite.Interpreter(model_path="modelnew.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Details:", input_details)
print("Output Details:", output_details)