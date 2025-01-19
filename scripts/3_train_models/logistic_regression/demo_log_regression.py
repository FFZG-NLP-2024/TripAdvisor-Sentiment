import gradio as gr
import tensorflow as tf

# Load a pre-trained model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Define a function that classifies images
def classify_image(image):
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    
    predictions = model.predict(image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    
    return {label: prob for (imagenet_id, label, prob) in decoded_predictions}

# Create the Gradio interface
interface = gr.Interface(fn=classify_image, inputs="image", outputs="json")

# Launch the interface
interface.launch()