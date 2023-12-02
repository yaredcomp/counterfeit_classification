import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# Define the available models
model_paths = {
    'Dense121 For New Banknotes': 'models/dense121.tflite',
    'Dense121 For Worn-out Banknotes': 'models/Dense121_best_weights_mixed-III.tflite',
    'Vgg19 For New Banknotes': 'models/Vgg19.tflite',
    'Vgg19 For Worn-out Banknotes': 'models/Vgg19_mixed.tflite',

}

def load_selected_model(model_path):
    return tf.lite.Interpreter(model_path=model_path)

# Define the classes
classes = ['genuine_200_etb', 'counterfeit_200_etb', 'genuine_100_etb', 'counterfeit_100_etb']

# Helper function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    return image

# Helper function to predict the image
def predict_image(model, image):
    processed_image = preprocess_image(image)
    expanded_image = np.expand_dims(processed_image, axis=0)

    # Make predictions
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], expanded_image)
    model.invoke()
    predictions = model.get_tensor(output_details[0]['index'])

    top_classes = np.argsort(predictions, axis=1)[0][-4:][::-1]
    top_confidences = predictions[0][top_classes]

    return top_classes, top_confidences

# Streamlit app
def main():
    st.title("Counterfeit and Genuine Ethiopian Banknote Classification by Yared D.")
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # Navigation menu to select the model
    selected_model = st.sidebar.selectbox("Select a model along with the Banknote Condition", list(model_paths.keys()))

    # Load the selected model
    model_path = model_paths[selected_model]
    model = load_selected_model(model_path)

    # File upload
    uploaded_files = st.file_uploader("Choose multiple image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.button("Classify All"):
        for uploaded_file in uploaded_files:
            # Read and preprocess the uploaded image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

            # Predict the image
            top_classes, top_confidences = predict_image(model, image)

            # Create two columns for image and classification result
            col1, col2 = st.columns(2)

            # Display the image in the first column
            resized_image = cv2.resize(image, (224, 224))
            col1.image(resized_image, caption='Uploaded Image', use_column_width=False)

            # Display classification result
            for i in range(len(top_classes)):
                class_name = classes[top_classes[i]]
                confidence = top_confidences[i]
                st.write(f"Class: {class_name}, Confidence: {confidence * 100:.2f}%")

            st.write('---')

# Run the app
if __name__ == '__main__':
    main()
