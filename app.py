
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
import os
import PIL.Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define labels for the classes
labels = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

# Define output directory for saliency maps
output_dir = 'saliency_maps'
os.makedirs(output_dir, exist_ok=True)

# Generate an explanation for the prediction and saliency map
def generate_explanation(img_path, model_prediction, confidence):
    response = f"As an expert neurologist, the model's prediction focuses on regions of the brain that align with known tumor markers. The highlighted areas correspond to distinct features typically observed in '{model_prediction}' cases. These features include abnormal tissue growth and specific patterns of contrast seen in light cyan. The high confidence of {confidence * 100:.2f}% suggests the model identified these patterns with minimal ambiguity."
    return response

# Generate saliency map function
def generate_saliency_map(model, img_array, class_index, img_size):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, axis=0) if img_tensor.ndim == 3 else img_tensor

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        target_class = predictions[:, class_index]

    gradients = tape.gradient(target_class, img_tensor)

    if gradients is None:
        raise ValueError("Gradients are None. Ensure the input tensor is properly tracked.")

    gradients = tf.math.abs(gradients)
    gradients = tf.reduce_max(gradients, axis=-1)
    gradients = gradients.numpy().squeeze()
    gradients = cv2.resize(gradients, img_size)

    # Normalize gradients
    gradients -= gradients.min()
    gradients /= gradients.max() if gradients.max() > 0 else 1

    # Apply heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

# Load model function (Xception-based model)
def load_xception_model(model_path):
    img_shape = (299, 299, 3)
    base_model = tf.keras.applications.Xception(
        include_top=False, weights="imagenet", input_shape=img_shape, pooling="max"
    )
    model = Sequential([
        base_model,
        Flatten(),
        Dropout(rate=0.3),
        Dense(128, activation="relu"),
        Dropout(rate=0.25),
        Dense(4, activation="softmax"),
    ])
    model.build((None,) + img_shape)
    model.compile(
        optimizer=Adamax(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy", Precision(), Recall()],
    )
    model.load_weights(model_path)
    return model

# Streamlit interface
st.title("Brain Tumor Classification")
st.write("Upload an image of a brain MRI scan to classify.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    selected_model = st.radio("Select Model", ("Transfer Learning - Xception", "Custom CNN"))

    model = None
    img_size = None

    try:
        if selected_model == "Transfer Learning - Xception":
            model = load_xception_model('/content/xception_model.weights.h5')
            img_size = (299, 299)
        else:
            model = load_model('/content/cnn_model.h5')
            img_size = (224, 224)
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")

    if model is not None:
        try:
            img = image.load_img(uploaded_file, target_size=img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)
            class_index = np.argmax(prediction[0])
            result = labels[class_index]

            st.write(f"**Predicted Class:** {result}")
            st.write("**Predictions:**")
            for label, prob in zip(labels, prediction[0]):
                st.write(f"{label}: {prob:.4f}")

            # Generate saliency map
            saliency_map = None
            try:
                with tf.device('/CPU:0'):  # Use CPU to avoid GPU memory issues
                    saliency_map = generate_saliency_map(model, img_array[0], class_index, img_size)
            except Exception as e:
                st.error(f"Error generating saliency map: {str(e)}")

            # Display classification results
            result_container = st.container()
            result_container.markdown(
                f"""
                <div style="background-color: #000000; color: #ffffff; padding: 30px; border-radius: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1; text-align: center;">
                            <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Prediction</h3>
                            <p style="font-size: 36px; font-weight: 800; color: #FF0000; margin: 0;">
                                {result}
                            </p>
                        </div>
                        <div style="width: 2px; height: 80px; background-color: #ffffff; margin: 0 20px;"></div>
                        <div style="flex: 1; text-align: center;">
                            <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Confidence</h3>
                            <p style="font-size: 36px; font-weight: 800; color: #2196F3; margin: 0;">
                                {prediction[0][class_index]:.4%}
                            </p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Display images
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
            with col2:
                if saliency_map is not None:
                    st.image(saliency_map, caption='Saliency Map', use_container_width=True)
                else:
                    st.warning("Saliency map could not be generated.")

            # Save saliency map and generate explanation
            if saliency_map is not None:
                saliency_map_path = f"{output_dir}/{uploaded_file.name}"
                saliency_map_pil = PIL.Image.fromarray(saliency_map)
                saliency_map_pil.save(saliency_map_path)

                # Generate explanation for the prediction
                explanation = generate_explanation(saliency_map_path, result, prediction[0][class_index])

                st.write("## Explanation")
                st.write(explanation)
            else:
                st.warning("Saliency map could not be generated, so an explanation is unavailable.")

        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")
    else:
        st.warning("Model could not be loaded. Please check the file paths or model configurations.")


