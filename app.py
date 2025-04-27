import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Cấu hình TensorFlow để tối ưu hiệu suất
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

@st.cache_resource # Cache the model for faster loading.
def load_model():
    model = tf.keras.models.load_model(
        "unet_model.keras",
        custom_objects={"mean_iou_coef": lambda y_true, y_pred: tf.reduce_mean(tf.keras.metrics.mean_iou(y_true, y_pred, num_classes=3))}
    )
    return model

model = load_model()

# Image processing and prediction function
def predict_seg(image):
    # Convert image to tensor.
    original_size = image.size
    img = image.resize((128, 128)).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = img_array[np.newaxis, ...]
    
    # Predict
    pred = model.predict(img_array)
    pred = np.argmax(pred[0], axis=-1) # Take the label
    
    # Convert the predicted labels into colors.
    mask_color = np.zeros((128, 128, 3), dtype=np.uint8)
    mask_color[pred == 0] = (255, 0, 0) # Pet: Red
    mask_color[pred == 1] = (0, 0, 0)   # Backgound: Black
    mask_color[pred == 2] = (0, 255, 0) # Border: Green
    
    mask = Image.fromarray(mask_color)
    mask = mask.resize(original_size)
    return mask

# Streamlit interface
st.title("Segmentation")
st.write("Upload an image to segment")

#Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image.
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", width=350)
    
    with st.spinner("Predicting..."):
        mask = predict_seg(image)
        with col2:
            st.image(mask, caption="Segmentation Mask", width=350)
            
            mas = io.BytesIO()
            mask.save(mas, format="PNG")
            byte_im = mas.getvalue()
            
            st.download_button("Download Mask", data=byte_im, file_name="mask_seg.png")
        
if __name__ == "__main__":
    st.write("Running on http://localhost:8501")