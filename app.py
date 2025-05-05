import streamlit as st
import numpy as np
import cv2
import os
import pickle
import json
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="🚦 Traffic Sign Recognition", layout="wide")

# --- Constants ---
MODEL_PATH = "GTSRBmodel.keras"
IMG_HEIGHT, IMG_WIDTH = 48, 48

# --- Load model ---
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# --- Class Labels ---
class_info = {
    0: ('0', "Speed Limit 20"), 1: ('1', "Speed Limit 30"), 2: ('10', "No passing > 3.5 tons"),
    3: ('11', "Right-of-way"), 4: ('12', "Priority road"), 5: ('13', "Yield"),
    6: ('14', "Stop"), 7: ('15', "No vehicles"), 8: ('16', "No vehicles > 3.5 tons"),
    9: ('17', "No entry"), 10: ('18', "General caution"), 11: ('19', "Left curve"),
    12: ('2', "Speed Limit 50"), 13: ('20', "Right curve"), 14: ('21', "Double curve"),
    15: ('22', "Bumpy road"), 16: ('23', "Slippery road"), 17: ('24', "Road narrows"),
    18: ('25', "Road work"), 19: ('26', "Traffic signals"), 20: ('27', "Pedestrians"),
    21: ('28', "Children crossing"), 22: ('29', "Bicycles crossing"), 23: ('3', "Speed Limit 60"),
    24: ('30', "Ice/snow"), 25: ('31', "Wild animals"), 26: ('32', "End restrictions"),
    27: ('33', "Turn right"), 28: ('34', "Turn left"), 29: ('35', "Ahead only"),
    30: ('36', "Go straight or right"), 31: ('37', "Go straight or left"), 32: ('38', "Keep right"),
    33: ('39', "Keep left"), 34: ('4', "Speed Limit 70"), 35: ('40', "Roundabout"),
    36: ('41', "End no passing"), 37: ('42', "End no passing > 3.5 tons"),
    38: ('5', "Speed Limit 80"), 39: ('6', "End Speed Limit 80"),
    40: ('7', "Speed Limit 100"), 41: ('8', "Speed Limit 120"), 42: ('9', "No passing")
}

# --- UI Tabs ---
tab1, tab2, tab3,tab4 = st.tabs(["📷 Predict", "📊 Performance", "📚 Class Info","Real Time detection"])

# --- TAB 1: Prediction ---
with tab1:
    st.header("📷 Upload a Traffic Sign Image")
    st.markdown("Use the uploader below to classify a traffic sign using the trained CNN model.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        col1, col2 = st.columns([1, 2])
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            img = image.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            class_id, class_label = class_info[predicted_class]

            st.subheader("🔍 Prediction")
            st.success(f"**{class_label}** with confidence **{confidence * 100:.2f}%**")

            st.subheader("🔝 Top-5 Predictions")
            top5_indices = prediction[0].argsort()[-5:][::-1]
            for idx in top5_indices:
                top_label = class_info[idx][1]
                top_conf = prediction[0][idx]
                st.write(f"{top_label}: {top_conf * 100:.2f}%")

# --- TAB 2: Performance ---
with tab2:
    st.header("📊 Training & Test Evaluation")

    if os.path.exists("history.pkl"):
        with open("history.pkl", "rb") as f:
            history = pickle.load(f)

        with st.expander("📈 Accuracy & Loss Curves"):
            fig1, ax1 = plt.subplots()
            ax1.plot(history["accuracy"], label="Train Accuracy")
            ax1.plot(history["val_accuracy"], label="Val Accuracy")
            ax1.set_title("Accuracy over Epochs")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Accuracy")
            ax1.legend()
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ax2.plot(history["loss"], label="Train Loss")
            ax2.plot(history["val_loss"], label="Val Loss")
            ax2.set_title("Loss over Epochs")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.legend()
            st.pyplot(fig2)
    else:
        st.warning("🚫 `history.pkl` not found.")

    if os.path.exists("X_test.npy") and os.path.exists("y_test.npy") and os.path.exists("class_indices.json"):
        st.subheader("🧪 Model Evaluation on Test Set")

        X_test = np.load("X_test.npy")
        y_test = np.load("y_test.npy")

        with open("class_indices.json", "r") as f:
            class_indices = json.load(f)
        idx_to_label = {v: int(k) for k, v in sorted(class_indices.items(), key=lambda item: int(item[0]))}
        label_names = [str(idx_to_label[i]) for i in range(len(idx_to_label))]

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_pred_labels = np.array([idx_to_label[i] for i in y_pred_classes])

        accuracy = accuracy_score(y_test, y_pred_labels)
        st.metric("🧮 Test Accuracy", f"{accuracy * 100:.2f}%")

        with st.expander("📉 Confusion Matrix"):
            cm = confusion_matrix(y_test, y_pred_labels)
            fig_cm, ax_cm = plt.subplots(figsize=(15, 15))

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap="BuGn",
                xticklabels=label_names,
                yticklabels=label_names,
                cbar=True,
                linewidths=0.1,
                square=True,
                ax=ax_cm
            )

            ax_cm.set_title("Confusion Matrix", fontsize=18, fontweight='bold', pad=12)
            ax_cm.set_xlabel("Predicted Label", fontsize=14, labelpad=10)
            ax_cm.set_ylabel("True Label", fontsize=14, labelpad=10)
            st.pyplot(fig_cm)

        with st.expander("📋 Classification Report"):
            report = classification_report(y_test, y_pred_labels, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report)
    else:
        st.warning("🚫 `X_test.npy`, `y_test.npy`, or `class_indices.json` not found.")

# --- TAB 3: Class Info ---
with tab3:
    st.header("📚 Traffic Sign Classes")
    st.markdown("Below is the complete list of traffic sign classes used in the model.")

    df_class = pd.DataFrame([{"Class ID": v[0], "Label": v[1]} for k, v in class_info.items()])
    st.dataframe(df_class, use_container_width=True)

# --- TAB 4: Real-Time Detection ---
with tab4:
    st.header("📹 Real-Time Traffic Sign Detection")
    st.markdown("This uses your webcam feed to detect and classify traffic signs in real-time.")

    run_button = st.button("▶ Start Detection")
    stop_button = st.button("⏹ Stop Detection")

    if run_button:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.warning("🚫 Failed to read from webcam.")
                break

            h, w, _ = frame.shape
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            roi = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

            img = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > 0.70:
                class_label = class_info[class_id][1]
                class_code = class_info[class_id][0]
                label = f"{class_code} | {class_label} ({confidence * 100:.2f}%)"
                cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB")

        cap.release()
        stframe.empty()
        st.success("✅ Detection stopped.")
