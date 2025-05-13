readme_content = """
# 🚦 Traffic Sign Recognition with CNN

This project implements a deep learning-based traffic sign classification system using the [GTSRB (German Traffic Sign Recognition Benchmark)](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news) dataset. It includes data preprocessing, CNN training, evaluation, and an interactive Streamlit web application for image and real-time webcam prediction.

---

## 📁 Project Structure
├── GTSRB.zip # Dataset (extracted into Train/Test folders)
├── app.py # Streamlit app for prediction and performance visualization
├── GTSRBmodel.keras # Trained CNN model
├── X_test.npy / y_test.npy # Preprocessed test dataset
├── class_indices.json # Label-index mapping
├── history.pkl # Training history (accuracy/loss)
├── realtime_detect.py # Real-time webcam detection script
├── static.py # Helper functions (optional utilities)
├── sample_test/ # Sample test images for manual upload
└── Python_Project.ipynb # Jupyter notebook for data processing and training



---

## 🧠 Model Overview

- **Architecture:** Convolutional Neural Network (CNN)
- **Input Size:** 48x48 RGB
- **Classes:** 43 traffic sign types
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Regularization:** Dropout Layers
- **Epochs:** 15 (EarlyStopping applied)
- **Test Accuracy:** ~96.4%

---

## 📊 Key Features

- 📦 **Dataset Handling**: GTSRB ZIP unpacked and parsed via CSV files  
- 🔄 **Data Augmentation**: Rotation, zoom, width/height shift  
- 🧠 **CNN Model**: Built and trained using TensorFlow/Keras  
- 🧪 **Evaluation**: Accuracy, loss plots, confusion matrix, and classification report  
- 📷 **Streamlit Web App**:  
  - Upload image for prediction  
  - See top-5 prediction confidence  
  - Real-time webcam traffic sign detection  
  - Visual performance metrics and class info  

---

## 🚧 Challenges Addressed

- **Label Mismatch:** TensorFlow's label indexing conflicted with original test CSV labels.  
  ➤ Solved by saving `class_indices.json` during training and reversing it during evaluation.  
- **Human-Readable Output:** A `class_info` dictionary was used to map predicted indices to descriptive names (e.g., “Speed Limit 30”).

---

## ▶ How to Run the Streamlit App

```bash
# 1. Clone this repo and install dependencies
pip install -r requirements.txt

# 2. Start the Streamlit app
streamlit run app.py


## Download thease large files 
GTSRB data set :- https://drive.google.com/file/d/1QJAHDjtyg7oefFnKC7Lgw6QLfIKEfQMQ/view?usp=sharing

X_test.npy :- https://drive.google.com/file/d/1LPx-xldLORMh2rOnzC5HF3SVbUEWnfQz/view?usp=sharing

