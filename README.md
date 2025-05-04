# 🍎 ML-M: Fruit Disease Classifier

A Streamlit-based machine learning web app that detects fruit or foliage diseases using CNN.

## 💻 Features
- Upload fruit images to detect diseases
- Visualize model confidence
- Trained using TensorFlow and Keras

## 📁 Structure
- `app/fruit_classifier_streamlit_app.py`: Streamlit app
- `app/fruit_quality_cnn_model.h5`: Trained model (Git LFS)
- `app/class_indices.json`: Label index
- `app/training_plots.png`: Training accuracy/loss

## 🚀 How to Run

1. Clone the repo:
    ```bash
    git clone https://github.com/momotherealone0/ML-M.git
    cd ML-M
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app/fruit_classifier_streamlit_app.py
    ```

## 📦 Requirements
See `requirements.txt`.

## 🧠 Model
CNN trained on fruit images with data augmentation, using categorical crossentropy.
