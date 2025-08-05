# 🔬 AI-Driven Prediction Model for Chronic Kidney Disease (CKD) Diagnosis

This project presents an AI-powered system for **early and accurate detection of Chronic Kidney Disease (CKD)** using **deep learning models** on medical images (such as kidney scans). The system is designed to assist healthcare professionals by providing **automated, non-invasive**, and **real-time diagnosis** of CKD.

---

##  Objective

To develop a robust and scalable AI model that can:
- Detect CKD at early stages.
- Improve diagnostic accuracy.
- Minimize reliance on expert opinion.
- Enable real-time predictions in low-resource environments.

---

##  Models Used

Four deep learning architectures were implemented and compared:
- **Convolutional Neural Networks (CNN)** – For structural and textural feature extraction.
- **MobileNet** – Lightweight and optimized for deployment on edge devices.
- **Vision Transformer (ViT)** – Captures long-range spatial dependencies with attention mechanisms.
- **MobileNet + LSTM Hybrid** – Combines spatial and temporal analysis for disease progression detection.

---

## ⚙ Methodology

1. **Data Collection**: Medical kidney scan images sourced from Kaggle/public datasets.
2. **Preprocessing**: Image resizing, normalization, feature extraction.
3. **Model Training**: Each model trained using 80% of the dataset; 20% used for evaluation.
4. **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC.
5. **User Interface**: Flask-based web interface for image upload and diagnosis result display.

---

##  Tech Stack

- **Frontend**: HTML, CSS, JS, Bootstrap
- **Backend**: Python (Flask), TensorFlow, PyTorch
- **Database**: MySQL
- **Deployment**: XAMPP Server, VSCode IDE

---

##  Results

- **MobileNet + LSTM and ViT** outperformed other models with **up to 93% accuracy**.
- All models demonstrated high performance in detecting multiple CKD stages.
- Class 2 remained the most challenging for all models, needing further refinement.

---

##  Use Case

- Can be deployed in **clinics, hospitals, and rural healthcare centers**.
- Supports **non-specialist users** through a user-friendly prediction portal.
- Enables **early intervention and better treatment planning**.

---

##  Future Enhancements

- Integrate **larger and more diverse datasets**.
- Add **clinical data integration** alongside images.
- Improve real-time performance and add **explainable AI** components.
- Expand to other renal disorders and imaging modalities.

---

