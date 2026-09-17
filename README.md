# 🏔️ Intel Image Classification & Area Detection

An end-to-end Computer Vision project focused on classifying natural and urban landscapes into 6 distinct categories using MobileNetV2 transfer learning, deployed as an interactive web interface with Streamlit.

---

## 📌 Dataset Overview

The dataset consists of **150x150 RGB images** divided into 6 distinct classes, evaluated for data integrity and class balance during exploratory data analysis (EDA).

### 📋 Key Dataset Statistics
* **Total Classes:** 6 (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`)
* **Image Dimensions:** Standardized to **150x150**
* **Color Mode:** RGB
* **Data Health:** 0 corrupted or unreadable images
* **Class Balance:** Balanced distribution across all categories

| Class Name | Train Image Count |
| :--- | :--- |
| **Glacier** | 2,404 |
| **Mountain** | 2,512 |
| **Street** | 2,382 |
| **Sea** | 2,274 |
| **Forest** | 2,271 |
| **Buildings** | 2,191 |

---

## ⚙️ Project Pipeline & Strategy

1. **Exploratory Data Analysis (EDA):** Image dimension checks, color distribution analysis, and class ratio verification (`task4_final.ipynb`).
2. **Train/Validation Split Strategy:** Dynamic validation set creation from training data to avoid data leakage while keeping test data untouched.
3. **Model Architecture:** Fine-tuned MobileNetV2 deep neural network saved as `best_intel_mobilenetv2.keras`.
4. **Web Deployment:** Real-time user interface powered by Streamlit (`app.py`).

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x
* **Deep Learning Framework:** TensorFlow / Keras
* **Web Framework:** Streamlit
* **Data & Image Processing:** NumPy, Pandas, OpenCV / PIL, Matplotlib


   ```bash
   git clone [https://github.com/omarggtbeh-eng/Area_Detection_Project-DL-.git](https://github.com/omarggtbeh-eng/Area_Detection_Project-DL-.git)
   cd Area_Detection_Project-DL-
