

# **Clash Royale Emote Detector**

A machine learning–powered emote recognition system built using **OpenCV**, **scikit-learn**, **NumPy**, and **Pygame**. This project detects and classifies Clash Royale emotes by combining image processing, feature engineering, and ML classification techniques.

---

## 🚀 **Features**

* **Emote Detection Pipeline:** Uses OpenCV to preprocess images with grayscale conversion, thresholding, contour extraction, and noise removal.
* **Machine Learning Classifier:** Trains a scikit-learn model to recognize emotes from extracted feature vectors.
* **Efficient Feature Engineering:** NumPy is used for pixel-level operations and matrix computations to speed up preprocessing.
* **Interactive UI:** Pygame-based interface for real-time emote display and detection with confidence scores.
* **High Performance:** Optimized preprocessing and ML parameters for faster predictions and smooth runtime.

---

## 🛠️ **Tech Stack**

* **Python 3.x**
* **OpenCV (opencv-python)**
* **NumPy**
* **scikit-learn**
* **Pygame**

---

## 📦 **Installation**

Clone the repository:

```bash
git clone https://github.com/your-username/clash-royale-emote-detector.git
cd clash-royale-emote-detector
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

*(Your requirements file should contain: opencv-python, numpy, scikit-learn, pygame.)*

---

## ▶️ **How to Run**


```bash
python main.py
```

This:

* Opens the Pygame UI
* Lets you upload/view emotes
* Shows detection results in real time

---


## 🧠 **How It Works**

1. **Preprocessing with OpenCV**

   * Converts images to grayscale
   * Removes noise
   * Extracts relevant contours & features

2. **Feature Vector Creation**

   * Uses NumPy to generate flattened pixel arrays
   * Normalizes data

3. **Model Training**

   * scikit-learn classifier
   * Hyperparameter tuning
   * Accuracy evaluation

4. **Real-Time Detection**

   * Pygame window displays emotes
   * Model predicts label + confidence

---

## 📊 **Results**

* Accurate emote classification after optimization
* Real-time detection supported through efficient preprocessing and light-weight ML model


---

## 📜 **License**

This project is open-source. You may modify and use it for personal or educational purposes.

---

