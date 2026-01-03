# Distracted Driver Detection using CNN & VGG16
A deep learning–based computer vision project to **detect and classify distracted driving behaviors** using **Custom CNNs** and **VGG16 (Transfer Learning)**. The system identifies unsafe driver actions in real-time with high accuracy.

---

## Problem Statement

Distracted driving is a major cause of road accidents worldwide. This project builds an ML model that:

* Detects whether a driver is distracted
* Identifies the **type of distraction**
* Works reliably under real-world driving conditions

---

## 🧠 Models Implemented

* **Custom CNN**

  * Batchwise training
  * Non-batch training
    
* **VGG16 (Transfer Learning)**

  * Fine-tuned batchwise
  * Fine-tuned non-batchwise

All models classify **10 driver states** with ~**99% test accuracy**.

---

## Dataset

**StateFarm Distracted Driver Detection Dataset (Kaggle)**

**Classes:**

* Safe driving
* Texting (left/right)
* Talking on phone (left/right)
* Operating the radio
* Drinking
* Reaching behind
* Hair & makeup
* Talking to the passenger

~22K training images, ~79K test images (RGB, in-car images)

---

## Tech Stack

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy, Pandas
* Matplotlib
* Scikit-learn

---

## Results

| Model             | Test Accuracy |
| ----------------- | ------------- |
| CNN (Non-Batch)   | **99.88%**    |
| CNN (Batchwise)   | 99.79%        |
| VGG16 (Non-Batch) | 99.86%        |
| VGG16 (Batchwise) | 99.46%        |

## CNN Batchwise Video Demo

[Click to watch the demo](assets/cnn_batchwise_output_video.mp4)


**Key Insights**

* Batchwise training → smoother convergence
* Non-batch training → noisier but sometimes higher peak accuracy
* VGG16 shows strong feature extraction
* Custom CNN generalizes exceptionally well

---

## Applications

* Advanced Driver Assistance Systems (ADAS)
* Autonomous / semi-autonomous vehicles
* In-vehicle safety monitoring
* Smart traffic systems

---

## Future Work

* Driver drowsiness detection
* Edge-device deployment
* Integration with traffic cameras
* Robustness under low-light & varying angles

---

## Authors

* **Manya Goel**
* **Diya Budlakoti**


# DistractedDriver

vgg16 fine tuned nonbatch : 64,64
vgg16 fine tuned batchwise: 224,224
cnn non batch: 128, 128
cnn batchwise: 224, 224
shallow non batch: 224, 224
shallow deep non batch: 64, 64
