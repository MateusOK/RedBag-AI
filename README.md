![banner](https://github.com/user-attachments/assets/3b92c249-3956-49c8-88fb-811fa400f6e4)
<h1 align="center">🧠 RedBag-AI</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
  <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
</p>

<p align="center">
  <a href="https://bitbucket.org/lbesson/ansi-colors"><img src="https://img.shields.io/badge/Maintained%3F-no-red.svg" /></a>
  <a href="https://github.com/Naereen/StrapDown.js/blob/master/LICENSE"><img src="https://img.shields.io/github/license/Naereen/StrapDown.js.svg" /></a>
  <a href="https://GitHub.com/Naereen/ama"><img src="https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg" /></a>
</p>

## 📌 **Project Overview**

The **RedBag-AI** is the machine learning core of the RedBag project, developed at Fatec Registro. This repository contains the code responsible for training the predictive model used in the **RedBag-Predictor**, which performs cataract diagnosis in dogs through image classification.

Using **Convolutional Neural Networks (CNNs)**, the model is trained to classify images into two categories: **Healthy** and **Unhealthy**, ensuring a fast and efficient preliminary diagnosis.

👉 Check out the [RedBag-Predictor](https://github.com/MateusOK/RedBag-Predictor) to see how this model is integrated into the system.

---

## 🚀 **Getting Started**

### **Prerequisites**

Before running this project, ensure you have the following installed:

- [Python](https://www.python.org/downloads/)
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)

You can install all dependencies using:

```bash
pip install -r requirements.txt
```
### **Cloning the Repository**

Clone this project to your local machine:

```bash
git clone https://github.com/MateusOK/RedBag-AI
cd RedBag-AI
```
### **Dataset**

The model is trained on a dataset of canine eye images categorized as Healthy or Unhealthy. 
Ensure you have the dataset properly structured inside the `data/` directory before running the training script.

Expected structure:
```
data/
├── train/
│   ├── healthy/
│   ├── unhealthy/
├── valid/
│   ├── healthy/
│   ├── unhealthy/
├── test/
│   ├── healthy/
│   ├── unhealthy/
```

## **🏋️ Training the Model**

To start training the model, simply run:

```bash
python dnn.py
```
This will process the dataset, train the CNN model, and save the trained weights for later use.

### **🔍 Classifying an Image**

To classify an image using the trained model, modify the paths in `classify_a_image.py`:

```python
# Load the trained model
model = load_model("path_to_model_trained")

# Path to the image to analyze
img_path = "path_to_image_to_analyze"
```
Run the script:

```bash
python classify_a_image.py
```
This script loads a trained model and classifies a given image as Healthy or Unhealthy based on the model’s predictions.

#### Example Output:
```
Predicted class: Unhealthy
Probability: 87.45%
```

---
## 🤝 **Collaborators**

Special thanks to all contributors to this project:

- [Mateus Ribeiro](https://www.linkedin.com/in/dev-mateus-ribeiro)
- [Gustavo Eyros](https://www.linkedin.com/in/gustavoeyros)
- [Rian Yuri](https://www.linkedin.com/in/rian-yuri-b36563158/)
- [Luiz Lopes](https://www.linkedin.com/in/luizlopes12)

---
This project is licensed under the [MIT License](LICENSE).
