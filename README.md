 Real-Time Face Mask Detection System 

This project is an AI-based system that detects whether a person is wearing a face mask in real time using a webcam. It uses MobileNetV2, TensorFlow/Keras, and OpenCV for face detection and mask classification.

 Features

 Real-time mask detection via webcam
 Deep learning model built using MobileNetV2
 Trained on face mask dataset
 Detects and displays “Mask” or “No Mask” with color indicators

 Project Structure

train.py - Used to train the CNN model (MobileNetV2)
app.py - Runs real-time mask detection using a webcam
mask_detector.h5 - Trained model file
requirements.txt - Dependencies for running the project 



 Dataset

The dataset used for training is **not included** in this repository due to size limits.
You can download a similar dataset from [Kaggle - Face Mask Detection Dataset](https://www.kaggle.com/datasets).

 Installation and Usage

1. Clone this repository:

   git clone https://github.com/your-username/Real-time-face-mask-detection.git
 
2. Install required libraries:

   pip install -r requirements.txt

3. To train the model :

   python train.py
   
4. To run real-time detection:
 
   python app.py

 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* MobileNetV2
* NumPy

 Output

The system shows webcam video with text:

Green - Mask detected

Red - No mask detected



