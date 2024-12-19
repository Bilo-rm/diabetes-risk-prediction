Diabetes Risk Prediction

This project predicts the likelihood of a person having diabetes based on medical attributes using machine learning and neural networks. The aim is to explore and compare traditional ML methods (Random Forest) and a deep learning approach using TensorFlow.

Table of Contents:
1.Project Description
2.Technologies Used
3.Installation and Setup
4.How to Run the Project
5.Evaluation Metrics


1.Project Description:

Diabetes is a chronic condition that requires early detection for effective management. This project uses a dataset of medical features such as glucose levels, BMI, and age to predict diabetes risk. The solution involves:

Data Preprocessing: Cleaning, handling missing values, and scaling.
Machine Learning Model: Random Forest Classifier.
Deep Learning Model: A simple feedforward neural network with TensorFlow.
Evaluation Metrics: Precision, recall, F1-score, accuracy, and ROC-AUC.


2.Technologies Used:
Python 3.13

Libraries:
pandas - Data manipulation.
numpy - Numerical computations.
scikit-learn - Machine learning algorithms and preprocessing.
tensorflow - Neural network implementation.
matplotlib - Visualization.
joblib - Saving and loading models.

Dataset: Pima Indians Diabetes Dataset 
link:https://www.kaggle.com/datasets/mathchi/diabetes-data-set?resource=download


3.Installation and Setup
1. Clone the Repository

git clone https://github.com/yourusername/diabetes-risk-prediction.git
cd diabetes-risk-prediction

2. Set Up a Virtual Environment
Create and activate a virtual environment:

python -m venv venv
# Activate on Windows
source venv/Scripts/activate
# Activate on Mac/Linux
source venv/bin/activate


3. Install Required Libraries
Install the dependencies from the requirements.txt file:

pip install -r requirements.txt


5.How to Run the Project:
Run the Main Script Execute the following command to preprocess data, train models, and view results:

python main.py


6.Evaluation Metrics:

The models are evaluated using:

Accuracy: Percentage of correct predictions.
Precision: Correctly predicted positives / Total predicted positives.
Recall (Sensitivity): Correctly predicted positives / Total actual positives.
F1-Score: Harmonic mean of precision and recall.
ROC-AUC: Measures the model's ability to distinguish between classes.
