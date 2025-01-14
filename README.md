# DIabetes-predection
Uses various cosine similarities to predict diabetes
Diabetes Prediction Using Machine Learning
Overview
This project predicts the likelihood of diabetes in individuals based on health parameters such as glucose level, blood pressure, BMI, and more. The machine learning model utilizes a Support Vector Machine (SVM) with a linear kernel to classify individuals as diabetic or non-diabetic.

Features
Data Preprocessing: Includes data standardization using StandardScaler.
Exploratory Data Analysis: Analyzing dataset characteristics and identifying statistical insights.
Model Training and Testing:
Uses an 80-20 train-test split.
Trains an SVM model with a linear kernel.
Accuracy Evaluation: Provides accuracy metrics for both training and test datasets.
Prediction System: Implements a predictive system for new input data.
Technologies Used
Programming Language: Python
Libraries:
numpy for numerical computations.
pandas for data manipulation.
scikit-learn for preprocessing, model building, and evaluation.
Dataset
The dataset used is Pima Indians Diabetes Dataset, containing the following fields:

Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI
Diabetes Pedigree Function
Age
Outcome (0 = Non-Diabetic, 1 = Diabetic)
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction
Install the required Python packages:

bash
Copy code
pip install -r requirements.txt
Run the script:

bash
Copy code
python diabetes_prediction.py
Usage
Load the dataset: The script reads diabetes.csv.
Preprocess the data: Standardizes the features.
Train the model: Trains an SVM classifier.
Test the model: Evaluates the accuracy on test data.
Make predictions: Use input_data to test with custom values.
Results
Training Accuracy: ~78.66%
Test Accuracy: ~77.27%
Example
Input Example:

python
Copy code
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
Run the predictive system:

bash
Copy code
python diabetes_prediction.py
Output:

csharp
Copy code
The person is Diabetic.
Future Improvements
Implement additional machine learning models for comparison.
Tune hyperparameters to improve accuracy.
Add data visualization for better insights.
