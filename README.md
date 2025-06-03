# 🩺 Heart Disease Prediction with Decision Trees & Random Forests

This project explores how tree-based machine learning models like **Decision Trees** and **Random Forests** can be used to predict the presence of heart disease using patient data.

## 📌 Objective

Learn and apply tree-based models for **classification**, analyze overfitting, interpret important features, and compare performance using cross-validation.

## 📁 Dataset

I have used the [Heart Disease dataset from Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). It includes various health-related features like cholesterol levels, resting ECG results, blood pressure, etc., and a target label indicating the presence of heart disease.

### Features in the dataset:
- Age, Sex, Chest Pain Type
- Resting Blood Pressure, Cholesterol
- Fasting Blood Sugar, Rest ECG, Max Heart Rate
- Exercise Induced Angina, ST depression, and more...

## 🛠️ Tools & Libraries

- Python
- scikit-learn
- matplotlib, seaborn
- pandas, numpy

## 🔍 What the Script Does

✅ Loads and prepares the dataset  
✅ Splits the data into training and test sets  
✅ Trains a **Decision Tree Classifier** and visualizes it  
✅ Tunes tree depth to reduce overfitting  
✅ Trains a **Random Forest Classifier** and compares accuracy  
✅ Displays feature importances  
✅ Evaluates both models using 5-fold cross-validation  

## 📊 Output

- Visualized decision tree plot
- Feature importance graph (bar chart)
- Accuracy scores and classification reports for both models
- Cross-validation results

## 🚀 How to Run

1. Download or clone the repository.
2. Make sure you have Python and the required libraries installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   
---

Let me know if you'd like to add badges, a license section, or a sample output screenshot to this `README.md`.

