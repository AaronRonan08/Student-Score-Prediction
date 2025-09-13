# Student Score Prediction 📘

## 📌 Problem Statement
Predict student exam scores based on the number of study hours.

## 📊 Dataset
A small dataset with 2 columns:
- Hours studied
- Scores obtained

## 🔧 Workflow
1. Load dataset using pandas
2. Split into training and testing sets
3. Train a Linear Regression model (scikit-learn)
4. Predict scores
5. Evaluate with R² score
6. Visualize with matplotlib

## ✅ Results
- Predicted score for 7 hours = **~71**
- Model accuracy (R² score) = **95%**
- Graph shows a strong positive relationship between study hours and scores.

## 🛠 Requirements
- Python 3.x
- pandas
- scikit-learn
- matplotlib

## 🚀 How to Run
```bash
python model.py
