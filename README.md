# 10_Student-Club-Participation-Prediction_Anubhav_202401100400040
Load and Train Model
Ensure you have a trained model named model, and split data into X_train, X_test, y_train, y_test.

Predict New Student
import pandas as pd

# Example student data as DataFrame (with correct feature names)
new_student = pd.DataFrame([[7, 10]], columns=X.columns)

# Predict using the trained model
prediction = model.predict(new_student)
probability = model.predict_proba(new_student)

# Output result
result = "Yes" if prediction[0] == 1 else "No"
print(f"🎓 Will the student join a club? → {result}")
print(f"📊 Prediction Probability: Yes = {probability[0][1]:.2f}, No = {probability[0][0]:.2f}")

Plot Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

Repository Structure
├── data/                  # Your CSV data files
├── models/                # Saved trained models
├── notebooks/             # Jupyter notebooks
├── scripts/               # Python scripts for training and evaluation
└── README.md              # This file
