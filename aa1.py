# student_scores_fallback.py
import sys
import numpy as np

# Try to import libraries and give friendly install hints if missing
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
except Exception as e:
    print("Missing a required package. Run the following in your system terminal:")
    print(f"    {sys.executable} -m pip install pandas scikit-learn matplotlib")
    print("Then re-run this script in the same Python/IDLE you used above.")
    raise

# GitHub raw URL (may fail in some setups)
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random-Forest-Regression/master/studentscores.csv"

# Embedded fallback dataset (used if URL fails)
fallback = {
    "Hours": [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9],
    "Scores": [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62]
}

# Load data (try URL first, then fallback)
try:
    print("Trying to load dataset from GitHub URL...")
    data = pd.read_csv(url)
    print("Loaded dataset from URL.")
except Exception as e:
    print("Could not load from URL (will use fallback embedded data).")
    print("Error message:", e)
    data = pd.DataFrame(fallback)

print("\nFirst rows of dataset:\n", data.head(), "\n")

# Prepare features and target
X = data[["Hours"]]
y = data["Scores"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model evaluation:")
print("  Mean Squared Error:", mse)
print("  R² Score:", r2)

# Predict for 7 hours
pred_7 = model.predict(np.array([[7]]))[0]
print(f"\nPredicted score for 7 hours study: {pred_7:.2f}")

# Plot (ensure IDLE shows the window)
plt.scatter(X, y, label="Actual data")
plt.plot(X, model.predict(X), label="Regression line")
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.title("Study Hours vs Scores")
plt.legend()
plt.show(block=True)
