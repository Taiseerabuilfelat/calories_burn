import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load the new dataset
file_path = "calories_burn.csv"
df = pd.read_csv(file_path)

# Drop User_ID as it's not needed for prediction
df = df.drop(columns=["User_ID"])

# Convert categorical variable 'Gender' to numerical (0 for Female, 1 for Male)
df["Gender"] = df["Gender"].map({"female": 0, "male": 1})

# Check for missing values
df.dropna(inplace=True)

# Split data into features (X) and target (y)
X = df.drop(columns=["Calories"])
y = df["Calories"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Save the trained model to a pickle file
model_filename = "calories_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved as {model_filename}")

# Plot actual vs predicted values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Actual vs Predicted Calories Burned")
plt.show()
