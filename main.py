# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing dataset
housing_data = load_boston()

# Convert to a Pandas DataFrame
data_frame = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
data_frame['TARGET_PRICE'] = housing_data.target

# Display the first few rows of the dataframe
print(data_frame.head())

# Check for missing values
print(data_frame.isnull().sum())

# Split the data into features (X) and target (y)
input_features = data_frame.drop('TARGET_PRICE', axis=1)
target_variable = data_frame['TARGET_PRICE']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_features, target_variable, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
predicted_prices = linear_model.predict(X_test_scaled)

# Evaluate the model
mean_squared_err = mean_squared_error(y_test, predicted_prices)
r_squared = r2_score(y_test, predicted_prices)

print(f"Mean Squared Error: {mean_squared_err}")
print(f"R^2 Score: {r_squared}")

# Plot the results
plt.scatter(y_test, predicted_prices)
plt.xlabel("Actual Prices")
plt.ylabel("Estimated Prices")
plt.title("Actual vs Estimated Housing Prices")
plt.show()
