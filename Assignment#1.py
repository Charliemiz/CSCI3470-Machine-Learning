from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.model_selection
import pandas as pd

# Question: 1.1

# Read / prep dataset
data = pd.read_excel('dataset.xlsx')

df = pd.DataFrame(data)
print(df)

# Split into test / train
X = df[['YearsExperience']]
y = df['Salary']
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
w = model.coef_[0]
b = model.intercept_

print("\n--- Model Parameters ---")
print("Coefficient (w):", w)
print("Intercept (b):", b)

# Now predict on the test set
y_pred = model.predict(X_test)
comparison_df = pd.DataFrame({
    'YearsExperience': X_test['YearsExperience'],
    'Actual Salary': y_test,
    'Predicted Salary': y_pred
})

print("\n--- Comparison of Actual vs Predicted ---")
print(comparison_df)

# Plotting the results
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Testing Data')

X_range = np.linspace(df['YearsExperience'].min(), df['YearsExperience'].max(), 100)
Y_range_pred = model.predict(X_range.reshape(-1, 1))
plt.plot(X_range, Y_range_pred, color='green', label='Regression Line')

plt.title('Linear Regression: Years of Experience vs. Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()


# Question: 1.2