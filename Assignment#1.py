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
X_range_df = pd.DataFrame(X_range, columns=['YearsExperience']) 
Y_range_pred = model.predict(X_range_df)

plt.plot(X_range, Y_range_pred, color='green', label='Regression Line')

plt.title('Linear Regression: Years of Experience vs. Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()


# Question: 1.2

# Naive Bayes (from scrath)

# Training data
C1_data = [
    [1, 1, 1],
    [0, 1, 0],
    [1, 1, 0]
]
C2_data = [
    [0, 0, 0],
    [1, 0, 1],
    [1, 0, 0]
]

# New sample to classify
test_x = [0, 1, 0]  

# 1) Calculate Priors: P(C1) and P(C2)
num_C1 = len(C1_data)
num_C2 = len(C2_data)
total_samples = num_C1 + num_C2

P_C1 = num_C1 / total_samples
P_C2 = num_C2 / total_samples
print("P(C1) =", P_C1)
print("P(C2) =", P_C2)

# 2) Estimate Likelihoods P(x|C) = product of P(x_i|C)
#    For each feature i, we need P(feature i = 1 | C) and P(feature i = 0 | C)

def feature_probabilities(class_data):
    """
    parameter - class_data, list of samples beloning to
    particular class, each sample is a list of features (0, 1)

    returns - probabilites, list of probabilites where each element
    represents P(feature i = 1 | class), how often each feature
    appears as 1 in the class
    """
    # Number of samples for this class
    n_samples = len(class_data)
    # Number of features (assuming all samples have same length)
    n_features = len(class_data[0])

    # Initialize counters for how many times each feature is "1"
    counts = [0] * n_features

    # Count how often each feature is 1
    for sample in class_data:
        for i in range(n_features):
            counts[i] += sample[i]

    # Probability of feature i = 1 => counts[i] / n_samples
    # We'll store them in a list
    probabilities = [counts[i] / n_samples for i in range(n_features)]
    return probabilities

# Get probabilities for each feature P(feature i=1|C1), P(feature i=1|C2)
C1_feature_probs = feature_probabilities(C1_data)
C2_feature_probs = feature_probabilities(C2_data)

def naive_likelihood(x, feature_probability):
    """"
    parameters:
    - x (list): The new test sample, a list of binary values (0, 1).
    - feature_probability (list): A list of probabilities where each element represents 
      P(feature i = 1 | class).
      
    returns - likelihood (float): The product of probabilities computed using 
    the Naïve Bayes assumption:
    P(x | C) = P(x_1 | C) * P(x_2 | C) * ... * P(x_n | C)
    """
    likelihood = 1.0
    for i, val in enumerate(x):
        p1 = feature_probability[i]
        if val == 1:
            likelihood *= p1
        else:
            likelihood *= (1 - p1)  

    return likelihood

P_x_given_C1 = naive_likelihood(test_x, C1_feature_probs)
P_x_given_C2 = naive_likelihood(test_x, C2_feature_probs)
print("\nP(x|C1) =", P_x_given_C1)
print("P(x|C2) =", P_x_given_C2)

# 3) Compute posteriors (we can compare P_x_given_C * P_C)
posterior_C1 = P_x_given_C1 * P_C1
posterior_C2 = P_x_given_C2 * P_C2

P_C1_given_x = posterior_C1 / (posterior_C1 + posterior_C2)
P_C2_given_x = posterior_C2 / (posterior_C1 + posterior_C2)
print("\nP(C1|x) =", P_C1_given_x)
print("P(C2|x) =", P_C2_given_x)

# 4) Classification decision
if P_C1_given_x > P_C2_given_x:
    print("\nWe classify x as Class 1 (C1).")
else:
    print("\nWe classify x as Class 2 (C2).")