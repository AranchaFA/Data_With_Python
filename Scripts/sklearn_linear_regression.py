import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# In this sample we will program a model to perform a LINEAR REGRESSION (supervised)
# WE WILL WORK WITH SAMPLE ScikitLearn DATASET,
# Dataset provides information about house prices in Boston and other features of this houses

#region LOAD DATA
# Load dataset
dataset_boston = load_boston() # <class 'sklearn.utils.Bunch'>

# Obtain DF from dataset:
# ds.data -> values (ndarray)
# ds.feature_names -> column names
df_boston = pd.DataFrame(dataset_boston.data, columns=dataset_boston.feature_names)
# shape = (506, 13)
print(df_boston.columns)
print(df_boston.head(2))

# Add column 'PRICE' with data from .target dataset attribute
df_boston['PRICE'] = dataset_boston.target
# shape = (506,)

# Check final DataFrame structure
print('\n-- FINAL DATAFRAME STRUCTURE --')
print(df_boston.columns)
print(df_boston.head(2))
#endregion

#region ANALYZE DATA
# Summary statistics of each feature
print('\n-- SUMMARY STATISTICS --')
print(df_boston.describe().round(2))
# Relations and dependencies of features with each others
print('\n-- CORRELATION MATRIX --')
# RM and LSTAT have greatest correslations with price, we will work with PRICE-RM (number of rooms)
print(df_boston.corr().round(2))
# Plotting to show more clearly
df_boston.plot(kind='scatter', x='RM', y='PRICE', marker='.', color='orange')
plt.show()
#endregion

#region LINEAR REGRESSION (one parameter)
print('\n\n--- ONE PARAMETER LINEAR REGRESSION MODEL ---')
# DATA we will work with
X = df_boston[['RM']] # X must be 2D (we could use several variables)
Y = df_boston['PRICE'] # Y is 1D

# Split data randomly into training (70%) and testing data (30%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# TRAIN model
linear_model = LinearRegression()
linear_model.fit(X_train,Y_train)
# Check  fitting results ( y = coef_ * x + intercept_ )
print('Scope:')
print(linear_model.coef_.round(2)) # <class 'numpy.ndarray'>
print('Intercept:')
print(linear_model.intercept_.round(2)) # Number

# PREDICT with model
new_RM = np.array([6.5]).reshape(-1,1) # Reshape to make it 2D
print('\n- Predicting single value -')
print(linear_model.predict(new_RM).round(2))


# EVALUATE model
# Predict values for test set
print('\n- Predicting test set values -')
Y_test_predicted = linear_model.predict(X_test).round(2)
print(Y_test_predicted)
# Desviation between predicted-existing values
print('\n- Difference predicted-real test set values (RESIDUALS) -')
Y_residuals = Y_test - Y_test_predicted
print(Y_residuals)
print('\n- Mean Squared Error (MSE) -')
mse = mean_squared_error(Y_test, Y_residuals)
print(mse)
print('\n- R2 -')
r2 = linear_model.score(X_test, Y_test) # or r2_score(Y_test, Y_test_predicted)
print(r2)


# Plotting predicted line and real test data
# (line could be less fitted to new data than to train data if model predictions aren't accurate enough)
#plt.scatter(X_train, Y_train, marker='.', color='orange', label='Train data') # If we wont to show train data also
plt.scatter(X_test, Y_test, marker='.', color='b', label='Test data')
plt.plot(X_test, Y_test_predicted, lw=1.5, color='r', label='Prediction')
plt.legend(loc='upper left')
plt.xlabel('Rooms')
plt.ylabel('Price')
plt.show()

# Plotting residuals to check if some pattern exists between x and y_error
# If so, it means model could have bias and must be improved
plt.scatter(X_test, Y_residuals, marker='.', color='r')
plt.hlines(y=0, xmin=X_test.min(), xmax=X_test.max(), colors='black')
plt.xlabel('x value')
plt.ylabel('y bias')
plt.show()
# Results don't seem to follow a pattern, so model has an acceptable fitting


#endregion

#region LINEAR REGRESSION (two parameters)
print('\n\n--- TWO PARAMETERS LINEAR REGRESSION MODEL ---')
# DATA we will work with ( RM&LSTAT - PRICE this time )
X2 = df_boston[['RM', 'LSTAT']] # X must be 2D (we could use several variables)
Y2 = df_boston['PRICE'] # Y is 1D

# Split data randomly into training (70%) and testing data (30%)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.3, random_state=1)

# TRAIN model2
linear_model2 = LinearRegression()
linear_model2.fit(X2_train,Y2_train)
# Check  fitting results ( y = coef_[0] * x1 + coef_[1] * x2 + intercept_ )
print('Partial scopes:')
print(linear_model2.coef_.round(2)) # <class 'numpy.ndarray'>
print('Intercept:')
print(linear_model2.intercept_.round(2)) # Number

# PREDICT with model2
new_RM2 = np.array([6.5, 0.5]).reshape(-1, 2) # Reshape to make it match to X shape
print('\n- Predicting single value -')
print(linear_model2.predict(new_RM2).round(2))


# EVALUATE model
# Predict values for test set
print('\n- Predicting test set values -')
Y2_test_predicted = linear_model2.predict(X2_test).round(2)
print(Y2_test_predicted)
# Desviation between predicted-existing values
print('\n- Difference predicted-real test set values (RESIDUALS) -')
Y2_residuals = Y2_test - Y2_test_predicted
print(Y2_residuals)
print('\n- Mean Squared Error (MSE) -')
mse2 = mean_squared_error(Y2_test, Y2_residuals)
print(mse2)
print('\n- R2 -')
r2_2 = linear_model2.score(X2_test, Y2_test) # or r2_score(Y2_test, Y2_test_predicted)
print(r2_2)
"""
# IT DOESN'T WORK! To test how each parameter individually affects to Y estimated value
print('R2 partial RM')
r2_2RM = linear_model2.score(X2_test['RM'],Y2_test)
print(round(r2_2RM,2))
print('R2 partial LSTAT')
r2_2LSTAT = linear_model2.score(X2_test['LSTAT'],Y2_test)
print(round(r2_2LSTAT,2))
"""

# Graphic representation for multiparameter model would need more than 2D graph (3D for 2 variables)

# Plotting residuals to check if some pattern exists between each variable and y_error
# If so, it means model could have bias and must be improved
plt.scatter(X2_test['LSTAT'], Y2_residuals, marker='.', color='r', label='LSTAT')
plt.scatter(X2_test['RM'], Y2_residuals, marker='.', color='b', label='RM')
plt.hlines(y=0, xmin=X2_test['LSTAT'].min(), xmax=X2_test['LSTAT'].max(), colors='black')
plt.xlabel('x value')
plt.ylabel('y bias')
plt.show()
# Results don't seem to follow a pattern, so model has an acceptable fitting

#endregion

#region COMPARING BOTH UNI/MULTIPARAMETER MODELS
print('\n\n--- COMPARING BOTH UNI/MULTIPARAMETER MODELS ---')
print('MSE uni/multi')
print(round(mse,2))
print(round(mse2,2))
print('MSE(uni)-MSE(multi)')
print(round(mse - mse2,2)) # -21.14 -> 21% improvement with multiparameter model
print('R2 uni/multi')
print(round(r2,2))
print(round(r2_2,2))
print('R2(uni)-R2(multi)')
print(round(r2 - r2_2,2)) # -0.08
#endregion