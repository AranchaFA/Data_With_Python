import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Color map, to represent different clusters in different colors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV # holdout / k-fold cross validations
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix

"""
In this samples we will program two models to perform a MULTI-CLASS CLASSIFICATION,
both will be K-NEAREST NEIGHBORS (supervised methods), difference will be in CROSS VALIDATION method:
    1) HOLDOUT cross validation: Splitting train-test datasets, fit with the first and test with the second one
    2) K-FOLD cross validation: Split dataset into N sub-sets, fit with (N-1) and test with the last, repeat N times rotationally 
WE WILL WORK WITH SAMPLE ScikitLearn DATASET, it provides information about iris plants:
width and length of petals and sepals (4 measurements) and type to which each one belongs
"""

#region LOAD DATASET
print('\n ------- LOAD DATA -------')
# Data is stored in a CSV file into 'files' folder, we dump it in a DataFrame
iris_dataset = pd.read_csv('../files/iris_data.csv')
# Watch dataset features
print('\nSHAPE and COLUMNS')
print(iris_dataset.shape)
print(iris_dataset.columns)
print('\n HEAD data')
print(iris_dataset.head())
#endregion

#region ANALYZE DATA
print('\n ------- ANALYZE DATA -------')
print('\n SUMMARY STATISTICS')
# Not missing values, all fields are numeric, they have different ranges but similar magnitude
# so we can skip STANDARIZATION (to compare features with different ranges and magnitudes in classifications,
# attributes must be standarized: have mean of 0 and standard deviation of 1)
print(iris_dataset.describe())

# Check data related to each specie (categorical variable)
print('\n SPECIES INFO (categorical variable)')
print('\nSpecies unique values')
print(iris_dataset['species'].unique())
print('\n Value count by specie')
# All species have equal amount of data, so model will be 'balanced' trained
print(iris_dataset.groupby('species').size()) # or iris_dataset['species'].value_counts()

# Visualize data
print('\n VISUALIZE DATA')
# To see scatter plots of all features with each others, use pandas.plotting scatter_matrix
scatter_matrix(iris_dataset, diagonal='kde') # by default diagonal=histogram
# In histogram, petal measurements shows patterns more differenciated as sepal measurements,
# which have more balanced data distribution. So we can notice different clusters based on petal measurements.
iris_dataset.hist() # Plotting quick histogram
plt.show()


# To see data representation 'labeled' by species, we use scatter plot with different color for each specie datapoints
def scatter_groupby_data(x_feature: str, y_feature: str, groupby_feature: str, dataset):
    colors = cm.rainbow(np.linspace(0, 1, dataset[groupby_feature].nunique()))  # 3 colors from rainbow :)
    for feature_group, index in zip(dataset[groupby_feature].unique(), range(len(colors))):
        query_str = groupby_feature + ' == "' + feature_group + '"'
        subdataset = dataset.query(query_str)
        plt.scatter(subdataset[x_feature], subdataset[y_feature],
                    color=colors[index], label=feature_group, marker='.')
        # color=single_color, c=array_colors -> array_colors.size = subdataset.size
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend()
    plt.show()


# We can see that correlation between petal length/width allow us cluster iris types clearly than sepal length/width
scatter_groupby_data('petal_length', 'petal_width', 'species', iris_dataset)
scatter_groupby_data('sepal_length', 'sepal_width', 'species', iris_dataset)
#endregion

#region PREPARE DATA
print('\n ------- PREPARE DATA -------')
X = iris_dataset[['petal_length', 'petal_width']]
Y = iris_dataset['species']
# We will use, 70% data to traind and 30% data to test, data splitted randomly but STRATIFIED by 'species' feature (Y)
# stratify assures us 30% of each specie data will be in text set (and 70% in train test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1, stratify=Y)
print('\nX_train shape')
print(X_train.shape)
print('\nY_train value_counts()')
print(Y_train.value_counts())
print('\nY_test value_counts()')
print(Y_test.value_counts())
#endregion

#region 1)  HOLDOUT VALIDATION
print('\n ******* HOLDOUT VALIDATION MODEL *******')
#region TRAIN MODEL
print('\n ------- TRAIN MODEL (k-nearest neighbors) -------')
print('......  :)')
# Instantiate (k=5 randomly selected)
knn_holdout = KNeighborsClassifier(n_neighbors=5)
# Train
knn_holdout.fit(X_train, Y_train)
#endregion

#region TEST MODEL
print('\n ------- TESTING MODEL -------')
print('\nVALUE COUNTS')
Y_predicted = knn_holdout.predict(X_test)
print(pd.Series(Y_predicted).value_counts()) # DataFrame or ndarrays haven't value_counts() method!
# To see probabilities of each element to be predicted as every 'species' value
print('\nPREDICTING PROBABILITIES')
Y_predicted_probabilities = knn_holdout.predict_proba(X_test)
print(Y_predicted_probabilities[10:15]) # Element 11 has 20% to be predicted as 'versicolor' and 80% 'virginica'
print(Y_predicted[11])
print('\nACCURACY')
accuracy = accuracy_score(Y_test, Y_predicted)
# or knn_holdout.score(X_test, Y_test) or (Y_predicted == Y_test).sum() / Y_test.size
print(accuracy.round(2)) # 98% correctly predicted
print('\nCONFUSION MATRIX')
confusion_matrix = confusion_matrix(Y_test, Y_predicted, labels=iris_dataset['species'].unique())
print(confusion_matrix)
plot_confusion_matrix(knn_holdout, X_test, Y_test, cmap=plt.cm.Greens)
plt.show()
#endregion

#endregion

#region 2)  K-FOLD CROSS VALIDATION
print('\n ******* K-FOLD CROSS VALIDATION MODEL *******')

#region TRAIN AND TUNE MODEL
# Train initial model with random k=5
print('\n ------- TRAIN INITIAL MODEL -------')
# Instantiate classifier
knn_kfold = KNeighborsClassifier(n_neighbors=5)
print('Accuracies with n_neighbors = 5')
# Train classifier -> cross_val_score() doesn't work as fit() ! To predict later with this model we must fit it
knn_kfold.fit(X, Y)
knn_kfold_scores = cross_val_score(knn_kfold, X, Y, cv=5) # cv=5 -> 5 sub-datasets containing each one 20% of data
print(knn_kfold_scores)

# Analyze most efficient K value
print('\n ------- TUNE K HYPERPARAMETER -------')
# Test n_neighbors values from 2 to 10
# Instantiate classifier WITHOUT N_NEIGHBORS PARAMETER because we are going to search it!!
knn_kfold2 = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(2, 10)} # key must be 'n_neighbors'! Range of values to test (list or dict!)
# Create GridSearch object with our model instance, param_grid to test and amount of cross validation tests (cv)
knn_kfold_gscv = GridSearchCV(knn_kfold2, param_grid, cv=5) # we are testing for cv=5
# Train GridSearch
knn_kfold_gscv.fit(X, Y)

# Show best searched values
print('Best n_neighbors value')
print(knn_kfold_gscv.best_params_)
best_k = knn_kfold_gscv.best_params_['n_neighbors']
print('Accuracy for n_neighbors = ' + str(best_k))
print(knn_kfold_gscv.best_score_)

# Initial / Tuned model comparisson
print('\nFirst model tested')
print(knn_kfold)
print('Best model')
print(knn_kfold_gscv.best_estimator_)

print('\n ------- CREATE FINAL MODEL (tuned) -------')
# Instantiate and train final model
knn_kfold_tuned = KNeighborsClassifier(n_neighbors=best_k)
knn_kfold_tuned.fit(X, Y)
#endregion train and tune

#region TEST RESULTS
print('\n ------- ACCURACY COMPARISSON -------')
initial_accuracy = knn_kfold.score(X, Y)
tuned_accuracy = knn_kfold_tuned.score(X, Y)
accuracy_improvement = ((tuned_accuracy-initial_accuracy)*100).round(2)

print('Not tuned model accuracy')
print(initial_accuracy)
print('Tuned model accuracy')
print(tuned_accuracy)
print('Accuracy improvement')
print(str(accuracy_improvement) + '%')


# CROSS_VAL_SCORE ISN'T VALID TO OBTAIN SCORE !! WE MUST USE MODEL.SCORE(X, Y) !!
# Obtained values are the same in both cases ?¿ :( I don't understand because partial accuracies should be slightly different
print('\n\nCROSS_VAL_SCORE ISN\'T VALID TO OBTAIN SCORE !! WE MUST USE MODEL.SCORE(X, Y) !! :(')
print('This values are obtained using cross_val_score (same in both cases ?¿):')
knn_kfold_scores_tuned = cross_val_score(knn_kfold_tuned, X, Y, cv=5)

print('\nNot tuned model accuracy')
print(knn_kfold_scores)
print('Mean accuracy')
print(knn_kfold_scores.mean())

print('\nTuned model accuracy')
print(knn_kfold_scores_tuned)
print('Mean accuracy')
print(knn_kfold_scores_tuned.mean())

#endregion test results

#endregion k-fold cross validation

#region PREDICT
x_new = np.array([[2.50, 1.20], [3.60, 1.20], [2.50, 1.60]]) # Input (X) must be 2D (n, 2), output (Y) will be 1D
print('New data')
print(x_new)
print('Prediction with holdout model + Confusion matrix')
print(knn_holdout.predict(x_new))
print(knn_holdout.predict_proba(x_new))
print('Prediction with k-fold model (tuned) + Confusion matrix')
print(knn_kfold_tuned.predict(x_new))
print(knn_kfold_tuned.predict_proba(x_new))
#endregion