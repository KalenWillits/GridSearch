# %% markdown
# ## Grid Search Hyperparameter optimization
# %% markdown
# This case study is all about using grid searches to identify the optimal parameters for a machine learning algorithm. To complere this case study, you'll use the Pima Indian diabetes dataset from Kaggle and KNN. Follow along with the preprocessing steps of this case study.
# %% markdown
# Load the necessary packages
# %% codecell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# %% markdown
# #### Load the diabetes data
# %% codecell
cd_data = 'data/'
df = pd.read_csv(cd_data+'diabetes.csv')
df.head()
# %% markdown
# **<font color='teal'> Start by reviewing the data info.</font>**
# %% codecell
df.info()
# %% markdown
# **<font color='teal'> Apply the describe function to the data.</font>**
# %% codecell
df.describe()
# %% markdown
# **<font color='teal'> Currently, the missing values in the dataset are represented as zeros. Replace the zero values in the following columns ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] with nan .</font>**
# %% codecell
columns = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for column in columns:
    df[column].replace(0, np.NaN, inplace=True)
# %% markdown
# **<font color='teal'> Plot histograms of each column. </font>**
# %% codecell
cd_figures = 'figures/'

for column in df.columns:
    plt.hist(df[column])
    plt.title('Histogram of {}'.format(column))
    plt.xlabel(column)
    plt.ylabel('Number of Values')
    plt.grid()
    plt.savefig(cd_figures+column+'.png')
    plt.show()
    plt.clf()

# %% markdown
# #### Replace the zeros with mean and median values.
# %% codecell
df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)
df['Insulin'].fillna(df['Insulin'].median(), inplace = True)
df['BMI'].fillna(df['BMI'].median(), inplace = True)
# %% markdown
# **<font color='teal'> Plot histograms of each column after replacing nan. </font>**
# %% codecell
for column in df.columns:
    plt.hist(df[column], color='teal')
    plt.title('Histogram of {} (no_NaNs)'.format(column))
    plt.xlabel(column)
    plt.ylabel('Number of Values')
    plt.grid()
    plt.savefig(cd_figures+column+'(no NaNs).png')
    plt.show()
    plt.clf()
# %% markdown
# #### Plot the correlation matrix heatmap
# %% codecell
plt.figure(figsize=(12,10))
print('Correlation between various features')
p=sns.heatmap(df.corr(), annot=True,cmap ='Blues')
# %% markdown
# **<font color='teal'> Using Sklearn, standarize the magnitude of the features by scaling the values. </font>**
# %% codecell
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

for column in df.columns:
    df[column] = ss.fit_transform(df[[column]])
# %% markdown
# **<font color='teal'> Define the `y` variable as the `Outcome` column.</font>**
# %% codecell
X = df.drop('Outcome', axis=1)
y = df['Outcome'].astype(int)
# %% markdown
# **<font color='teal'> Create a 70/30 train and test split. </font>**
# %% codecell
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1111)
# %% markdown
# #### Using a range of neighbor values of 1-10, apply the KNearestNeighbor classifier to classify the the data.
# %% codecell
from sklearn.neighbors import KNeighborsClassifier

test_scores = []
train_scores = []

for i in range(1,10):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)

    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))
# %% markdown
# **<font color='teal'> Print the train and test scores for each iteration.</font>**
# %% codecell
print(train_scores)
print(test_scores)
# %% markdown
# **<font color='teal'> Identify the number of neighbors between 1-15 that resulted in the max score in the training dataset. </font>**
# %% codecell
parameters = {'n_neighbors':range(1,16)}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(knn, parameters)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
# %% markdown
# **<font color='teal'> Identify the number of neighbors between 1-15 that resulted in the max score in the testing dataset. </font>**
# %% codecell
grid_search.fit(X_test, y_test)
print(grid_search.best_params_)
# %% markdown
# Plot the train and test model performance by number of neighbors.
# %% codecell
plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,10),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,10),test_scores,marker='o',label='Test Score')
plt.savefig('figures/train_vs_test_scores')
# %% markdown
# **<font color='teal'> Fit and score the best number of neighbors based on the plot. </font>**
# %% codecell

# %% codecell
from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
pl = confusion_matrix(y_test,y_pred)
# %% markdown
# **<font color='teal'> Plot the confusion matrix for the model fit above. </font>**
# %% codecell
plt.plot(pl)
plt.title('confusion_matrix')
plt.savefig('figures/confusion_matrix.png')
# %% markdown
# **<font color='teal'> Print the classification report </font>**
# %% codecell
from sklearn.metrics import classification_report

cr = classification_report(y_test, y_pred)
print(cr)
# %% markdown
# #### In the case of the K nearest neighbors algorithm, the K parameter is one of the most important parameters affecting the model performance.  The model performance isn't horrible, but what if we didn't consider a wide enough range of values in our neighbors for the KNN? An alternative to fitting a loop of models is to use a grid search to identify the proper number. It is common practice to use a grid search method for all adjustable parameters in any type of machine learning algorithm. First, you define the grid — aka the range of values — to test in the parameter being optimized, and then compare the model outcome performance based on the different values in the grid.
# %% markdown
# #### Run the code in the next cell to see how to implement the grid search method for identifying the best parameter value for the n_neighbors parameter. Notice the param_grid is the range value to test and we apply cross validation with five folds to score each possible value of n_neighbors.
# %% codecell
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,y)
# %% markdown
# #### Print the best score and best parameter for n_neighbors.
# %% codecell
print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))
# %% markdown
# Here you can see that the ideal number of n_neighbors for this model is 14 based on the grid search performed.
# > Actually it's 25. However if we change the limit of parameters to 15, 14 would be the pest paramter to use.
# %% markdown
# **<font color='teal'> Now, following the KNN example, apply this grid search method to find the optimal number of estimators in a Randon Forest model.
# </font>**
# %% codecell
from sklearn.ensemble import RandomForestClassifier
param_grid = {'n_estimators': range(100)}
rfc = RandomForestClassifier()
rfc_cv = GridSearchCV(rfc, param_grid, cv=5)
rfc_cv.fit(X,y)
# %% codecell
print("Best Score:" + str(rfc_cv.best_score_))
print("Best Parameters: " + str(rfc_cv.best_params_))
