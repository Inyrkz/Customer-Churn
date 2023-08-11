# Project Overview
In this project I created an artificial neural network with Keras to predict customers that will churn from a company.

The  `churn_modelling_with_ann.ipynb` notebook contains code to predict customers that will churn from a bank. The `customer-churn` dataset was retrieved from the Kaggle repository.

The  `Telcom_customer_churn.ipynb` notebook contains code to predict customer churn from s telecommunication industry. The `SyriaTel` dataset was used for this part of the project.
Full documentation [here](https://docs.google.com/document/d/1iTzE5IaBaH0Bok__YcEkCfNt8kSNXVfgCu_wZNCW7fY/edit?usp=sharing).


## Customer Churn Prediction for Telecommunication Report

The goal of the project is to use machine learning to predict customer churn from a telecommunication company. That is, predicting whether a customer will change telecommunications provider. The SyriaTel dataset was used for this project. The dataset consists of 3333 records of customers and 32 columns.

[!Pie-Chart](/images/pie_chart.PNG)
Figure 1: Pie chart showing the percentage of customers that churned vs customers that didn’t churn in the dataset.

The dataset consists of 2850 records of customers that did not churn. It contains 483 records of customers that churn. The dataset is imbalanced because it has more data of customers that did not churn than customers that churned.

### Exploratory Data Analysis
Exploratory data analysis is carried out using the Pandas-Profiling library. From the overview above, there are no missing values. The visualization above shows that none of the variables have missing values. They all have 3333 samples. The images below show the analysis of each of the columns in the dataset.

The ‘phone number’ column has a cardinality of 100%. This means all the values in that column are unique. It doesn’t have any information that will help us determine if a customer will churn or not. We won’t use this variable as a feature.


[!Correlation Matrix](/images/correlation_chart.png)
The Seaborn library is used to show the Pearson Correlation of features in the dataset. Pearson Correlation is given as:

Where, 
n = total number of observations,
x = first variable
y = second variable
r = Pearson correlation value.


Figure 6: Pearson Correlation of Features in the Dataset using Seaborn
From the Seaborn Heatmap, we see that the total day minutes column is highly positively correlated with the total day charges. Total evening minutes is highly positively correlated with total evening charges. Total night minutes is highly positively correlated with total night charges. Total intl minutes is highly positively correlated with total intl charges. 

Data Cleaning
This first step of data pre-processing (cleaning) is to remove irrelevant features. The features I used are 'international plan', 'voice mail plan', 'number vmail messages', 'total day minutes', 'total day calls', 'total day charge', 'total eve minutes', 'total eve calls', 'total eve charge', 'total night minutes', 'total night calls', 'total night charge', 'total intl minutes', 'total intl calls', 'total intl charge', 'customer service calls'.

The next step is to separate the features and the target variable. The “churn” column is our target variable. It is what we are trying to predict. 

Figure 7: Extracted Features



We have to convert the boolean values in the international plan and voice mail plan columns to numbers. We apply ordinal encoding here. These values must be converted to numbers before we can start training the machine learning algorithms on the data.


Figure 8: Features after encoding

We perform label encoding to convert the categorical variables in the target column to numbers. The number 0 represents the false class [customers that did not churn]. While the number 1 represents the true class [customers that churn]. 

The dataset is split into the training and validation set. 75% of the data was used for training and 25% was used for validation. There are 2499 records in the training set and 834 records in the validation set. The last step in the data cleaning process is feature scaling. The method of feature scaling used is standardization. Standardization is applied to the training set and test set to keep all the features on the same scale. It also helps to speed up the training process. The formula for standardization is shown below. 
x =[x − mean(x)]/standard deviation(x) 
Training Model
I trained six machine learning algorithms on the training data. The algorithms include Logistic Regression, K Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Tree, Random Forest and XGBoost
Training with the Logistic Regression Algorithm

Figure 9: Training with the Logistic Regression Algorithm
I import the LogisticRegression class from the linear_model module of the Sci-kit Learn library. The maximum number of training iterations, max_iter was set to 300. The penalty was set to ‘none’.
Training with the K-Nearest Neighbor (KNN) Algorithm

Figure 10: Training with the KNN Algorithm
I import the KNeighborsClassifier class from the neighbors module of the Sci-kit Learn library. The parameter n_neighbors, which represents the number of neighbors, was set to 5. Five is also the default value for this parameter. The metric parameter represents the metric to be used to calculate the distance from the k neighbors. The metric was set to ‘minkowski’, which is the default value. The p parameter is the power parameter. It is used only when the Minkowski metric is used. Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. We are using the Euclidean distance to calculate the distance between two points by setting p to a value of 2. The formula for Euclidean distance is sqrt(sum((x - y)^2)). The formula for Manhattan distance is sum(|x - y|). 

Training with the Support Vector Machine (SVM) Algorithm

Figure 11: Training with the SVM Algorithm
The SVC (Support Vector Classifier) class is imported from the svm module of the Sci-kit learn library. The sigmoid SVM kernel is used. The other SVM kernels available in the Sci-kit Learn library are ‘poly’, ‘rbf’, ‘linear’ and ‘precomputed.’
Training with the Decision Trees Algorithm
 
Figure 12: Training with the Decision Tree Algorithm
I import the DecisionTreeClassifier class from the tree module of the Sci-kit Learn library. The criterion parameter is a function for determining the quality of a split. The criterion is “entropy.” The min_samples_split represents the minimum amount of samples needed to separate an internal node in the decision tree. This parameter helps us avoid overfitting. The min_samples_split  is 25. It means once we have 25 samples remaining, they should not be split again into various classes.
Training with the Random Forest Algorithm

Figure 13: Training with the Random Forest Algorithm
I import the RandomForestClassifier class from the ensemble module of the Sci-kit Learn library. The n_estimators parameter represents the number of decision trees we use to build our random forest classifier. The n_estimators parameter is 100. It means we are using 100 decision trees in our random forest classifier. The min_samples_split is 25, and the criterion parameter is ‘entropy.’ 
Training with the XGBoost Algorithm

Figure 14: Training with the XGBoost Algorithm
I import the XGBClassifier class from the xgboost library.
RESULTS
For the training and validation results, we focus on the F1 score metric. This is because the dataset is imbalanced. Accuracy will not be efficient on a dataset with imbalanced classes. We will use the accuracy, precision, recall, and f1-score metrics to evaluate the test set.

VALIDATION RESULTS
20% of the data was kept aside for use as a validation set. The result of each model on the validation set is shown below. The confusion matrix is also shown below.

Algorithms
Accuracy (%)
Precision (%)
Recall (%)
F1-Score (%)
Logistic Regression
86.33
83.55
86.33
83.43
KNN
88.97
87.75
88.97
87.65
SVM
80.22
75.43
80.22
77.55
Decision Tree
91.13
90.76
91.13
90.90
Random Forest
96.28
96.20
96.28
96.19
XGBoost
96.28
96.21
96.28
96.16

Table 4.0: Test Results



The top two models are the XGBoost model and the Random Forest model.          

## Confusion Matrix Plots
The confusion matrix shows the number of true positives, false positives, true negatives, and false negatives.

Our class of interest is the ‘Churned’ class. The logistic regression model correctly classified 25 customers that churned. It misclassified 18 customers as churned, whereas the customers did not leave. It correctly classified 695 customers as retained. It misclassified 96 customers as retained, whereas they actually churned.

The KNN model correctly classified 49 customers that churned. It misclassified 20 customers as churned, whereas the customers did not leave. It correctly classified 693 customers as retained. It misclassified 72 customers as retained, whereas they actually churned.

The SVM model correctly classified 10 customers that churned. It misclassified 54 customers as churned, whereas the customers did not leave. It correctly classified 659 customers as retained. It misclassified 111 customers as retained, whereas they actually churned.

The Decision Tree model correctly classified 77 customers that churned. It misclassified 30 customers as churned, whereas the customers did not leave. It correctly classified 683 customers as retained. It misclassified 44 customers as retained, whereas they actually churned.
The Random Forest model correctly classified 99 customers that churned. It misclassified 9 customers as churned, whereas the customers did not leave. It correctly classified 704 customers as retained. It misclassified 22 customers as retained, whereas they actually churned.

The XGBoost model gave us the lowest number of false positives, and highest number of true negatives. While the random forest model gave us the lowest number of false negatives, and highest number of true positives. 

