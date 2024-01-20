###Heart Disease ML Classification Project

#Author: Ray Lopez
#Last Updated: 1/20/24

#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, confusion_matrix

#Defining function to plot our ROC curve
def plot_roc_curve(fpr, tpr, name):
    fig = figure(figsize=(4,3))
    fig.set_dpi(300)
    plt.plot(fpr, tpr, color="blue", label="ROC")
    plt.plot([0,1], [0,1], color="red", linestyle="--", label="Guessing")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(name + " ROC Curve")
    plt.legend()
    plt.show()

#Defining function to plot our Confusion Matrix
def plot_conf_matrix(conf_mat, name):
    fig, ax = plt.subplots()
    fig.set_size_inches(3,3)
    fig.set_dpi(300)
    ax = sns.heatmap(conf_mat, annot=True, fmt=str())
    plt.title(name + " Confusion Matrix")

#Importing Data Set
df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

"""
display(df)
"""

#Preparing Data For Analysis
y = df['target']
x = df.drop('target', axis=1)

"""
#Verifying x and y splitting
display(y)
display(x)
"""

#Data Splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25)

"""
#Verifying data training and testing split
display(x_train)
display(x_test)
display(y_train)
display(y_test)
"""

#Linear SVC Classifier
lsvc = LinearSVC(max_iter=10000)

#Fitting and scoring the model with our data
lsvc.fit(x_train, y_train)
display(lsvc.score(x_test, y_test))
display("low score indicating poor fit to model")

#Not going to evaluate due to low score

##############################################################################

#Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100)

#Fitting and scoring the model with our data
rfc.fit(x_train, y_train)
display(rfc.score(x_test, y_test))
display("high score indicating good fit to model")

#RFC Evaluation Using ROC Curve
y_rfc_probs = rfc.predict_proba(x_test)
y_rfc_probs_positive = y_rfc_probs[:, 1]

rfc_fpr, rfc_tpr, rfc_thresholds = roc_curve(y_test, y_rfc_probs_positive)

#Plotting ROC Curve
plot_roc_curve(rfc_fpr, rfc_tpr, "Random Forest Classifier")

#RFC Evaluation Using Confusion Matrix
y_rfc_preds = rfc.predict(x_test)
confusion_matrix(y_test, y_rfc_preds)
conf_mat = pd.crosstab(y_test, y_rfc_preds, rownames=["Actual Labels"], colnames=["Predicted Labels"])
display(conf_mat)

#Displaying Our Confusion Matrix
plot_conf_matrix(conf_mat, "Random Forest Classifier")
