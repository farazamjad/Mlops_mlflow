#import libraries
import setuptools

import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

import mlflow

import mlflow.sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#setting mlflow experiment name
experiment_name = "mlops_demo"
mlflow.set_experiment(experiment_name)

#starts mlflow logging
with mlflow.start_run():

    df=pd.read_csv("diabetes.csv")
    #log data to mlflow
    mlflow.log_artifact('diabetes.csv')

#seprate target column from dataset
    X=df.drop("Outcome",axis=1)
     #X.head()

#split train and test data
    y=df['Outcome'].values
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    
#apply logistic regression model   
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)


#performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    mlflow.sklearn.log_model(lr_model, "model")

#log performance metrics to mlflow
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision) 
    mlflow.log_metric("Recall", recall) 
    mlflow.log_metric("f1_score", f1) 
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


