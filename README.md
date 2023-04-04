# Mlops_mlflow
In this task mlflow was integrated with machine learning workflow to track model runs.
I used logistic regression ML model to predict if a person has diabetes or not using publically available dataset diabetes.csv on kaggle.

To integrate with mlflow firstly we need to install mlflow using following command:

<pre><code>pip install mlflow</pre></code>

The next step is to run the mlflow_app.py file using the command:

  <pre><code>python mlops.py</pre></code>

After model is trained and saved we will use the following command to initialize mlflow:

  <pre><code>mlflow ui</pre></code>

Then navigate to https://localhost:5000

The below screenshots shows our model runs and performance metrics


<img width="1792" alt="Screenshot 2023-04-05 at 2 14 54 AM" src="https://user-images.githubusercontent.com/81928514/229924878-42781ab0-196e-4f68-87ac-55805780c0ea.png">

<img width="1791" alt="Screenshot 2023-04-05 at 2 15 16 AM" src="https://user-images.githubusercontent.com/81928514/229924905-ba025ee2-c011-43a1-9a1f-918953be9957.png">
