import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = pd.read_csv("data.csv")

X = data["review"]
y = data["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
mlflow.set_experiment("Sentiment Analysis Experiment")
with mlflow.start_run(run_name="Logistic Regression Model"):

    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_features", 3000)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "sentiment_model")
print("Model trained and logged successfully")

