import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import mlflow
import mlflow.sklearn
import joblib

df = pd.read_csv("data/processed/processed_data.csv")

X = df.drop("is_high_risk", axis=1)
y = df["is_high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "gradient_boosting": GradientBoostingClassifier()
}

mlflow.set_experiment("credit_risk_model")

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"\n{name}")
        print("Accuracy:", acc)
        print("AUC:", auc)

        # log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # log model
        mlflow.sklearn.log_model(model, name)

        best_model = RandomForestClassifier(n_estimators=100)
best_model.fit(X_train, y_train)

joblib.dump(best_model, "models/best_model.pkl")

