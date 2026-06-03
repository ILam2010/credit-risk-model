import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

def extract_time_features(df):

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["TransactionHour"] = (
        df["TransactionStartTime"].dt.hour
    )

    df["TransactionDay"] = (
        df["TransactionStartTime"].dt.day
    )

    df["TransactionMonth"] = (
        df["TransactionStartTime"].dt.month
    )

    df["TransactionYear"] = (
        df["TransactionStartTime"].dt.year
    )

    return df

def create_customer_features(df):

    agg = (
        df.groupby("CustomerId")
        .agg(
            TotalTransactionAmount=("Amount", "sum"),
            AvgTransactionAmount=("Amount", "mean"),
            TransactionCount=("Amount", "count"),
            StdTransactionAmount=("Amount", "std")
        )
        .reset_index()
    )

    df = df.merge(
        agg,
        on="CustomerId",
        how="left"
    )

    return df

categorical_features = [
    "ProviderId",
    "ProductCategory",
    "ChannelId",
    "CurrencyCode"
]

numerical_features = [
    "Amount",
    "Value",
    "CountryCode",
    "PricingStrategy",

    "TransactionHour",
    "TransactionDay",
    "TransactionMonth",
    "TransactionYear",

    "TotalTransactionAmount",
    "AvgTransactionAmount",
    "TransactionCount",
    "StdTransactionAmount"
]

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numerical_features
        ),

        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)

def process_data(filepath):

    df = pd.read_csv(filepath)

    df = extract_time_features(df)

    df = create_customer_features(df)

    X = preprocessor.fit_transform(df)

    return X

if __name__ == "__main__":

    X = process_data(
        "data/raw/data.csv"
    )

    print(X.shape)