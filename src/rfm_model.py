import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
def load_data(path):
    df = pd.read_csv(path)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    return df

def create_rfm(df):

    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerId").agg(
        Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount", "sum")
    ).reset_index()

    return rfm

def scale_rfm(rfm):

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    return scaled

def cluster_customers(rfm):

    scaled = scale_rfm(rfm)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(scaled)

    return rfm

def analyze_clusters(rfm):

    summary = rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]].mean()

    print(summary)

    return summary

def create_target(rfm, high_risk_cluster):

    rfm["is_high_risk"] = np.where(
        rfm["cluster"] == high_risk_cluster,
        1,
        0
    )

    return rfm

def merge_target(df, rfm):

    return df.merge(
        rfm[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )

def main():

    df = load_data("data/raw/data.csv")

    rfm = create_rfm(df)

    rfm = cluster_customers(rfm)

    analyze_clusters(rfm)

    # CHANGE THIS AFTER YOU SEE OUTPUT
    HIGH_RISK_CLUSTER = 1

    rfm = create_target(rfm, HIGH_RISK_CLUSTER)

    final_df = merge_target(df, rfm)

    final_df.to_csv("data/processed/processed_data.csv", index=False)

    print("Task 4 completed successfully")

print("SCRIPT STARTED")


if __name__ == "__main__":
    main()
