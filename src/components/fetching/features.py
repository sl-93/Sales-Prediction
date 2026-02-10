import numpy as np
from sklearn.preprocessing import MinMaxScaler


def add_features(train_df, forecast_df):
    for df in [train_df, forecast_df]:
        df["Shock"] = (df["YearMonth"] == 140102).astype(int)
        df["pShock"] = (df["YearMonth"] > 140102).astype(int)
        df["Corona"] = ((df["Date"] > 13981202) & (df["Date"] < 13990121)).astype(int)        

    train_df["Discount"] = train_df["DiscountAmount"] / (train_df["Gross"] + 1e-8)
    train_df["Price"] = train_df["Gross"] / (train_df["WeightQTY"] + 1e-8)

    train_df["Discount"] = MinMaxScaler().fit_transform(train_df[["Discount"]])
    train_df["Price"] = MinMaxScaler().fit_transform(train_df[["Price"]])

    forecast_df["Discount"] = train_df["Discount"][-10:].mean()
    forecast_df["Price"] = train_df["Price"][-10:].mean()
    
    return train_df, forecast_df