import numpy as np
import pandas as pd

campaign = np.array([14030328,14030329,14030330,14030331,
                     14030401,14030402,14030403,14030404,
                     14030302,14030301,14030303,
                     14030501,14030502,14030503,14030504,
                     14030601,14030602,14030603,
                     14030629,14030630,14030631,
                     14030701,14030702,14030703,
                     14030801,14030802,14030803])


def add_paycheck_feature(df):
    """
    :param df (DataFrame)
    Add a binary indicator for paycheck periods.
    This feature captures demand patterns around salary payment dates,
    Logic:
  - paycheck = 1 for days close to the end or beginning of the month
    """
    df["paycheck"] = 0
    idx = df[(df.Day >= 30) | (df.Day < 2)].index
    df.loc[idx, "paycheck"] = 1
    return df

def add_ma_features(df, weight_col, ma_variations):
    for feature in ma_variations:
        for key, value in feature.items():
            df[key] = 1
            df.loc[1: df.shape[0], key] = [
                np.mean(weight_col[i - value:]) for i in range(1, df.shape[0])
            ]
    return df


def add_t_lag_features(df, weight_col, t_variations):
    for feature in t_variations:
        for key, value in feature.items():
            df[key] = 0
            df.loc[value: df.shape[0], key] = [
                weight_col[i - value] for i in range(value, df.shape[0])
            ]
    return df


def apply_sine_features(df, 
                        sin_features
                        ):
    """
    Apply sine transformation to cyclical features.

    :param df: Description
    :param sin_features: Description
    """

    for feature in sin_features:
        for col, period in feature.items():
            df[f"{col}Sin"] = np.sin(2 * np.pi * (df[col] / period))
            df[f"{col}Cos"] = np.cos(2 * np.pi * (df[col] / period))
    return df


def apply_dummy_features(df, features_to_dummies):
    for feature in features_to_dummies:
        for key, _ in feature.items():
            if key in df.columns:
                df = df.join(pd.get_dummies(df[key], prefix=key).astype(int))
                df = df.drop(columns=[key])
    return df



def remove_sin_features(df, 
                        sin_features
                        ):
    """
    Remove features made sin and cosine features.
    :param df: Description
    """

    for feature in sin_features:
        for col, _ in feature.items():
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
    return df

def add_ramadan(df):
    df['Ramadan'] = 0
    ind = df[df.Occastion_ID == 1].index
    df.loc[ind,'Ramadan'] = 1

    return df

def add_school(df):
    df["school"] = 1
    df.loc[(df['Month'].isin([6, 4, 5, 3])) & (df['Year'].astype(int) > 1400), 'school'] = 0
    df.loc[(df['Month'] == 1) & (df['Year'].astype(int) > 1400) & (df['Day'].astype(int) < 16), 'school'] = 0
    df.loc[(df['Month'] == 12) & (df['Year'].astype(int) > 1400) & (df['Day'].astype(int) > 23), 'school'] = 0
    return df 


def add_end_of_year(df):
    df['end_of_year'] = 0
    df.loc[(df.Day>=28) & (df.Month==12),'end_of_year'] = 1

    return df


def add_start_of_year(df):
    df['start_of_year']=0
    df.loc[(df.Day < 3) & (df.Month==1) , 'start_of_year'] = 1
    return df  


def add_weekend(df):
    df['weekend'] = 0
    df.loc[df.WeekDays == 6 ,'weekend'] = 1

    df['weekend_c'] = 1
    df.loc[df.WeekDays == 6 ,'weekend_c'] = 0

    return df
def add_isholiday(df):

    df["IsHolliday"] = df["IsHolliday"] * (1 - df["weekend"])
    return df


def update_ma(df,
              level4_id,
              feature_rules):
    ma_features = ["MA-60","MA-30","MA-15","MA-7"]
    if level4_id in feature_rules:
        for i in range(len(feature_rules[level4_id]["ma"])):
            if feature_rules[level4_id]["ma"][i] == 0:
                df.drop(columns = ma_features[i], inplace=True, errors="ignore")

    return df
    
def update_lag(df,
              level4_id,
              feature_rules):
    lag_features = ["T-7","T-6","T-3","T-1"] # T-5 is not included
    if level4_id in feature_rules:
        for i in range(len(feature_rules[level4_id]["auto_corr"])):
            if feature_rules[level4_id]["auto_corr"][i] == 0:
                df.drop(columns = lag_features[i], inplace=True, errors="ignore")

    return df


def add_new_custom_feature(df,
                           feature_rules,
                           level4_id,
                           features_recipe):
    if level4_id in feature_rules:
        to_add = feature_rules[level4_id]["to_add"] 
        to_remove = feature_rules[level4_id]["to_remove"]
        for feature_name in to_add:

            if feature_name == "campaign": 
                df['campaign']= 0
                df.loc[df.Date.isin(campaign),'campaign'] = 1
            
            if feature_name == "campaign_coef":
                df['campaign_coef']= 1
                df.loc[df.Date.isin(campaign),'campaign_coef'] = 1
            
            if feature_name == "summerdays":
                df[feature_name] = 0
                df.loc[df.Month.isin([3,4,5,6]),feature_name] = 1
            
            if feature_name == "IsBeforeHolliday":
                df['IsBeforeHolliday'] = df.IsHolliday.shift(-1,axis = 0)
                df.loc[pd.isna(df['IsBeforeHolliday'])== True, 'IsBeforeHolliday'] = 0
                df.loc[(df['IsBeforeHolliday']==1)&(df['IsHolliday']== 1),'IsBeforeHolliday'] = 0

            if feature_name not in df.columns and (features_recipe[feature_name]['a'] in df.columns) and (features_recipe[feature_name]['b'] in df.columns):
                df[feature_name] = df[features_recipe[feature_name]['a']] * df[features_recipe[feature_name]['b']]
            
        for feature_name in to_remove:
            if feature_name in df.columns:
                df.drop(columns=feature_name, inplace=True, errors="ignore")
               
    return df


def add_week_day_indicator(df,
                           level4_id,
                           feature_rules):

    if level4_id in feature_rules and len(feature_rules[level4_id]["week_day_indicator"]) == 7:
        for i in range(7):
            if feature_rules[level4_id]["week_day_indicator"][i]==1:
                c = i + 1
                feature_name = "WeekDays_"+ str(c)
                df[feature_name] = 0
                df.loc[df.WeekDays == c , feature_name] = 1
    return df