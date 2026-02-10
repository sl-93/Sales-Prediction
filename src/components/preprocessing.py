import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from src import logger

from src.components.Preprocess.features import (
                                                add_paycheck_feature,
                                                add_ma_features,
                                                add_t_lag_features,
                                                apply_sine_features,
                                                apply_dummy_features,
                                                add_start_of_year, 
                                                add_ramadan, 
                                                add_end_of_year, 
                                                add_new_custom_feature, 
                                                add_week_day_indicator, 
                                                add_weekend,
                                                add_isholiday, 
                                                update_ma,
                                                update_lag, 
                                                add_school,
                                                remove_sin_features
    
)
# from src.components.preprocess.rules import apply_feature_rules
    


class Preprocessing:
    def __init__(
        self,
        train_data,
        forecast_data,
        features_to_keep,
        sin_features,
        features_to_dummies,
        MA_variations,
        T_variations,
        feature_rules, 
        feature_recipe
        ):
        
        self.data = train_data
        self.forecast_data = forecast_data
        self.features_to_keep = features_to_keep
        self.sin_features = sin_features
        self.features_to_dummies = features_to_dummies
        self.MA_variations = MA_variations
        self.T_variations = T_variations
        self.feature_rules = feature_rules
        self.feature_recipe=feature_recipe

    def preprocess_data(self):
        """
        train_data : dict[str, pd.DataFrame]
           Preprocessed training data per Level4_ID.
        forecast_data : dict[str, pd.DataFrame]
           Preprocessed forecast data per Level4_ID.
       feature_names : list[str]
            List of final feature column names
        it returns 
        train_data_processed(Dictionary)
        forecast_data_processed(Dictionary)
        features (list)
        
        """
        
        # feature_ids = list(self.feature_rules.keys())

        try:
            for key in self.data.keys():
                data = self.data[key].reset_index(drop=True)
                forecast = self.forecast_data[key].reset_index(drop=True)

                train_actual = data["WeightQTY_Actual"]
                has_actual = "WeightQTY_Actual" in forecast.columns
                has_sarima = "SarimaOutput" in forecast.columns

                if has_actual:
                    actual_weight = forecast["WeightQTY_Actual"]
                if has_sarima:
                    sarima_output = forecast["SarimaOutput"]
           
                cols_to_keep = [list(f.keys())[0] for f in self.features_to_keep]
                data = data[cols_to_keep]
                forecast = forecast[cols_to_keep]

                l_data = len(data)

                data = pd.concat([data, forecast]).reset_index(drop=True)
                WeightQTY = data["WeightQTY"]

                data = add_ma_features(data, WeightQTY, self.MA_variations)
                data = add_t_lag_features(data, WeightQTY, self.T_variations)
                data["Ratio"] = data["MA-7"] / data["MA-60"]
                data = add_paycheck_feature(data )
                data = add_ramadan(data)
                data = add_school(data)
                data = add_weekend(data)
                data = add_isholiday(data)
                data = add_start_of_year(data)
                data = add_end_of_year(data)
                data = add_week_day_indicator(data , key , self.feature_rules)
                data = apply_sine_features(data, self.sin_features)
                data = add_new_custom_feature(data , self.feature_rules , key , self.feature_recipe)
                data = update_ma (data , key, self.feature_rules)
                data = update_lag (data , key , self.feature_rules)
                data = apply_dummy_features(data, self.features_to_dummies)
                data = remove_sin_features(data, self.sin_features)

                # ---------------- train
                train_df = data[:l_data].reset_index(drop=True)
                train_df["target"] = self.data[key]["WeightQTY"].reset_index(drop=True)
                train_df["Date"] = self.data[key]["Date"].reset_index(drop=True)
                train_df["WeightQTY_Actual"] = train_actual
                train_df = train_df.drop("WeightQTY", axis=1)

                # ---------------- forecast
                forecast_df = data[l_data:].reset_index(drop=True)

                if "Price" in train_df.columns:
                    forecast_df["Price"] = train_df["Price"][-7:].mean()
                if "Discount" in train_df.columns:
                    forecast_df["Discount"] = train_df["Discount"][-7:].mean()

                forecast_df["Date"] = self.forecast_data[key]["Date"].reset_index(drop=True)

                if has_actual:
                    forecast_df["WeightQTY_Actual"] = actual_weight
                if has_sarima:
                    forecast_df["SarimaOutput"] = sarima_output

                forecast_df = forecast_df.drop("WeightQTY", axis=1)

                self.data[key] = train_df
                self.forecast_data[key] = forecast_df

            return self.data, self.forecast_data

        except Exception as e:
            logger.exception(e)
            raise
