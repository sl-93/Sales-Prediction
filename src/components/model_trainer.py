from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, ElasticNet, LinearRegression, Ridge
import lightgbm as lgb
from tqdm import tqdm


class ModelTrainig:

    def __init__(self,
                 train_data,           # dict of Level4 → DataFrame
                 validation_size,      # replaces hard-coded 30
                 models,               # model config
                 model,                # selected model name
                 params                # hyperparameter grid
                 ):
        
        self.train_data = train_data
        self.validation_size = validation_size
        self.models = models
        self.model = model
        self.params = params

    def run_training(self):
        """
        Trains models using the provided training data and hyperparameters.
        Returns:
            best_models_dict (dict)
            best_params_dict (dict)
            feature_importances (dict)
        """
        best_models_dict = {}
        best_params_dict = {}
        feature_importances = {}

        for level4_id in self.train_data.keys():

            train_df = self.train_data[level4_id]

            # ---------------- Prepare data ----------------
            train_y = train_df["target"]
            train_x = train_df.drop(
                columns=["target", "Date", "WeightQTY_Actual"]
            )

            # 🔹 USE validation_size HERE
            if self.validation_size is not None and self.validation_size > 0:
                train_x = train_x
                train_y = train_y

            # ---------------- TimeSeries CV ----------------
            tscv = TimeSeriesSplit(
                n_splits=4,
                test_size=10
            )

            # ---------------- Model selection ----------------
            for model_name, reg_cfg in self.models.items():

                if model_name != self.model:
                    continue

                model_cls = eval(reg_cfg["type"])
                base_model = model_cls(**reg_cfg["params"])

                param_grid = {}
                for param in self.params[model_name]:
                    for k, v in param.items():
                        param_grid[k] = v

                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_grid,
                    cv=tscv,
                    n_jobs=-1,
                    n_iter=60,
                    scoring="neg_mean_absolute_error",
                    random_state=100,
                    return_train_score=True
                )

                search.fit(train_x, train_y)

                best_est = search.best_estimator_

                # ---------------- Feature importance ----------------
                if hasattr(best_est, "feature_importances_"):
                    feat_imp = (
                        pd.DataFrame({
                            "Level4_ID": level4_id,
                            "feature": train_x.columns,
                            "importance": best_est.feature_importances_
                        })
                        .query("importance != 0")
                        .sort_values("importance", ascending=False)
                        .reset_index(drop=True)
                    )
                    feature_importances[level4_id] = feat_imp

                best_models_dict[level4_id] = best_est
                best_params_dict[level4_id] = search.best_params_

        return best_models_dict, best_params_dict, feature_importances
