from src.components.data_fetching import DataFetching
from src.components.MA_Sarima import SarimaPredictor
from src.components.preprocessing import Preprocessing
from src.components.model_trainer import ModelTrainig
from src.components.model_evaluation import ModelEvaluation
from src.utils.utils import read_yaml
from src import logger
from pathlib import Path
import json
import jdatetime as jdt


import warnings
warnings.filterwarnings('ignore')

model_name = "lgb"
level4 = '7_5_9_2'  # specific group
date = None     # Either None or int (e.g. 14040301)
sarima_steps = 9    # Number of steps for Sarima prediction
date_15_days_ago = jdt.date.today() - jdt.timedelta(days = 15)
start_forecast = jdt.datetime.strptime(str(date if date != None else date_15_days_ago.strftime('%Y%m%d')), '%Y%m%d')
end_forecast = start_forecast + jdt.timedelta(days = sarima_steps * 7 )

CONFIG_FILE = read_yaml(Path("config/config.yaml"))
PARAMS_FILE = read_yaml(Path("params.yaml"))

with open('features_recipe.json', 'r') as openfile:
    FEATURES_RECIPE = json.load(openfile)

with open('feature_rules.json', 'r') as openfile:
    FEATURE_RULES = json.load(openfile)

STAGE_NAME = "Data Fetching"
try:
    logger.info(f"--- Stage {STAGE_NAME} started ---")

    data_fetching = DataFetching(CONFIG_FILE["data_fetching"]["server_name"],
                                 CONFIG_FILE["data_fetching"]["database_name"],
                                 start_forecast,
                                 end_forecast,
                                 level4
                                 )

    train_data, forecast_data, forecast_date = data_fetching.run()

    logger.info(f"--- Stage {STAGE_NAME} completed ---\n\n")

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Sarima Prediction"
try:
    logger.info(f"--- Stage {STAGE_NAME} started ---")

    sm_predictor = SarimaPredictor()
    forecast_data = sm_predictor(train_data, 
                                 forecast_data,
                                 forecast_date,
                                 steps = sarima_steps
                                 )


    logger.info(f"--- Stage {STAGE_NAME} completed ---\n\n")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Preprocessing"
try:
    logger.info(f"--- Stage {STAGE_NAME} started ---")
    
    preprocessing = Preprocessing(train_data,
                                  forecast_data,
                                  CONFIG_FILE["preprocessing"]["features_to_keep"], 
                                  CONFIG_FILE["preprocessing"]["sin_features"],   
                                  CONFIG_FILE["preprocessing"]["features_to_dummies"],
                                  CONFIG_FILE["preprocessing"]["MA_variations"],
                                  CONFIG_FILE["preprocessing"]["T_variations"],
                                  FEATURE_RULES,
                                  FEATURES_RECIPE
                                  )

    train_data_processed, forecast_data_processed = preprocessing.preprocess_data()
    
    logger.info(f"--- Stage {STAGE_NAME} completed ---\n\n")

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Model Training"
try:
    logger.info(f"--- Stage {STAGE_NAME} started ---")

    trainer = ModelTrainig(train_data_processed,
                           CONFIG_FILE["training"]["validation_size"],  
                           CONFIG_FILE["training"]["models"], 
                           model_name, 
                           PARAMS_FILE
                           )

    best_model, best_params, feature_importances = trainer.run_training()

    logger.info(f"--- Stage {STAGE_NAME} completed ---\n\n")

except Exception as e:
    logger.exception(e)
    raise e




STAGE_NAME = "Model Evaluation"

try:
    logger.info(f"--- Stage {STAGE_NAME} started ---")

    evaluator = ModelEvaluation(best_model,
                                best_params,
                                train_data_processed,
                                forecast_data_processed,
                                feature_importances,
                                CONFIG_FILE["training"]["validation_size"]
                                )
    evaluator.run_evaluation()

    logger.info(f"--- Stage {STAGE_NAME} completed ---\n\n")

except Exception as e:
    logger.exception(e)
    raise e
