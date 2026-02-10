from src.utils.utils import read_yaml
from sqlalchemy import create_engine
import pandas as pd
import urllib
from src import logger
from pathlib import Path
from src.components.data_fetching import DataFetching
from src.components.MA_Sarima import SarimaPredictor
from src.components.preprocessing import Preprocessing
from src.components.model_trainer import ModelTrainig
from src.components.model_evaluation import ModelEvaluation
import pyodbc
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback
import warnings
import json
import jdatetime as jdt
warnings.filterwarnings('ignore')

# === Define absolute paths ===
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# === Load configs & parameters ===
model_name = "lgb"
write_to_db = False
date = None         # Either None or int (e.g. 14040301)
target_month = 140412
sarima_steps = 9        # Number of steps for Sarima prediction
num_workers = cpu_count() - 1
insert_time = jdt.date.today()
date_15_days_ago = jdt.date.today() - jdt.timedelta(days = 15)
start_forecast = jdt.datetime.strptime(str(date if date != None else date_15_days_ago.strftime('%Y%m%d')), '%Y%m%d')
end_forecast = start_forecast + jdt.timedelta(days = sarima_steps * 7 )

CONFIG_FILE = read_yaml(Path("config/config.yaml"))
PARAMS_FILE = read_yaml(Path("params.yaml"))

with open('features_recipe.json', 'r') as openfile:
    FEATURES_RECIPE = json.load(openfile)

with open('feature_rules.json', 'r') as openfile:
    FEATURE_RULES = json.load(openfile)


def process_level4(level4):
    """Process one Level4_ID independently using absolute paths."""
    try:
        # === Data fetching ===
        data_fetching = DataFetching(CONFIG_FILE["data_fetching"]["server_name"],
                                     CONFIG_FILE["data_fetching"]["database_name"],
                                     start_forecast,
                                     end_forecast,
                                     level4
                                     )

        train_data, forecast_data, forecast_date = data_fetching.run()

        # === Sarima Prediction ===
        sm_predictor = SarimaPredictor()
        forecast_data = sm_predictor(train_data, 
                                     forecast_data,
                                     forecast_date, 
                                     steps = sarima_steps)

        # === Preprocessing ===
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

        # === Model training ===
        trainer = ModelTrainig(train_data_processed,
                               CONFIG_FILE["training"]["validation_size"],  
                               CONFIG_FILE["training"]["models"], 
                               model_name, 
                               PARAMS_FILE
                               )

        best_model, best_params, feature_importances = trainer.run_training()

        # === Model evaluation ===
        evaluator = ModelEvaluation(best_model,
                                    best_params,
                                    train_data_processed,
                                    forecast_data_processed,
                                    feature_importances,
                                    CONFIG_FILE["training"]["validation_size"]
                                    )
        evaluator.run_evaluation()

        # === Check output folder ===
        level4_dir = OUTPUT_DIR / str(level4)
        csv_files = list(level4_dir.glob("*.csv"))
        if csv_files:
            return f"✅ Level4 {level4} completed successfully ({len(csv_files)} file(s) saved)."
        else:
            return f"⚠️ Level4 {level4} finished, but no CSV file found in {level4_dir}."

    except Exception as e:
        traceback.print_exc()
        return f"❌ Level4 {level4} failed: {e}"


if __name__ == "__main__":

    # === Connect to SQL Server ===
    drivers = [driver for driver in pyodbc.drivers() if "SQL Server" in driver]
    driver = drivers[-1]

    odbc_str = (
        f"DRIVER={driver};"
        f"SERVER={CONFIG_FILE['data_fetching']['server_name']};"
        f"DATABASE={CONFIG_FILE['data_fetching']['database_name']};"
        f"Trusted_Connection=yes;"
    )
    connection_string = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(odbc_str)}"
    cnxn = create_engine(connection_string)

    # === Load initial data for Level4 IDs ===
    query = """
        SELECT distinct CodeLevel4
        FROM "Table"
    """
    print("🔄 Reading initial Level4 IDs from database...")
    level4_ids = pd.read_sql_query(query, cnxn)["CodeLevel4"].tolist()
    print(f"✅ Found {len(level4_ids)} unique Level4 IDs.\n")

    # === Run multiprocessing ===
    print(f"🚀 Starting multiprocessing with {num_workers} processes...\n")

    with Pool(processes = num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_level4, level4_ids),
                  total=len(level4_ids),
                  ascii=True))


    # === Combine all output CSV files ===
    print("\n📂 Combining all CSV outputs...")
    output_df = pd.DataFrame()

    for folder in OUTPUT_DIR.iterdir():
        if folder.is_dir():
            csv_files = list(folder.glob("*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
                df["Level4_ID"] = folder.name
                df["TargetMonth"] = target_month
                df = df[["Level4_ID", "Date",  "Forecast", "WeightQTY", "TargetMonth"]]
                output_df = pd.concat([output_df, df], ignore_index=True)

    output_df["insert_date"] = jdt.date.today()
    final_path = OUTPUT_DIR / "total_output.csv"
    output_df.to_csv(final_path, index=False)
    print(f"✅ All results combined and saved to {final_path}")

    if write_to_db == True:
        print("🔄 Writing combined results back to database...")
        output_df.to_sql("OKForecast_MA_Sarima_Forecast", cnxn, if_exists="append", index=False)
        print("✅ Data written to database successfully.")