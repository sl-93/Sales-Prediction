import urllib
import pyodbc
import pandas as pd
from src.components.fetching.read_sql import read_sql_data
from src.components.fetching.features import add_features


class DataFetching:
 

    def __init__(self, 
                 server_name, 
                 database_name, 
                 start_forecast,
                 end_forecast,
                 level4
                 ):
        
        self.server_name = server_name
        self.database_name = database_name
        self.connection_string = self._build_connection_string()
        self.start_forecast = start_forecast
        self.end_forecast = end_forecast
        self.level4 = level4

    def _build_connection_string(self):
        """
        Docstring for _build_connection_string

        This method, builds connection strings to used to connect to database
        """
      
     
        drivers = [d for d in pyodbc.drivers() if "SQL Server" in d]
        driver = drivers[-1]

        odbc_str = (
            f"DRIVER={driver};"
            f"SERVER={self.server_name};"
            f"DATABASE={self.database_name};"
            f"Trusted_Connection=yes;"
        )


        return f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(odbc_str)}"

    def run(self):

        # Read the data from database
        train_data, forecast_data, val_query = read_sql_data(self.connection_string,
                                                             self.level4,
                                                             self.start_forecast,
                                                             self.end_forecast
                                                             )
        
        train_data, forecast_data = add_features(train_data, 
                                             forecast_data
                                             )

        train_dict = {}
        forecast_dict = {}

        # train_data["WeightQTY_Actual"] = train_data["WeightQTY"]
        forecast_data = pd.merge(forecast_data, val_query[['Date','WeightQTY_Actual']], how="left", on="Date")
        forecast_data["WeightQTY"] = train_data["WeightQTY_Actual"][-14:].median()
        forecast_date = forecast_data[["Date", "Year"]]

        # Create the dictionaries of train and forecast data
        train_dict[self.level4] = train_data
        forecast_dict[self.level4] = forecast_data

        return train_dict, forecast_dict, forecast_date