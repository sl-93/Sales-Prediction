import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import jdatetime as jdt
from src.utils.utils import get_year_week


class SarimaPredictor:

    def __call__(self, 
                 train_data_general, 
                 forecast_data_general,
                 forecast_dates, 
                 steps=9):
        """Generates SARIMA forecast for the given training data and forecast dates.
        Args:
            train_data_general (Dictionary): Training data containing different level4s each containing 'Level4_ID', 'Date', and 'WeightQTY' columns.
            forecast_data_general (Dictionary): Dictionary containing different level4s each containing 'Date' column for which forecast is to be generated.
            forecast_dates (DataFrame): DataFrame containing 'Date' column for date-related features.
            steps (int): Number of steps to forecast. Default is 9.
            Returns:'
            daily_forecast (DataFrame): DataFrame containing the forecasted 'WeightQTY' for each date."""
        
        for id in train_data_general:
            try:
                train_data = train_data_general[id]
                forecast_data = forecast_data_general[id]
                temp_train_data = train_data[['Date','Year','WeightQTY_Actual']].reset_index(drop = True)
                #### Convert PersianInts into jdatetime and extract week of year for each date
                forecast_dates = forecast_data[['Date','Year']]
                temp_train_date =  train_data[['Date','Year']].reset_index(drop = True)
                temp_forecast_date = forecast_dates
                temp_dates = pd.concat([temp_train_date, temp_forecast_date]).reset_index(drop = True)
                temp_dates['jdate'] = [jdt.datetime.strptime(str(x),'%Y%m%d') for x in temp_dates.Date]
                temp_dates['WeekofYear'] = temp_dates['jdate'].apply(get_year_week)

                weekly_dates_refs = temp_dates.groupby(['Year','WeekofYear']).agg(C=('Date','nunique')).reset_index()
                weekly_dates_refs.reset_index(drop = False, inplace = True)

                temp_train_data = pd.merge(temp_train_data, temp_dates[['Date','WeekofYear']], on = 'Date', how = 'inner', copy = False)
                Week_Days_Count = temp_train_data.groupby(['Year','WeekofYear']).agg(C=('Date','nunique')).reset_index()

                last_week = Week_Days_Count.WeekofYear[-1:].values[0]
                last_week_days = Week_Days_Count.C[-1:].values[0]
                last_week_year = Week_Days_Count.Year[-1:].values[0]

                ## Estimate coef data
                daily_sales = train_data.groupby(['Level4_ID','WeekDays']).agg(WeightQTY=('WeightQTY', 'sum')).reset_index()
                weekly_sales = train_data.groupby(['Level4_ID']).agg(W_WeightQTY=('WeightQTY', 'sum')).reset_index()
                daily_coef = pd.merge(daily_sales,weekly_sales,on = 'Level4_ID',copy = False)
                daily_coef = daily_coef.assign(daily_shares = daily_coef.WeightQTY/daily_coef.W_WeightQTY)

                start_jdate = jdt.datetime(1401,1,1)
                gregorian_dates = [
                    (start_jdate + jdt.timedelta(i-1)).togregorian()
                    for i in np.cumsum(Week_Days_Count.C)]

                if (last_week<53) & (last_week_days<7):
                    idx = temp_train_data[(temp_train_data.Year==last_week_year) & (temp_train_data.WeekofYear==last_week)].index
                    temp_train_data.drop(index = idx, inplace = True )
                    gregorian_dates = gregorian_dates[:-1]
                    
                ## SARIMA Model
                weekly_sales = temp_train_data.groupby(['Year','WeekofYear']).agg(W = ('WeightQTY_Actual' , 'sum')).reset_index(drop = False)
                weekly_sales.index = pd.to_datetime(gregorian_dates)

                try:
                    model = SARIMAX(
                        pd.to_numeric(weekly_sales.W), order=( 1, 0, 1), seasonal_order=(1, 1, 1, 53)
                        ).fit()
                    
                    forecast_index = pd.date_range(
                        start = weekly_sales.index[-1] + pd.Timedelta(weeks=1),
                        periods = steps, freq='7D' )

                    forecast = model.forecast(steps = steps)
                except:
                    print("Model didn't converge")

                forecast_df = forecast.to_frame()
                forecast_df.reset_index(inplace = True , drop = False)
                forecast_df = pd.merge(forecast_df, weekly_dates_refs , on = 'index', how = 'inner', copy = False)
                daily_forecast = pd.merge(forecast_df,temp_dates, on = ['Year','WeekofYear'], how = 'inner', copy = False)
                daily_forecast['Level4_ID'] = id
                daily_forecast['WeekDays'] = [x.weekday() for x  in daily_forecast.jdate]
                daily_forecast = pd.merge(daily_forecast, daily_coef, on = ['Level4_ID', 'WeekDays'],how = 'inner' ,copy = False)
                daily_forecast = daily_forecast.drop_duplicates()
                
                ## Modify the daily shares for the last week of year
                daily_forecast.loc[daily_forecast['WeekofYear']==53,'daily_shares'] = daily_forecast.loc[daily_forecast['WeekofYear']==53,'C']*0.38
                
                daily_forecast['WeightQTY'] = daily_forecast['predicted_mean']*daily_forecast['daily_shares']

                forecast_data = forecast_data.drop(columns=["WeightQTY"]).reset_index(drop=True)
                forecast_data = pd.merge(forecast_data.reset_index(drop=True), 
                                         daily_forecast[["Date", "WeightQTY"]],
                                         on="Date", 
                                         how="left")
                
                forecast_data = forecast_data[:len(daily_forecast)]
                forecast_data["SarimaOutput"] = forecast_data["WeightQTY"]
                forecast_data_general[id] = forecast_data
            except Exception as e:
                print(e)

        return forecast_data_general