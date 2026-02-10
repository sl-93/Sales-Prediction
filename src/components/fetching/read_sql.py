import pandas as pd
from sqlalchemy import create_engine


def read_sql_data(connection_string: str, 
                  level4: str,
                  start_forecast,
                  end_forecast):
    
    cnxn = create_engine(connection_string, pool_pre_ping=True, poolclass=None)

    query = f"""
        WITH cte AS (
            SELECT D.ID, L.LocationID, L.DistrictID
            FROM "Table" L
            LEFT JOIN "Table" D ON L.LocationID = D.LocationID
            where D.DepTypeSN = 2 and D.IsActive = 1
        )
        SELECT 
            I.CodeLevel4 AS Level4_ID,
            D.PersianInt AS Date,
            D.PersianDayOfWeekInt AS WeekDays,
            D.PersianMonthNo AS Month,
            D.PersianDayInMonth AS Day,
            D.PersianYearMonthInt AS YearMonth,
            D.PersianWeekOfYearNo AS Sol_WeekOfYear,
            D.PersianWeekOfMonthNo AS MonthWeeks,
            D.PersianYearInt AS Year,
            CAST(D.HasOKHoliday AS INT) AS IsHolliday,
            D.Occastion_ID,
            SUM(QTY) QTY,
            SUM(Gross) Gross,
            SUM(DiscountAmount) DiscountAmount,
            SUM(WeightQTY) WeightQTY
        FROM "Table" S
        LEFT JOIN cte ON cte.ID = S.COM_Dim_InventLocationRef
        LEFT JOIN "Table" I ON I.ID = S.Level4_ID
        LEFT JOIN "Table" D ON D.DateKey = S.COM_Dim_DateRef
		where  I.CodeLevel4 = '{level4}' and D.PersianInt <= {start_forecast.strftime('%Y%m%d')}
        GROUP BY
            I.CodeLevel4, D.PersianInt,
            D.PersianDayOfWeekInt, D.PersianMonthNo,
            D.PersianDayInMonth, D.PersianYearMonthInt,
            D.PersianWeekOfYearNo, D.PersianWeekOfMonthNo,
            D.PersianYearInt, D.HasOKHoliday, D.Occastion_ID
		order by Level4_ID, Date
    """

    forecast_query = f"""
        SELECT 
            PersianInt AS Date,
            PersianDayOfWeekInt AS WeekDays,
            PersianMonthNo AS Month,
            PersianWeekOfMonthNo AS MonthWeeks,
            PersianDayInMonth AS Day,
            PersianYearMonthInt AS YearMonth,
            PersianWeekOfYearNo AS Sol_WeekOfYear,
            PersianYearInt AS Year,
            CAST(HasOKHoliday AS INT) AS IsHolliday,
            Occastion_ID
        "Table" 
        WHERE PersianInt > {start_forecast.strftime('%Y%m%d')}
          AND PersianInt <= {end_forecast.strftime('%Y%m%d')}
        ORDER BY PersianInt
    """

    val_query = f"""
        WITH cte AS (
            SELECT D.ID, L.LocationID, L.DistrictID
            FROM "Table" L
            LEFT JOIN "Table" D ON L.LocationID = D.LocationID
            where D.DepTypeSN = 2 and D.IsActive = 1
        )
        SELECT 
            I.CodeLevel4 AS Level4_ID,
            D.PersianInt AS Date,
            SUM(WeightQTY) WeightQTY_Actual
        FROM "Table" S
        LEFT JOIN cte ON cte.ID = S.COM_Dim_InventLocationRef
        LEFT JOIN "Table" I ON I.ID = S.Level4_ID
        LEFT JOIN "Table" D ON D.DateKey = S.COM_Dim_DateRef
		where  I.CodeLevel4 = '{level4}' and D.PersianInt > {start_forecast.strftime('%Y%m%d')}
        GROUP BY
            I.CodeLevel4, D.PersianInt
		order by Level4_ID, Date
    """

    long_weekend_calendar_query = f"""
        SELECT *
        FROM [Forecasting].[Long_Weekend_Calendar]
        order by date
    """

    event_effect_query = f"""
        select 
        Level4,
        PersianInt,
        Modified_Actual * 1000 AS Modified_Actual
        from "Table"
        where level4 = '{level4}'
    """

    # Create the dataframes of train and forecast sets
    train_data = pd.read_sql_query(query, cnxn).sort_values("Date")
    forecast_data = pd.read_sql_query(forecast_query, cnxn).sort_values("Date")
    val_query = pd.read_sql_query(val_query, cnxn).sort_values("Date")

    # Create the dataframe of longweekend
    long_weekend = pd.read_sql_query(long_weekend_calendar_query, cnxn).astype(float)

    train_data = pd.merge(train_data, long_weekend, on="Date", how="left")
    forecast_data = pd.merge(forecast_data, long_weekend, on="Date", how="left")
    event_effect = pd.read_sql_query(event_effect_query, cnxn).sort_values(by="PersianInt")

    train_data["WeightQTY_Actual"] = train_data["WeightQTY"]
    if len(event_effect) > 0:
        ee = (event_effect[event_effect["Level4"] == level4].reset_index(drop=True))

        # Replace WeightQTY only on affected dates
        train_data["WeightQTY"] = (train_data["Date"]
                                   .map(ee.set_index("PersianInt")["Modified_Actual"])
                                   .fillna(train_data["WeightQTY"])
                                   )


    return train_data, forecast_data, val_query
