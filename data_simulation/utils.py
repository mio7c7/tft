import numpy as np
import pandas as pd
def assign_period(df):
    # idle = 0, transaction = 1, delivery = 2
    conditions = [
        (df['Del_tc'] != 0),
        (df['Sales_Ini'] == 0),
        (df['Sales_Ini'] != 0) & (df['Del_tc'] == 0),
    ]
    # conditions = [
    #     (df['Del_tc'] != 0),
    #     (df['Sales_tc'] == 0),
    #     (df['Sales_tc'] != 0) & (df['Del_tc'] == 0),
    # ]
    values = [2, 0, 1]
    df['period'] = np.select(conditions, values)
    deli = list(df.index[df['period'] == 2])
    for ind in deli:
        for i in range(3):
            df.loc[ind - i, 'period'] = 2
    return df

def df_format(df):
    retained = df[['Time', 'Time_DN', 'TankTemp', 'OpeningStock_readjusted', 'ClosingStock_readjusted',
                   'OpeningHeight_readjusted', 'ClosingHeight_readjusted', 'ClosingStock_tc_readjusted', 'OpeningStock_tc_readjusted',
                   'ClosingHeight_tc_readjusted', 'OpeningHeight_tc_readjusted', 'Sales_Ini', 'Sales_Ini_tc',
                    'Del_tc', 'Del', 'Var_readjusted', 'Var_tc_readjusted', 'period']]
    retained['Time'] = pd.to_datetime(retained['Time'])
    retained['Year'] = retained['Time'].dt.year
    retained['Month'] = retained['Time'].dt.month
    def time_of_day(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 24:
            return 'evening'
        else:
            return 'night'
    retained['Time_of_day'] = retained['Time'].dt.hour.apply(time_of_day)
    def season(month):
        if 9 <= month <= 11:
            return 'spring'
        elif 3 <= month <= 5:
            return 'autumn'
        elif 6 <= month <= 8:
            return 'winter'
        else:
            return 'summer'
    retained['Season'] = retained['Month'].apply(season)
    return retained