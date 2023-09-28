import numpy as np
import pandas as pd

def load_data(path, mode='test'):
    data = pd.read_csv(path, index_col=0).reset_index(drop=True)
    data['group_id'] = str(0)
    data["time_idx"] = data.index
    tank_max_height = data["OpeningHeight_readjusted"].max()
    tank_max_volume = data["ClosingHeight_readjusted"].max()
    data['tank_max_height'] = tank_max_height
    data['tank_max_volume'] = tank_max_volume
    data['Month'] = data['Month'].astype(str)
    data['Year'] = data['Year'].astype(str)
    data['period'] = data['period'].astype(str)
    return data