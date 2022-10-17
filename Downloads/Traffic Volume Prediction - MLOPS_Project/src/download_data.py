import os
import pandas as pd
from datetime import datetime
from sodapy import Socrata



def get_data(state: str, year: tuple):
    """
    Download data from Scorata Online based on the year required in
    Staten Island
    """
    query = f"""
    select 
    * 
    where 
    yr IN {year}
    and boro = 'Staten Island'
    
    limit
    1000000
    """
    api_key= "3n6p17M38sFlzBvS1cnosPxah"
    client = Socrata("data.cityofnewyork.us", api_key)
    client.timeout = 100000

    results = client.get("7ym2-wayt",  query=query)
    results_df = pd.DataFrame.from_records(results)   # Convert to pandas DataFrame
    return results_df

def convert_to_parquet(df: pd.DataFrame, dest_path: str):
    df.to_parquet( dest_path, engine='pyarrow',compression=None,index= False)

def read_downloaded_data(file_path: str, save_to_local=False):
    """This fuction is meant to downlad the data from the API specified

    Args:
        file_path (str): The path to the download location of the file
        save_to_local (bool, optional): Specify if to save the downloaded data on our local device.
         Defaults to False.
    """
    # run to download the data online
    df_train = get_data(state='Staten Island', year=(2011,2012,2013))
    df_valid = get_data(state='Staten Island', year=(2014,2015,2016))
    df_test= get_data(state='Staten Island', year=(2017,2018))

    if save_to_local==True:
        # run this code to convert the data to parquet to avoid hitting the API everytime since the data is static.
        convert_to_parquet(df_train, os.path.join(DATA_PATH, 'train.parquet'))
        convert_to_parquet(df_test, os.path.join(DATA_PATH, 'test.parquet'))
        convert_to_parquet(df_valid, os.path.join(DATA_PATH, 'valid.parquet'))


if __name__ =='__main__':

    DATA_PATH = "../data"
    read_downloaded_data(DATA_PATH, save_to_local=True )



