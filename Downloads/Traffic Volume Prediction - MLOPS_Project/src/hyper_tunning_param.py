import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.holiday import *
from pandas.tseries.offsets import CustomBusinessDay


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import holidays
import pickle
import lightgbm as lgb
import xgboost as xgb
import typing
import os

# tracking model
import mlflow
from mlflow import log_metric, log_param, log_artifacts
from download_data import get_data

from hyperopt import fmin , tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

DATA_PATH = '../data'  #can be changed latter
TRAIN_PATH = os.path.join(DATA_PATH, 'train.parquet')
TEST_PATH = os.path.join(DATA_PATH, 'test.parquet')
VALID_PATH = os.path.join(DATA_PATH, 'valid.parquet')

BACKEND_STORE = "sqlite:///../experiment_tracking_log/mlflow.db"

mlflow.set_tracking_uri(BACKEND_STORE)  #change for use in cloud platform
mlflow.set_experiment("Traffic Count Project")
ml_flow_tags = {'author': 'Warrie Warrie',
                'model': 'xgb_baseline'}


def analyse_data(df: pd.DataFrame, target_col: str):
    print("The size of the data is: ", df.shape)
    return None

def preprocess_data_1(df: pd.DataFrame):
    """Groupy by hour to generatehourly traffic flow and create a date_time variable 
    from the time_related columns

    Args:
        df (pd.DataFrame): The main dataframe set

    Returns:
        df: Preprocessed data fram
    """

    # convert the target volume to interger
    df['vol'] = df['vol'].astype(int)

    # aggregate by segment for each daily hr
    df = df.groupby(['segmentid','yr', 'm','d','hh','street']).sum().reset_index().copy()  

    # convert to datetime for datetime processing
    for col in ['m', 'd', 'hh']:
        df[col] = df[col].str.zfill(2)
    df['date_time'] = pd.to_datetime( df['yr']+df['m']+df['d']+(df['hh']),
                                      format='%Y%m%d%H', 
                                      errors='raise' )
    return df

def preprocess_data_2(df: pd.DataFrame, dv:DictVectorizer, fit_dv: bool=False):

    # everything is categorical for now
    categorical_col = ['segmentid','m','d','hh', 'street','week_number', 'quarter', 'week_day','holiday_check']
    # categorical_col = ['hh']

    df[categorical_col] =  df[categorical_col].astype('str')

    train_dicts = df[categorical_col].to_dict(orient='records')
    print(train_dicts)
    if fit_dv:
        X = dv.fit_transform(train_dicts)
    else:
        X = dv.transform(train_dicts)
    return X, dv

def ft_engineering(df: pd.DataFrame):

    # feature engineering for date time which includes
    df['week_number'] = df['date_time'].dt.isocalendar().week
    df['quarter'] = df['date_time'].dt.quarter
    df['week_day'] = df['date_time'].dt.weekday

    # holdays 
    US_hols = holidays.US(years=range(2010, 2025, 1))
    df['holiday_check'] = df.date_time.apply(lambda x: x in US_hols)
    return df

def print_metrics(model, X, Y):
    # Evaluate the data 
    y_pred= model.predict(X)
    rmse = mean_squared_error(Y, y_pred, squared=False)
    # print(rmse)
    return rmse

def dump_pickle(obj, filename):

    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

def convert_to_parquet(df: pd.DataFrame, dest_path: str):
    df.to_parquet( dest_path, engine='pyarrow',compression=None,index= False)


######### Parameter Optimizer ############################

def hyperOpt_train(train, test, valid, runs):    
    # hyper paramter tunning
    search_space = {
        "max_depth": scope.int(hp.quniform('max_depth',4,100,1)),
        "learning_rate": hp.loguniform('learning_rate', -3, 0),
        "reg_alpha": hp.loguniform("reg_alpha", -5,-1),
        "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
        "min_child_weight": hp.loguniform('min_child_weight', -1, 3),
        "objective": 'reg:squarederror',
        'seed': 2021
    } 
    def objective(params):
        with mlflow.start_run():
            
            mlflow.set_tags(ml_flow_tags)

            mlflow.log_params(params)
            # train the model
            xgb_model = xgb.train(params, dtrain=train,evals=[(valid, 'val')], num_boost_round =1000, early_stopping_rounds=100)
            
            # evaluate the model for each optimised iterative
            metrics = {'rmse_train': print_metrics(xgb_model, train, Y_train),
                    'rmse_test':  print_metrics(xgb_model, test,  Y_test),
                    'rmse_valid': print_metrics(xgb_model, valid, Y_valid)}

            mlflow.log_metrics(metrics)
        return {"loss": metrics["rmse_test"], "status": STATUS_OK}

    best_result = fmin(
                fn = objective,
                space = search_space,
                algo = tpe.suggest,
                max_evals=runs,
                trials = Trials()
    )

    print(best_result)

if __name__ =='__main__':

    # read parquet file
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)
    df_valid = pd.read_parquet(VALID_PATH)

    print("Shape of train, valid and test")
    print(df_train.shape, df_valid.shape, df_test.shape)

    # Preprocessing datas
    df_train = ft_engineering(preprocess_data_1(df_train))
    df_test = ft_engineering(preprocess_data_1(df_test))
    df_valid = ft_engineering(preprocess_data_1(df_valid))

    # vectorizer
    dv = DictVectorizer()
    X_train, dv = preprocess_data_2(df_train, dv, fit_dv=True)
    X_valid, _ = preprocess_data_2(df_valid, dv, fit_dv=False)
    X_test, _ = preprocess_data_2(df_test, dv, fit_dv=False)

    # Y values
    Y_train = df_train['vol'].values
    Y_test = df_test['vol'].values
    Y_valid = df_valid['vol'].values

    # Start training
    train = xgb.DMatrix(X_train, label =Y_train)
    test = xgb.DMatrix(X_test, label = Y_test)
    valid = xgb.DMatrix(X_valid, label = Y_valid)

    # Evaluate on each Optimization
    hyperOpt_train(train, test, valid, runs=10)



  




    



    

    


