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
import xgboost as xgb
import typing
import os

# tracking model
import mlflow
from mlflow.tracking import MlflowClient

from download_data import get_data
from hyper_tunning_param import * 

from hyperopt import fmin , tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

DATA_PATH = '../data'  #can be changed latter
TRAIN_PATH = os.path.join(DATA_PATH, 'train.parquet')
TEST_PATH = os.path.join(DATA_PATH, 'test.parquet')
VALID_PATH = os.path.join(DATA_PATH, 'valid.parquet')

BACKEND_STORE = "sqlite:///../experiment_tracking_log/mlflow.db"
client = MlflowClient(tracking_uri = BACKEND_STORE)

mlflow.set_tracking_uri(BACKEND_STORE)  #change for use in cloud platform
mlflow.set_experiment("Traffic Count Project")
ml_flow_tags = {'author': 'Warrie Warrie',
                'model': 'xgb_baseline'}
    

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
    ######### Model Training ############################
    with mlflow.start_run():
        mlflow.set_tags(ml_flow_tags)

        params = {
            "max_depth": 12,
            "learning_rate": 0.6963228530261002,
            "reg_alpha": 0.2068319417533728,
            "reg_lambda": 0.0026269553821513204,
            "min_child_weight": 0.9840443652428528,
            "objective": 'reg:squarederror',
            'seed': 2021
        } 

        mlflow.log_params(params)


        # train the model
        xgb_model = xgb.train(params, dtrain=train,evals=[(valid, 'val')], num_boost_round =1000, early_stopping_rounds=100)
        
        # evaluate the model for each optimised iterative
        metrics = {'rmse_train': print_metrics(xgb_model, train, Y_train),
                'rmse_test':  print_metrics(xgb_model, test,  Y_test),
                'rmse_valid': print_metrics(xgb_model, valid, Y_valid)}

        mlflow.log_metrics(metrics)
        # mlflow.xgboost.log_model(xgb_model, artifact_path="xgb_models")

        # save model
        dump_pickle((dv, xgb_model), "../models/xgb_baseline.bin")
    

    


