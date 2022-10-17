from distutils.log import debug
import json
import pickle
# import pandas as pd
import math
from re import L
from unittest import result
import holidays
import xgboost as xgb

from datetime import datetime
from flask import Flask, request, jsonify


def preprocess_feat(feat: dict):
    # convert to datetime for datetime processing
    # df_data = pd.DataFrame(dict)
    feat['date_time'] = datetime(feat['yr'], feat['m'], feat['d'], feat['hh'])
    
    # feature engineering for date time which includes
    feat['week_number'] = feat['date_time'].isocalendar().week
    feat['quarter'] = int(math.ceil(feat['m']/3.))
    feat['week_day'] = feat['date_time'].weekday()

    # holdays features
    US_hols = holidays.US(years=[feat['yr']])
    feat['holiday_check'] = feat['date_time'] in US_hols

        # convert data to string
    for value in ['m', 'd', 'hh']:
        feat[value] = str(feat[value]).zfill(2)

    del feat['date_time']
    return feat

def single_predict(feats):
    # web service deplpyment for the model of interest
    with open("../models/xgb_baseline.bin", "rb") as f_in:
        (dv, xgb_model) =  pickle.load(f_in)
        X = dv.transform(feats)
        X = xgb.DMatrix(X)
        preds= xgb_model.predict(X)
        return float(preds[0])


app = Flask('Traffic_Volume')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    
    # get input from ApI

    location_time = request.get_json()
    print(location_time)
    feats= preprocess_feat(location_time)
    preds = single_predict(feats)

    # return result as a dictionary
    result = {
        "Traffic_volume": preds
    }
    return result

if __name__ == "__main__":
    app.run(debug=True, port=2000)

