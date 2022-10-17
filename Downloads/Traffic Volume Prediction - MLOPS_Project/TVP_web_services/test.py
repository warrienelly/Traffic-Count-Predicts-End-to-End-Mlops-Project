import predict
import requests
import json

# input column for prediction
# input_col = ['segmentid','m','d','hh', 'street','week_number', 'quarter', 'week_day','holiday_check']

location_time = {   
                    "segment_id": "10043",
                    "yr": 2020,
                    "m": 12,
                    "d": 30,
                    "hh": 2,
                    "street": "CASTLETON AV"
                    
                    }

url = "http://localhost:2000/predict"
response  = requests.post(url, json=location_time)
print(response.json())
