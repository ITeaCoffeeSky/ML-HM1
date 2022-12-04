from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import re 
import numpy as np
import pandas as pd
import pickle
import sklearn

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

def unscale(x, scaler_mean_, scaler_scale_):
    # стандартизация значения
    return (x - scaler_mean_)/scaler_scale_

def itemTOrow(item: Item):
    # по item на вход формирует на выход список с коэффициентами согласно модели
    
    row = []
    scaler_mean_ = [2013.42842, 73952.24247, 19.48108, 1435.40479, 88.14983]
    scaler_scale_ = [4.09527, 60065.99321, 3.87009, 484.50481, 31.55576]
    fuel = {'Diesel': [1, 0, 0], 'Petrol': [0, 0, 1], 'LPG': [0, 1, 0], 'CNG': [0, 0, 0]}
    seller_type = {'Individual': [1, 0], 'Dealer': [0, 0], 'Trustmark Dealer': [0, 1]}
    transmission = {'Manual': [1], 'Automatic': [0]}
    owner = {'First Owner': [0, 0, 0, 0], 'Second Owner': [0, 1, 0, 0], 'Third Owner': [0, 0, 0, 1], 
             'Fourth & Above Owner': [1, 0, 0, 0], 'Test Drive Car': [0, 0, 1, 0]}
    seats = {5: [0, 1, 0, 0, 0, 0, 0, 0], 4: [1, 0, 0, 0, 0, 0, 0, 0], 7: [0, 0, 0, 1, 0, 0, 0, 0], 
             8: [0, 0, 0, 0, 1, 0, 0, 0], 6: [0, 0, 1, 0, 0, 0, 0, 0], 9: [0, 0, 0, 0, 0, 1, 0, 0],
             10: [0, 0, 0, 0, 0, 0, 1, 0], 14: [0, 0, 0, 0, 0, 0, 0, 1], 2: [0, 0, 0, 0, 0, 0, 0, 0]}
    
    row.append(unscale(item.year, scaler_mean_[0], scaler_scale_[0]))
    row.append(unscale(item.km_driven, scaler_mean_[1], scaler_scale_[1]))
    mileage = float(re.sub(r'[/ a-zA-Z]','',item.mileage))
    row.append(unscale(mileage, scaler_mean_[2], scaler_scale_[2]))
    engine = float(re.sub(r'[/ a-zA-Z]','',item.engine))
    row.append(unscale(engine, scaler_mean_[3], scaler_scale_[3]))
    max_power = float(re.sub(r'[/ a-zA-Z]','',item.max_power))
    row.append(unscale(max_power, scaler_mean_[4], scaler_scale_[4]))
    
    row.extend(fuel[item.fuel])
    row.extend(seller_type[item.seller_type])
    row.extend(transmission[item.transmission])
    row.extend(owner[item.owner])
    row.extend(seats[item.seats])   
    
    return row


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    row = np.asarray(itemTOrow(item))
    row = row.reshape(1, -1)
    loaded_model = pickle.load(open('model.pickle', 'rb'))
    result = loaded_model.predict(row)

    return result[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    rows = []
    
    for item in items:
        row = itemTOrow(item)
        rows.append(row)
    
    # rows = np.asarray(rows)

    loaded_model = pickle.load(open('model.pickle', 'rb'))
    result = loaded_model.predict(rows)
    
    return result
