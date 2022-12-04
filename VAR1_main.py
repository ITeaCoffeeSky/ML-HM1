# VAR1_main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import re 
import numpy as np

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

def selling_price_calc(row):
    # возвращает прогнозируемую стоимость авто
    
    model_coef_ = np.array([150402.73037, -29330.27968, 41933.16188, 47378.52736, 284503.56973, 48902.67853,
                            188546.62657, -1004.23705, -98510.88059, -117336.06978, -283405.94728,
                            -40751.01700, -51781.86892, 3307447.75974, -37331.00983, 598017.58381,
                            -6287.45505, -95874.44947, 6532.84706, 49515.02942, 33519.66154, 64985.62063,
                            -141995.09575])
    model_intercept_ = 852455.3938145775
    
    price = row @ model_coef_ + model_intercept_
    
    return round(price, 0)


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    row = itemTOrow(item)
    price = selling_price_calc(row)

    return price


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    prices = []

    for item in items:
        row = itemTOrow(item)
        prices.append(selling_price_calc(row))

    return prices
