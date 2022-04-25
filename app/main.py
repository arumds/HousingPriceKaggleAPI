import uvicorn
import joblib
import xgboost
import pandas as pd
import numpy as np
import json
import pickle
from flask import jsonify
from fastapi import FastAPI, File, UploadFile
from sklearn.base import BaseEstimator, TransformerMixin
from fastapi.responses import JSONResponse
from app import preprocesstest

app = FastAPI(
              title="Ames dataset HousePrice Prediction Model API",
              description="A simple API that use XGBoost model to predict the house price given the features as input in csv"
)

'''Load xgboost model pickle'''
xgb_modelfile = open('/code/app/model/XGBoostRegressor_hpo.pkl',"rb")
xgb_model = pickle.load(xgb_modelfile)


@app.post("/predict-houseprice")
async def predict_code(csv_file: UploadFile = File(...)):
    """
    A simple function that receives housing data and predicts the price of the house.
    Input: invoice csv and
    Return: House price
    """
    userData = pd.read_csv(csv_file.file)
    ID = userData['Id']
    #userData1 = pd.read_csv(csv_file.file,index_col='Id')

    year_features = ['YearBuilt','YearRemodAdd','GarageYrBlt']
    test = preprocesstest.features_age(userData)
    test = preprocesstest.impute_numeric(test)
    test = preprocesstest.testcatfeatures_ordinalmap(test)
    test = preprocesstest.testcatfeatures_ordinal(test) 
    test = preprocesstest.skewness_median(test)
    test = preprocesstest.testfeature_scaling(test)
    test = preprocesstest.test_finalfeatures(test)

    ###predict on new user data
    house_price = xgb_model.predict(test)
    house_price = pd.DataFrame({'Id': ID, 'SalePrice': house_price})
    house_price = house_price.to_json(orient='records')
    return JSONResponse(content=house_price)

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1",port=8000)
