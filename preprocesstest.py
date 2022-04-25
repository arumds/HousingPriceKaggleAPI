# %%
import pandas as pd
import numpy as np
from scipy.stats import skew
import pickle
from scipy import stats
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import xgboost
from sklearn.model_selection import RandomizedSearchCV


def find_missing_datatypes(dataset):
    features_nan=[features for features in dataset.columns if dataset[features].isnull().sum()>0]
    print(' Total Features with missing values:', len(features_nan))
    print(features_nan)

    ## Get categorical features (dtypes == "object")
    catfeatures=[feature for feature in dataset.columns if dataset[feature].dtype =='object']
    print(' Total Categorical Features:', len(catfeatures))
    print(catfeatures)
    
    ## Get categorical features (dtypes == "object") with missing values
    catfeatures_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>0 and dataset[feature].dtypes=='object']
    print(' Total Categorical Features with missing values:', len(catfeatures_nan))
    print(catfeatures_nan)


    ## Get numerical features (dtypes != "object")
    numfeatures=[feature for feature in dataset.columns if dataset[feature].dtype !='object']
    print(' Total Numerical Features:', len(numfeatures))
    print(numfeatures)

    ## Get numerical features (dtypes != "object") with missing values
    numfeatures_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>0    and dataset[feature].dtypes!='object']
    print(' Total Numerical Features with missing values:', len(numfeatures_nan))
    print(numfeatures_nan)

    return features_nan, catfeatures, catfeatures_nan, numfeatures, numfeatures_nan


year_features = ['YearBuilt','YearRemodAdd','GarageYrBlt']

## Date Time Variables, convert to age of house feature
def features_age(dataset):
    dataset['GarageYrBlt'].fillna('0', inplace=True)
    return(dataset)

def impute_object(dataset):
    
    dataset['MSSubClass']=dataset['MSSubClass'].astype(object)

    catfeatures=[feature for feature in dataset.columns if dataset[feature].dtype =='object' ]

    cat_nan_nofeature =['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
    'BsmtFinType2','FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
    'PoolQC', 'Fence', 'MasVnrType', 'MiscFeature']

    for feature in cat_nan_nofeature:
        dataset[feature].fillna('No Feature', inplace=True)

    dataset['Electrical'].fillna("SBrkr", inplace=True)
    numfeatures =[feature for feature in dataset.columns if  dataset[feature].dtypes!='object' ]
    catfeatures= [feature for feature in dataset.columns if dataset[feature].dtype =='object' and feature not in cat_nan_nofeature ]
    
    #year_price = [year_features + ['SalePrice']]

    features=[feature for feature in dataset.columns]

    pkl_file = open('/code/app/model/CatGroupbyImputer.pkl', 'rb')
    catfeatures_test = pickle.load(pkl_file) 
    pkl_file.close()
    dataset = catfeatures_test.transform(dataset)

    return(dataset)

def impute_numeric(test):
    year_features = ['YearBuilt','YearRemodAdd','GarageYrBlt']
    numfeatures =[feature for feature in test.columns if  test[feature].dtypes!='object' and feature not in year_features]
    for feature in numfeatures:
        test[feature].fillna(test[feature].median(), inplace=True)
        return test

    return test
    

def preprocess_data(data):
    find_missing_datatypes(data)
    features_age(data)
    impute_object(data)
    impute_numeric(data) 
    #impute_missing(data) 
    return(data)


def testcatfeatures_ordinalmap(test):
    pkl_file = open('/code/app/model/catfeatures_ordinalmap_encoder.pkl', 'rb')
    ordinalmap_test = pickle.load(pkl_file) 
    pkl_file.close()
    test = ordinalmap_test.transform(test)
    return test

def testcatfeatures_ordinal(test):
    pkl_file = open('/code/app/model/catfeatures_ordinal_encoder.pkl', 'rb')
    ordinal_test = pickle.load(pkl_file) 
    pkl_file.close()
    test = ordinal_test.transform(test)
    return test
    

def test_skewness(test):
    test = test.drop(skewed_features,axis=1)
    #test = test.drop(skewed_features].astype(float)
    #test[skewed_features] = np.log(1+test[skewed_features])
    return test


##check and fix skewness in final numeric variables
def skewness_median(test):
    drop_features=['TotRmsAbvGrd','GarageArea' ,'1stFlrSF']
    numfeatures=[feature for feature in test.columns if test[feature].dtype !='object']
    num_features = [ feature for feature in numfeatures if feature not in year_features + drop_features]
    skewness = test[num_features].skew().sort_values(ascending=False)
    skewed_features = list(skewness[abs(skewness) > 0.5].index)
    for feature in skewed_features:
        test.sort_values(by=feature, ascending=True, na_position='last')
        q1, q3 = np.nanpercentile(test[feature], [25,75])
        iqr = q3-q1
        lower_bound = q1-(1.5*iqr)
        upper_bound = q3+(1.5*iqr)
        median = test[feature].median()
        test.loc[test[feature] < lower_bound, [feature]] = median
        test.loc[test[feature] > upper_bound, [feature]] = median
        return test

def testfeature_scaling(test):
    features_scale=[feature for feature in test.columns if feature not in ['Id'] ]
    pkl_file = open('/code/app/model/MinMaxScaler.pkl', 'rb')
    scaler_test = pickle.load(pkl_file) 
    pkl_file.close()
    test[features_scale] = scaler_test.transform(test[features_scale])
    test = test[selected_feat]
    return test

selected_feat = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 
    'Alley','LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
    'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
    'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
    'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
    'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
    'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
    'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
    'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MoSold',
    'YrSold', 'SaleType', 'SaleCondition' ]
    
def test_finalfeatures(test):
    test = test[selected_feat]
    #test = test.drop(skewed_features].astype(float)
    #test[skewed_features] = np.log(1+test[skewed_features])
    return test
    

