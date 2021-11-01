import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import xgboost as xgb

xgb_params = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 10,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

output_file = 'final_model.bin'

# Data Preparation

df = pd.read_csv('click_data.csv')
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.click.values
y_val = df_val.click.values
y_test = df_test.click.values

del df_train['click']
del df_val['click']
del df_test['click']
df_train = df_train.drop(['url_hash'],axis=1)
df_val = df_val.drop(['url_hash'],axis=1)

def train(df_train,y_train,xgb_params):
    dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    dtrain = xgb.DMatrix(X_train, label=y_train,
                             feature_names=dv.get_feature_names())

    model = xgb.train(xgb_params, dtrain, num_boost_round=175)

    return dv, model

def predict(df, dv, model):
    dicts = df.to_dict(orient='records')
    X = dv.transform(dicts)
    dtest = xgb.DMatrix(X, feature_names=dv.get_feature_names())
    y_pred = model.predict(dtest)

    return y_pred

dv, model = train(df_full_train, df_full_train.click.values, xgb_params)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')

# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')