import pickle

model_file = 'model1.bin.1'
dv_file = 'dv.bin.1'

with open(model_file,'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file,'rb') as f_in:
    dv = pickle.load(f_in)

customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}

X = dv.transform([customer])
y_pred = model.predict_proba(X)[0,1]
print(y_pred)