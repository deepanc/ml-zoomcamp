import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model2.bin'
dv_file = 'dv.bin'

app = Flask('churn')

with open(model_file,'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file,'rb') as f_in:
    dv = pickle.load(f_in)

customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]

    result = {
        'churn_probability' : y_pred
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

