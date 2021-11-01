import pickle

from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb


model_file = 'final_model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('click_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    user = request.get_json()

    X = dv.transform([user])
    Xtest = xgb.DMatrix(X, feature_names=dv.get_feature_names())
    y_pred = model.predict(Xtest)
    click = y_pred >= 0.5

    result = {
        'click_probability': float(y_pred),
        'click': bool(click)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
