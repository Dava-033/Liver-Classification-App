# import library flask untuk 
from flask import Flask, render_template, request
import numpy as np
# import pickle (not in used)
import joblib

app = Flask(__name__)
model = joblib.load('model_new2.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/form_predict')
def form_predict():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output == 1:
        out = 'Anda terkena Liver'
    else:
        out = 'Anda tidak terkena Liver'

    return render_template('result_predict.html', prediction_text='{}'.format(out))

if __name__ == "__main__":
    app.run(debug=True)