# import library flask untuk membangun aplikasi web, terdapat beberapa library tambahan pada flask seperti render template digunakan untuk merenredering template html
# yang sudah dibuat
from flask import Flask, render_template, request
import numpy as np
# import pickle (not in used)
import joblib # untuk meload model knn yg sudah dibuat

app = Flask(__name__)

# Load model knn menggunakan joblib
model = joblib.load('model_new2.pkl')

# '/' berarti halaman main page yg diisi dengan template home html
@app.route('/')
def home():
    return render_template('home.html')

# "/about" berisi halaman about yg di render dari template about.html
@app.route('/about')
def about():
    return render_template('about.html')

# "/form-predict" berisi halaman pengisian form prediksi penyakit liver yg di render dari template predict.html
@app.route('/form_predict')
def form_predict():
    return render_template('predict.html')

# "/predict" berisi halaman hasil prediksi yg di render dari template about.html, dengan metode post atau menginputkan karakteristik dari penyakit liver
@app.route('/predict', methods=['POST'])
def predict():
    # variabel feature adl sebuah list yg berisi nilai-nilai fitur yg diterima dari permintaan POST pada form-predict
    features = [float(x) for x in request.form.values()]
    # variabel final_features adl list yg berisi array numpy dari variabel features
    final_features = [np.array(features)]
    # variabel prediction adl hasil prediksi yg diperoleh dari model menggunakan final_features
    prediction = model.predict(final_features)

    # variabel output adl hasil prediksi yang dibulatkan menjadi 2 angka desimal
    output = round(prediction[0], 2)

    # melakukan pengecekan setelah membuat label prediction selanjutnya melakukan pengecekan
    # jika output == 1 maka anda terkena liver jika tidak maka tidak terkena Liver
    if output == 1:
        out = 'Anda terkena Liver'
    else:
        out = 'Anda tidak terkena Liver'

    # Setelah melakukan pengecekan maka hasil pengecekan tadi akan dikirimkan ke halaman result_predict.html
    return render_template('result_predict.html', prediction_text='{}'.format(out))

if __name__ == "__main__":
    app.run(debug=True)