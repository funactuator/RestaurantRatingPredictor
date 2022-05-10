import numpy as np
import sys
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    # print("Kuldeep")
    # print(list(request.form.values()), flush=True)
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    print("reached here", flush = True)
    output = round(prediction[0], 1)
    print(output, flush=True)
    return render_template('predict.html', prediction_text='Your Rating is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=False)