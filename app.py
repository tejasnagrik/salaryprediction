import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    exp = request.form.get('experience')
    test_score = request.form.get('test_score')
    inter_score = request.form.get('interview_score')

    int_features = [int(exp), int(test_score), int(inter_score) ]


    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

