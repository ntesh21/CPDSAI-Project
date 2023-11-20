from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# import sqlite3 

app = Flask(__name__)

sub_mean = 22982412.06030151
upload_mean = 9187.125628140704
# upload_std = 34151.352253722594

channel_type = ['Music', 'Games', 'Entertainment', 'Education', 'People', 'Sports',
                'Film', 'News', 'Comedy', 'Howto', 'Nonprofit', 'Tech', 'Other',
                'Animals', 'Autos']

countries = ['India', 'United States', 'Japan', 'Russia', 'South Korea',
           'United Kingdom', 'Canada', 'Brazil', 'Argentina', 'Chile', 'Cuba',
           'El Salvador', 'Pakistan', 'Philippines', 'Thailand', 'Colombia',
           'Barbados', 'Mexico', 'United Arab Emirates', 'Spain',
           'Saudi Arabia', 'Indonesia', 'Turkey', 'Venezuela', 'Kuwait',
           'Jordan', 'Netherlands', 'Singapore', 'Australia', 'Italy',
           'Germany', 'France', 'Sweden', 'Afghanistan', 'Ukraine', 'Latvia',
           'Switzerland', 'Vietnam', 'Malaysia', 'China', 'Iraq', 'Egypt',
           'Andorra', 'Ecuador', 'Morocco', 'Peru', 'Bangladesh', 'Finland',
           'Samoa']

def calculate_onehot(data):
    values = np.array(data)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    onehot_dict = {}
    for idx, ct in enumerate(data):
        onehot_dict[ct] = onehot_encoded[idx]
    return onehot_dict

viewers_predictor = pickle.load(open('./models/rf_viewers_predictor', 'rb'))
income_predictor = pickle.load(open('./models/rf_income_predictor', 'rb'))

@app.route('/', methods=('GET', 'POST'))
def index():
    viewers_result = ''
    income_result = ''
    if request.method == 'POST':
        numSub = request.form['numSub']
        try:
            numSub = int(numSub)
        except:
            numSub = sub_mean
        
        numUpload = request.form['numUpload']
        try:
            numUpload = int(numUpload)
        except:
            numUpload = upload_mean
        channelType = request.form['channelType']
        country = request.form['country']
        country_onehot_dict = calculate_onehot(countries)
        country_onehot = list(country_onehot_dict[country])
        channel_onehot_dict = calculate_onehot(channel_type)
        channel_onehot = list(channel_onehot_dict[channelType])
        prediction_data = np.array([int(numSub), int(numUpload)]+country_onehot+channel_onehot)
        # print(prediction_data)
        predicted_viewers = viewers_predictor.predict([prediction_data])[0]
        viewers_result = f"The predicted viewers for this youtube channel is {predicted_viewers}"
        predicted_income = income_predictor.predict([prediction_data])[0]
        income_result = f"The predicted monthly income for this youtube channel is {predicted_income}"
        # print(predicted_viewers)
        # print(precicted_income)

    return render_template('predict.html', channel_type = channel_type, countries = countries, viewers_result=viewers_result, income_result=income_result)

if __name__ == '__main__':
    app.run(debug=True)
