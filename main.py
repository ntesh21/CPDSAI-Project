from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# import sqlite3 

app = Flask(__name__)


views_mean = 11249775535.65747
views_std = 14808945108.497347
views_max = 228000000000.0
views_min = 0.0
upload_mean = 9187.125628140704
upload_std = 36352.7837345139
upload_max = 301308
upload_min = 0
avg_mon_mean = 334093.7098103448
avg_mon_std = 619998.0598726937
avg_mon_max = 7225450.0
avg_mon_min = 0
avg_year_mean = 4011713.708798851
avg_year_std = 7445617.173565455
avg_year_max = 7225450.0
avg_year_min = 0.0

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

def scaling(data, min, max):
    scaled_value  = (data - min)/(max - min)
    return scaled_value

# viewers_predictor = pickle.load(open('./models/rf_viewers_predictor', 'rb'))
# income_predictor = pickle.load(open('./models/rf_income_predictor', 'rb'))
subscribers_predictor = pickle.load(open('./models/model6', 'rb'))

@app.route('/', methods=('GET', 'POST'))
def index():
    subscribers_result = ''
    # income_result = ''
    if request.method == 'POST':
        numView = request.form['numView']
        try:
            numView = int(numView)
        except:
            numView = sub_view
        scaled_numView = scaling(numView, views_min, views_max)
        
        numUpload = request.form['numUpload']
        try:
            numUpload = int(numUpload)
        except:
            numUpload = upload_mean
        scaled_numUpload = scaling(numUpload, upload_min, upload_max)

        avg_monthly_income = request.form['monInc']
        print(avg_monthly_income)
        try:
            avg_monthy_income = int(avg_monthy_income)
        except:
            avg_monthly_income = avg_mon_mean
        scaled_avg_monthly_income = scaling(avg_monthly_income, avg_mon_min, avg_mon_max)

        avg_yearly_income  = request.form['yearInc']
        try:
            avg_yearly_income = int(avg_yearly_income)
        except:
            avg_yearly_income = avg_year_mean
        scaled_avg_yearly_income = scaling(avg_yearly_income, avg_year_min, avg_year_max)
        # channelType = request.form['channelType']
        # country = request.form['country']
        # country_onehot_dict = calculate_onehot(countries)
        # country_onehot = list(country_onehot_dict[country])
        # channel_onehot_dict = calculate_onehot(channel_type)
        # channel_onehot = list(channel_onehot_dict[channelType])
        prediction_data = np.array([scaled_numView, scaled_numUpload, scaled_avg_monthly_income, scaled_avg_yearly_income])
        # print(prediction_data)
        predicted_subscribers = subscribers_predictor.predict([prediction_data])[0]
        subscribers_result = f"The predicted subscribers for this youtube channel is {predicted_subscribers}"
        # predicted_income = income_predictor.predict([prediction_data])[0]
        # income_result = f"The predicted monthly income for this youtube channel is {predicted_income}"
        # print(predicted_viewers)
        # print(precicted_income)

    return render_template('predict.html', channel_type = channel_type, countries = countries, subscribers_result=subscribers_result)

if __name__ == '__main__':
    app.run(debug=True)
