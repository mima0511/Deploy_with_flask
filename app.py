from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

data_preparation = joblib.load(open('dataPreparation.pkl', 'rb'))
final_model = joblib.load(open('RandomForestRegressor.pkl', 'rb'))

app = Flask(__name__)
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def home():
    longitude = request.form.get('longitude')
    latitude= request.form.get('latitude')
    housing_median_age= request.form.get('housing_median_age')
    total_rooms= request.form.get('total_rooms')
    total_bedrooms = request.form.get('total_bedrooms')
    population = request.form.get('population')
    households = request.form.get('households')
    median_income = request.form.get('median_income')
    ocean_praximity = request.form['ocean_proximity']

    #feature extraction
    rooms_per_household= float(total_rooms) / float(households)
    bedrooms_per_room= float(total_bedrooms)/float(total_rooms)
    population_per_household= float(population)/ float(households)

    features = np.array([longitude,latitude,housing_median_age,total_rooms,population,households,median_income,ocean_praximity,rooms_per_household,bedrooms_per_room,population_per_household])
    features_df = pd.DataFrame(data=[features],columns=['longitude','latitude','housing_median_age','total_rooms','population','households','median_income','ocean_proximity','rooms_per_household','bedrooms_per_room','population_per_household'])

    clean_features = data_preparation.transform(features_df)
    prediction = final_model.predict(clean_features)

    return render_template("index.html", prediction_text='The predictedPrice of this house is : {}'.format(prediction))


    if __name__ == "__main__":
        app.run(debug=True)