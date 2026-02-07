from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)


model = load_model('best_housing_model.h5' , compile=False)
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    lon = float(request.form['longitude'])
    lat = float(request.form['latitude'])
    age = float(request.form['housing_median_age'])
    rooms = float(request.form['total_rooms'])
    bedrooms = float(request.form['total_bedrooms'])
    pop = float(request.form['population'])
    hh = float(request.form['households'])
    inc = float(request.form['median_income'])
    
    
    ocean_raw = request.form['ocean_proximity'].upper().replace(" ", "")

    
    r_per_h = rooms / hh
    b_per_r = bedrooms / rooms
    p_per_h = pop / hh

    
    data_dict = {
        'longitude': lon,
        'latitude': lat,
        'housing_median_age': age,
        'total_rooms': rooms,
        'total_bedrooms': bedrooms,
        'population': pop,
        'households': hh,
        'median_income': inc,
        'rooms_per_household': r_per_h,
        'bedrooms_per_room': b_per_r,
        'population_per_household': p_per_h,
        
        'ocean_proximity_<1H OCEAN': 1 if ocean_raw in ["<1HOCEAN", "1HOCEAN"] else 0,
        'ocean_proximity_INLAND': 1 if ocean_raw == "INLAND" else 0,
        'ocean_proximity_ISLAND': 1 if ocean_raw == "ISLAND" else 0,
        'ocean_proximity_NEAR BAY': 1 if ocean_raw == "NEARBAY" else 0,
        'ocean_proximity_NEAR OCEAN': 1 if ocean_raw == "NEAROCEAN" else 0
    }

    
    df_input = pd.DataFrame([data_dict])
    features_list = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'rooms_per_household', 'bedrooms_per_room', 'population_per_household', 'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']
    df_input = df_input[features_list]

    
    scaled_data = scaler.transform(df_input)
    prediction_log = model.predict(scaled_data)
    
    
    output = np.expm1(prediction_log)[0][0]

    return render_template('index.html', prediction_text=f'Predicted Price: {output:,.2f}')
if __name__ == "__main__":
    app.run(debug=True)