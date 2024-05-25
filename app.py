import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Function to train the model
def train_model():
    url = 'https://earthquake.usgs.gov/fdsnws/event/1/query'
    params = {
        'format': 'geojson',
        'starttime': '2023-01-01',
        'endtime': '2023-12-31',
        'minmagnitude': 5
    }
    response = requests.get(url, params=params)
    data = response.json()

    earthquakes = [
        {
            'time': feature['properties']['time'],
            'latitude': feature['geometry']['coordinates'][1],
            'longitude': feature['geometry']['coordinates'][0],
            'depth': feature['geometry']['coordinates'][2],
            'magnitude': feature['properties']['mag'],
            'place': feature['properties']['place']
        }
        for feature in data['features']
    ]

    df = pd.DataFrame(earthquakes)
    df = df.dropna()
    df['magnitude_normalized'] = (df['magnitude'] - df['magnitude'].mean()) / df['magnitude'].std()
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute

    features = ['latitude', 'longitude', 'depth', 'magnitude', 'year', 'month', 'day', 'hour', 'minute']
    df['label'] = df['magnitude'].apply(lambda x: 1 if x >= 6 else 0)

    X = df[features]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test.columns.tolist()

# Train the model
model, feature_columns = train_model()

# Streamlit app
st.title("Earthquake Prediction Model")

st.write("""
Enter the details of the earthquake to predict if it's significant (magnitude >= 6).
""")

latitude = st.number_input('Latitude', value=0.0)
longitude = st.number_input('Longitude', value=0.0)
depth = st.number_input('Depth', value=0.0)
magnitude = st.number_input('Magnitude', value=0.0)
year = st.number_input('Year', value=2023)
month = st.number_input('Month', value=1)
day = st.number_input('Day', value=1)
hour = st.number_input('Hour', value=0)
minute = st.number_input('Minute', value=0)

data = {
    'latitude': latitude,
    'longitude': longitude,
    'depth': depth,
    'magnitude': magnitude,
    'year': year,
    'month': month,
    'day': day,
    'hour': hour,
    'minute': minute
}

if st.button('Predict'):
    features = pd.DataFrame(data, index=[0])
    features = features.reindex(columns=feature_columns, fill_value=0)
    prediction = model.predict(features)
    result = 'Significant' if prediction[0] == 1 else 'Not Significant'
    st.write(f'The earthquake is predicted to be: **{result}**')

if st.button('Show raw data'):
    st.write(data)
