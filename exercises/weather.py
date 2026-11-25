import requests
import pandas as pd


# Pulls 7-day hourly forecast for Tuscaloosa, AL
url = "https://api.open-mateo.com/v1/forecast"
params = {
    "latitude": 33.21,
    "longitude": -87.57,
    "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
    "temperature_unit": "fahrenheit"
}

response = requests.get(url, params=params)
data = response.json()

df = pd.DataFrame(data["hourly"])
print(df.shape)
