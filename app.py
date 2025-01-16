from flask import Flask, render_template, request, jsonify, redirect, session
from dotenv import load_dotenv
import os
import requests
from llama_index.llms.gemini import Gemini
from google_auth_oauthlib.flow import Flow
from datetime import timedelta
import re  # Import regex for basic NLP

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')  # Ensure secret key is set
app.permanent_session_lifetime = timedelta(minutes=30)  # Set session lifetime to 30 minutes

# Google API credentials
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
redirect_uri = os.getenv('REDIRECT_URI')

# Weather API endpoint
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

# Initialize Gemini LLM (to be done after OAuth flow)
llm = Gemini(
        model="models/gemini-1.5-flash",
        api_key="AIzaSyDHYgzfdbxO6eIGBAs1iWYHN3dEFpAm5nk"  # Use the token from OAuth flow
    )
clothing_prompt_template = """
Given the following weather conditions, recommend appropriate clothing:
Temperature: {temperature}Â°C
Condition: {condition}
Humidity: {humidity}%
Wind: {wind_kph} kph

Please provide a brief, practical clothing recommendation for these conditions.
"""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    if not llm:
        return jsonify({'error': 'LLM not initialized. Please log in first.'})

    data = request.json
    location = data.get('location')
    print(location)
    if not location:
        return jsonify({'error': 'Please provide a location'})

    try:
        # Get weather data
        weather_response = requests.get(WEATHER_API_URL, params={
            'key': os.getenv('WEATHER_API_KEY'),
            'q': location
        })
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        # Extract current weather data
        current = weather_data['current']
        weather_info = {
            'temperature': current['temp_c'],
            'condition': current['condition']['text'],
            'humidity': current['humidity'],
            'wind_kph': current['wind_kph']
        }

        # Generate clothing recommendation using the Gemini model
        formatted_prompt = clothing_prompt_template.format(**weather_info)
        response = llm.complete(formatted_prompt)

        return jsonify({
            'weather': weather_info,
            'recommendation': response.text
        })

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Weather API error: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400




if __name__ == '__main__':
    app.run(port=5000)