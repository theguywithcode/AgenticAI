from flask import Flask, render_template, request, jsonify, redirect, session
from dotenv import load_dotenv
import os
import requests
from google_auth_oauthlib.flow import Flow
from datetime import timedelta
from llama_index import SimpleKeywordTableIndex
from llama_index import GPTVectorStoreIndex, SQLDatabase, SimpleKeywordTableIndex

import sqlite3

# Allow insecure transport for development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')  # Ensure secret key is set
app.permanent_session_lifetime = timedelta(minutes=30)  # Set session lifetime to 30 minutes

# Google API credentials
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
redirect_uri = os.getenv('REDIRECT_URI', 'http://localhost:5000/callback')  # Default to localhost for development

# Weather API endpoint
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

# Database connection
DATABASE_URI = "sqlite:///your_database.db"  # Update with your actual database URI
db = SQLDatabase.from_uri(DATABASE_URI)

# Initialize LlamaIndex with the SQLDatabase
db_index = SimpleKeywordTableIndex.from_documents([], sql_database=db)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

    # Initiate OAuth flow
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uris": [redirect_uri],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "scope": ["https://www.googleapis.com/auth/cloud-platform"]
            }
        },
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
        redirect_uri=redirect_uri
    )
    authorization_url, state = flow.authorization_url(prompt='consent', access_type='offline', include_granted_scopes='true')
    session['state'] = state
    return redirect(authorization_url)

@app.route('/callback')
def callback():
    if 'state' not in session:
        return jsonify({'error': 'State not found in session. Please log in again.'}), 400

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uris": [redirect_uri],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "scope": ["https://www.googleapis.com/auth/cloud-platform"]
            }
        },
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
        state=session['state'],
        redirect_uri=redirect_uri
    )
    try:
        flow.fetch_token(authorization_response=request.url)
        return 'Logged in successfully! You can now use the LLM.'
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    data = request.json
    location = data.get('location')

    if not location:
        return jsonify({'error': 'Please provide a location'})

    try:
        weather_response = requests.get(WEATHER_API_URL, params={
            'key': os.getenv('WEATHER_API_KEY'),
            'q': location
        })
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        current = weather_data['current']
        weather_info = {
            'temperature': current['temp_c'],
            'condition': current['condition']['text'],
            'humidity': current['humidity'],
            'wind_kph': current['wind_kph']
        }

        # Generate clothing recommendation (simplified logic for now)
        recommendation = f"Wear comfortable clothing for {weather_info['condition']} and {weather_info['temperature']}°C."
        return jsonify({
            'weather': weather_info,
            'recommendation': recommendation
        })

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Weather API error: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/query', methods=['POST'])
def query_database():
    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'Please provide a prompt'})

    try:
        # Query the database using LlamaIndex
        response = db_index.query(prompt)
        return jsonify({
            'response': str(response)  # Convert response to string for JSON
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)
