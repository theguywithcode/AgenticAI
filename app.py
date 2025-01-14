from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import requests
from llama_index.llms import Gemini
from llama_index import ServiceContext
from llama_index.prompts import PromptTemplate
from llama_index import VectorStoreIndex, Document
from llama_index.node_parser import SimpleNodeParser

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize LlamaIndex with Gemini
llm = Gemini(api_key=os.getenv('GOOGLE_API_KEY'))
service_context = ServiceContext.from_defaults(llm=llm)

# Weather API endpoint
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

# Create prompt template using LlamaIndex
clothing_prompt = PromptTemplate(
    template="""Given the following weather conditions, recommend appropriate clothing:
    Temperature: {temperature}°C
    Condition: {condition}
    Humidity: {humidity}%
    Wind: {wind_kph} kph
    
    Please provide a brief, practical clothing recommendation for these conditions."""
)

# Example of using more LlamaIndex features
class WeatherAdvisor:
    def __init__(self):
        # Create a knowledge base of clothing recommendations
        clothing_guidelines = [
            Document(text="In hot weather (25°C+): Light, breathable fabrics..."),
            Document(text="In cold weather (0-10°C): Layer clothing with thermal base..."),
            Document(text="In rainy conditions: Waterproof jacket, umbrella..."),
            # More guidelines...
        ]
        
        # Parse documents into nodes
        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(clothing_guidelines)
        
        # Create a vector store index
        self.index = VectorStoreIndex(nodes)
        
    def get_recommendation(self, weather_info):
        # Create a query engine
        query_engine = self.index.as_query_engine()
        
        # Query the knowledge base with current weather
        query = f"What to wear in {weather_info['temperature']}°C with {weather_info['condition']}?"
        response = query_engine.query(query)
        
        return response.response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    data = request.json
    location = data.get('location')
    
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
        
        # Generate clothing recommendation using LlamaIndex
        formatted_prompt = clothing_prompt.format(**weather_info)
        response = llm.complete(formatted_prompt)
        
        return jsonify({
            'weather': weather_info,
            'recommendation': response.text
        })
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Weather API error: {str(e)}'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 