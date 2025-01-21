#.......................IMPORTS...........................
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from fastapi import FastAPI, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import os
import requests
import json

# Enable detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate and get API keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
JWT_TOKEN = os.getenv('JWT_TOKEN')

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
if not JWT_TOKEN:
    raise ValueError("JWT_TOKEN not found in environment variables")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

class PropertyManager:
    def __init__(self):
        self.base_url = "https://api.haletale.com"
        self.headers = {
            'Authorization': f'Bearer {JWT_TOKEN}',
            'Content-Type': 'application/json'
        }
        
        # Initialize LLM
        self.llm = Gemini(
            api_key=GEMINI_API_KEY,
            model="models/gemini-1.5-flash",
            temperature=0.1,
            max_tokens=2048
        )
        
        # Store properties data
        self.properties_data = None
        self.fetch_properties()

    def fetch_properties(self) -> None:
        """Fetch and store properties from the API"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/landlord-dashboard/dropdown-properties",
                headers=self.headers
            )
            response.raise_for_status()
            self.properties_data = response.json()
            logger.info(f"Successfully fetched {len(self.properties_data)} properties")
        except Exception as e:
            logger.error(f"API fetch error: {str(e)}")
            raise

    async def query_properties(self, query: str) -> Dict:
        """Query the properties data using Gemini"""
        try:
            if not self.properties_data:
                self.fetch_properties()

            # Create context with the properties data
            context = f"""
            You are a property management assistant. Here are the properties:
            {json.dumps(self.properties_data, indent=2)}

            Answer the following question about these properties.
            Question: {query}
            """

            # Get response from Gemini
            response = self.llm.complete(context)
            
            return {
                "query": query,
                "response": str(response),
                "properties_count": len(self.properties_data)
            }
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            raise

# Initialize FastAPI
app = FastAPI(title="Property Management AI Assistant")

class QueryRequest(BaseModel):
    query: str

# Initialize PropertyManager
manager = PropertyManager()

@app.get("/")
def read_root():
    return {"status": "online", "message": "Property Management AI Assistant"}

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        return await manager.query_properties(request.query)
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))