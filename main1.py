
#.......................DO NOT CHANGE IMPORTS...........................
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, HTTPException
from typing import List, Dict
from pydantic import BaseModel
import logging
import google.generativeai as genai

# Enable logging
logging.basicConfig(level=logging.DEBUG)


'''Need to Generate a Custom Embedding for Gemini as the GeminiEmbedding Class is no longer avaliable in llama_index.core.embeddings.gemini Module'''

'''High level understanding 
It is Converting text queries into high-dimensional numerical vectors
which in turn helps in capturing semantic meaning and context of user queries
eventually pefirms Symantic Search & Query Context Matching
'''

class GeminiCustomEmbedding(BaseEmbedding, BaseModel):
    api_key: str
    model: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_text_embedding(self, text: str) -> List[float]:
        gemini_llm = Gemini(api_key=self.api_key, model=self.model)
        return gemini_llm.embed(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._get_query_embedding(query)

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embedding = self._get_text_embedding(text)
            embeddings.append(embedding)
        return embeddings

# Initialize Gemini
API_KEY = 'AIzaSyDHYgzfdbxO6eIGBAs1iWYHN3dEFpAm5nk' #need to make secure..........
genai.configure(api_key=API_KEY)

# Set up the LLM
llm = Gemini(
    api_key=API_KEY,
    model="models/gemini-1.5-pro",
    temperature=0.1,  # Low temperature for more precise SQL generation
    max_tokens=2048
)

# Set the custom embedding model and LLM in the settings
Settings.embed_model = GeminiCustomEmbedding(
    api_key=API_KEY,
    model='models/gemini-1.5-flash'
)
Settings.llm = llm

# Initialize FastAPI app
app = FastAPI()

#Initialize Database  
DATABASE_PATH = "db.sqlite3" #Need a better Database initialixzer to connect to different databases

def get_table_schema() -> Dict[str, List[str]]:
    """Get the database schema information."""
    engine = create_engine(f"sqlite:///{DATABASE_PATH}")
    inspector = inspect(engine)
    
    schema = {}
    for table_name in inspector.get_table_names():
        columns = [column['name'] for column in inspector.get_columns(table_name)]
        schema[table_name] = columns
    
    return schema

def initialize_query_engine():
    try:
        # Set up the SQLite engine
        engine = create_engine(f"sqlite:///{DATABASE_PATH}")
        
        # Get schema information
        schema = get_table_schema()
        logging.info(f"Database schema: {schema}")
        
        # Initialize SQLDatabase with the engine
        sql_database = SQLDatabase(engine)
        
        # Initialize the NLSQLTableQueryEngine with both custom embedding model and LLM
        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=list(schema.keys()),  # Explicitly specify available tables
            embedding_model=Settings.embed_model,
            llm=Settings.llm,  # Explicitly set the LLM
            synthesize_response=True
        )
        return query_engine
    except Exception as e:
        logging.error(f"Error initializing query engine: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initializing query engine: {str(e)}")

# Initialize the query engine
query_engine = initialize_query_engine()


#defining queryrequest class according to pydantic "according to Base Embedding Model"
class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Semantic Search API using Custom Gemini Embeddings!"}


#Need to add a Token Based System "Possibally JWT or O-auth"
@app.post("/query/")
async def query_database(request: QueryRequest):
    try:
        logging.debug(f"Received query: {request.query}")
        
        # Validate if the query is not empty
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        # Get the schema for context
        schema = get_table_schema()
        
        # Add schema context to the query
        query_with_context = f"""
        Given these tables and their columns:
        {schema}
        
        Please answer: {request.query}
        """
        
        # Perform the query using the custom embedding model
        response = query_engine.query(query_with_context)
        
        # Extract SQL query from metadata
        sql_query = response.metadata.get("sql_query", "No SQL query generated")
        logging.debug(f"Generated SQL query: {sql_query}")
        
        if not response:
            return {
                "query": request.query,
                "response": "No results found",
                "sql_query": str(sql_query)
            }
        
        return {
            "query": request.query,
            "response": str(response),
            "sql_query": str(sql_query)
        }

    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")