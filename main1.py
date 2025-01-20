
#.......................DO NOT CHANGE IMPORTS...........................
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError
from fastapi import FastAPI, HTTPException
from typing import List, Dict
from pydantic import BaseModel
import logging
import google.generativeai as genai
import re
import sqlparse
from functools import wraps

# Enable logging
logging.basicConfig(level=logging.DEBUG)


'''Needed to Generate a Custom Embedding for Gemini as the GeminiEmbedding Class is no longer avaliable in llama_index.core.embeddings.gemini Module'''

'''High level understanding 
It is Converting text queries into high-dimensional numerical vectors
which in turn helps in capturing semantic meaning and context of user queries
eventually peforms Symantic Search & Query Context Matching
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


'''hahaha we need security 

I was able to delete entire table (crying in pain...)

this class basically check for any SQL injection / any actions that can peform write/delete options
'''


class SecureQueryEngine:
    """Secure wrapper for query engine with strict read-only enforcement"""
    
    FORBIDDEN_KEYWORDS = {
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE', 'TRUNCATE', 
        'RENAME', 'REPLACE', 'MERGE', 'UPSERT', 'REMOVE', 'MODIFY', 'EXEC', 
        'EXECUTE', 'PRAGMA', 'GRANT', 'REVOKE'
    }

    @staticmethod
    def is_read_only_query(sql: str) -> bool:
        """
        Thoroughly validate if a query is read-only using multiple checks
        """
        try:
            # Convert to uppercase and remove extra whitespace
            sql_normalized = ' '.join(sql.upper().split())
            
            # Parse the SQL statement
            parsed = sqlparse.parse(sql_normalized)
            if not parsed:
                return False
            
            statement = parsed[0]
            
            # Check 1: Must be a SELECT statement
            if statement.get_type() != 'SELECT':
                logging.warning(f"Query rejected - Not a SELECT statement: {sql}")
                return False
                
            # Check 2: Check for forbidden keywords
            for keyword in SecureQueryEngine.FORBIDDEN_KEYWORDS:
                if keyword in sql_normalized:
                    logging.warning(f"Query rejected - Contains forbidden keyword {keyword}: {sql}")
                    return False
            
            """I Hate you SQL Injections... """
            # Check 3: Regex pattern for common SQL injection patterns
            dangerous_patterns = [
                r';.*--',                    # Multiple statements
                r'/\*.*\*/',                 # Comment blocks
                r'xp_.*',                    # Extended stored procedures
                r'exec.*',                   # Execute statements
                r'sys\..*',                  # System table access
                r'sqlite_master',            # SQLite system table
                r'INTO\s+(?:OUTFILE|DUMPFILE)', # File operations
                r'UNION.*SELECT',            # UNION-based injections
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, sql_normalized, re.IGNORECASE):
                    logging.warning(f"Query rejected - Matches dangerous pattern {pattern}: {sql}")
                    return False
            
            # Check 4: Verify parentheses are balanced
            if sql.count('(') != sql.count(')'):
                logging.warning(f"Query rejected - Unbalanced parentheses: {sql}")
                return False
            
            # Check 5: Additional specific checks
            if 'INTO' in sql_normalized:
                logging.warning(f"Query rejected - Contains INTO keyword: {sql}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error validating query: {str(e)}")
            return False



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


"""Modify the query engine to only run read onyl / data retrival Querries"""

def create_readonly_engine(database_url: str):
    """Create a read-only database engine"""
    engine = create_engine(
        database_url,
        connect_args={
            'check_same_thread': False,  # Allow multiple threads to use the same connection
        },
        poolclass=StaticPool,  # Prevent connection pooling exploits
        isolation_level='SERIALIZABLE'  # Highest isolation level
    )
    
    """ This might need to be changed based on the Database connection...."""
    # Execute PRAGMA statements after engine creation to set read-only mode
    with engine.connect() as conn:
        conn.execute(text("PRAGMA query_only = ON"))  # SQLite 3.8.0 and later
        conn.execute(text("PRAGMA read_only = 1"))    # Older SQLite versions
        conn.commit()  # Ensure PRAGMAs are applied
    
    return engine


"""Hello Query engine... """
def initialize_query_engine():
    try:
        # Create read-only engine
        engine = create_readonly_engine(f"sqlite:///{DATABASE_PATH}")
        
        # Get schema information
        schema = get_table_schema()
        logging.info(f"Database schema: {schema}")
        
        # Initialize SQLDatabase with read-only engine
        sql_database = SQLDatabase(engine)
        
        # Initialize the NLSQLTableQueryEngine with security measures
        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=list(schema.keys()),
            embedding_model=Settings.embed_model,
            llm=Settings.llm,
            synthesize_response=True,
            sql_query_validator=SecureQueryEngine.is_read_only_query,
        )
        return query_engine
    except Exception as e:
        logging.error(f"Error initializing query engine: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initializing query engine: {str(e)}")
    



def secure_query(func):
    """Decorator to add security checks to query endpoint"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            # Get the query from the request
            request = kwargs.get('request')
            if not request or not hasattr(request, 'query'):
                raise HTTPException(status_code=400, detail="Invalid request")
            
            # Sanitize the input query
            query = request.query.strip()
            
            # Check for suspicious patterns in the natural language query
            suspicious_patterns = [
                r'delete', r'drop', r'update', r'insert', r'modify',
                r'remove', r'alter', r'create', r'truncate'
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    logging.warning(f"Suspicious natural language query detected: {query}")
                    raise HTTPException(
                        status_code=400,
                        detail="Only data retrieval queries are allowed"
                    )
            
            return await func(*args, **kwargs)
            
        except Exception as e:
            logging.error(f"Security check failed: {str(e)}")
            raise HTTPException(status_code=400, detail="Security check failed")
    
    return wrapper




# Initialize the query engine
query_engine = initialize_query_engine()


#defining queryrequest class according to pydantic "according to Base Embedding Model"
class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Semantic Search API using Custom Gemini Embeddings!"}


"""  Here we goo kinda main part.. Hahaha """
#Need to add a Token Based System "Possibally JWT or O-auth"
@app.post("/query/")
@secure_query
async def query_database(request: QueryRequest):
    try:
        logging.debug(f"Received query: {request.query}")
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        schema = get_table_schema()
        
        # Enhanced context with explicit read-only instructions
        query_with_context = f"""
        Given these tables and their columns:
        {schema}
        
        IMPORTANT RESTRICTIONS:
        1. Generate ONLY SELECT queries
        2. DO NOT generate any data modification queries (INSERT, UPDATE, DELETE, etc.)
        3. DO NOT include any system tables or administrative commands
        4. Focus only on data retrieval operations
        
        Please provide a SELECT query to answer: {request.query}
        """
        
        response = query_engine.query(query_with_context)
        sql_query = response.metadata.get("sql_query", "No SQL query generated")
        
        # Final safety check
        if not SecureQueryEngine.is_read_only_query(sql_query):
            raise HTTPException(
                status_code=400,
                detail="Invalid query detected. Only SELECT queries are allowed."
            )
        
        logging.debug(f"Validated SQL query: {sql_query}")
        
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