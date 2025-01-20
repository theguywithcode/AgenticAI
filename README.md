This is our LLM code for creating a conversational AI


## 🏗️  System Architecture 
![System Arcitecture](./System_Architecture.svg)



## Process Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant SecurityLayer
    participant QueryEngine
    participant GeminiLLM
    participant CustomEmbedding
    participant Database

    Client->>FastAPI: POST /query/ with natural language query
    FastAPI->>SecurityLayer: Apply @secure_query decorator
    SecurityLayer->>SecurityLayer: Check for suspicious patterns
    SecurityLayer->>QueryEngine: Pass sanitized query
    
    QueryEngine->>CustomEmbedding: Generate query embeddings
    CustomEmbedding->>GeminiLLM: Convert text to embeddings
    
    QueryEngine->>GeminiLLM: Generate SQL with context
    GeminiLLM->>QueryEngine: Return SQL query
    
    QueryEngine->>SecurityLayer: Validate SQL (read-only)
    SecurityLayer->>Database: Execute safe SELECT query
    Database->>FastAPI: Return results
    FastAPI->>Client: JSON response with results

```





## Developer Notes
    Updated Requirements.txt

    Do Not Change this README and IMPORT Statments from the main app 
    They are written in a way to not cause Import issue
    user python -m uvicorn main1:app --reload to run the app

    Query Body --->
    query:{
        "Write your prompt here"
    }


```
(⌐■_■)   < I code, therefore I am! >
     ╭───────────────╮
    / theguywithcode  \
   |    /////\\\\\\    |
   |   ///  o  o \\\   |
    \  ||    ∆    ||  /
     '--\   ---   /--'
         |_______|
        //       \\
       //         \\
     _//           \\_
    (__)           (__)


```