import requests
import json

def query_rag_system(query_text: str, base_url: str = "http://localhost:8000") -> dict:
    """
    Send a query to the RAG system's FastAPI endpoint.
    
    Args:
        query_text (str): The query text to send
        base_url (str): Base URL of the FastAPI server
        
    Returns:
        dict: The response from the server
    """
    # Construct the endpoint URL
    endpoint = f"{base_url}/process-client-query"
    
    # Prepare the request payload
    payload = {
        "query": query_text
    }
    
    # Set headers for JSON content
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    try:
        # Send POST request
        response = requests.post(endpoint, json=payload, headers=headers)
        
        # Raise an exception for bad status codes
        response.raise_for_status()
        
        # Return the JSON response
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

if __name__ == "__main__":
    # Example query
    query = "Tell me about places in Asia."
    
    # Make the request
    result = query_rag_system(query)
    
    # Print the result
    if result:
        print("Response from server:")
        print(json.dumps(result, indent=2))