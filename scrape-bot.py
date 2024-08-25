import requests
from bs4 import BeautifulSoup
import pandas as pd
from fake_useragent import UserAgent
import time
import random
import openai
import json

# Function to search for relevant websites using Bing Search API
def search_relevant_websites(query, api_key, num_results=10):
    """
    Uses Bing Search API to find relevant websites based on the query.
    
    Args:
        query (str): The search query string.
        api_key (str): The subscription key for Bing Search API.
        num_results (int): Number of search results to retrieve.
    
    Returns:
        list: A list of URLs extracted from the search results.
    """
    search_url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "textDecorations": True, "textFormat": "HTML", "count": num_results}
    
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    
    urls = []
    # Extract URLs from the search results
    if 'webPages' in search_results:
        for item in search_results['webPages']['value']:
            urls.append(item['url'])
    
    return urls

# Function to fetch HTML content of a webpage
def fetch_page_content(url):
    """
    Fetches the HTML content of a given URL.
    
    Args:
        url (str): The URL of the webpage to fetch.
    
    Returns:
        str or None: The HTML content if the request is successful; otherwise, None.
    """
    headers = {'User-Agent': UserAgent().random}  # Rotate User-Agent
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch {url}: Status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

# Function to parse and extract grant information using LLM
def llm_parse_grant_information(html_content):
    """
    Uses an LLM to parse HTML content and extract structured grant information.
    
    Args:
        html_content (str): The HTML content of the webpage.
    
    Returns:
        dict or None: A dictionary containing extracted grant information; otherwise, None.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract raw text from the HTML
    raw_text = soup.get_text(separator=' ', strip=True)
    
    # Prepare the prompt for the LLM
    prompt = f"""
    You are an expert in analyzing grant-related information from text. Given the following text from a website:

    {raw_text}

    Please extract and summarize the following information in a structured JSON format:
    1. Organization Name
    2. Grant Details:
        a. Grant Name
        b. Description
        c. Eligibility Criteria
        d. Application Deadline
        e. Contact Information
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing grant-related information from text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        grant_information = response.choices[0].message['content']
        # Attempt to parse the JSON
        try:
            grant_info_json = json.loads(grant_information)
        except json.JSONDecodeError:
            # If LLM didn't return valid JSON, return the raw text
            grant_info_json = {"grant_information": grant_information}
        return grant_info_json
    except Exception as e:
        print(f"Error with LLM processing: {e}")
        return None

# Function to save extracted data to a CSV file
def save_to_csv(data, filename='grants_with_llm.csv'):
    """
    Saves the extracted grant data to a CSV file.
    
    Args:
        data (list): A list of dictionaries containing grant information.
        filename (str): The name of the CSV file to save.
    """
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Main function to orchestrate the scraping process
def scrape_grants_with_search_and_llm(search_query, search_api_key, openai_api_key, num_search_results=10):
    """
    Orchestrates the process of searching for websites, scraping content, processing with LLM, and saving data.
    
    Args:
        search_query (str): The search query string.
        search_api_key (str): The subscription key for Bing Search API.
        openai_api_key (str): The API key for OpenAI.
        num_search_results (int): Number of search results to retrieve.
    """
    openai.api_key = openai_api_key  # Set OpenAI API key
    # Step 1: Search for relevant websites
    try:
        urls = search_relevant_websites(search_query, search_api_key, num_search_results)
        print(f"Found {len(urls)} URLs from search.")
    except Exception as e:
        print(f"Error during search: {e}")
        urls = []
    
    all_data = []
    # Step 2: Scrape each URL
    for url in urls:
        print(f"Processing URL: {url}")
        html_content = fetch_page_content(url)
        if html_content:
            grant_data = llm_parse_grant_information(html_content)
            if grant_data:
                # Optionally, add the URL to the data
                grant_data['website_url'] = url
                all_data.append(grant_data)
        # Respectful crawling: random delay
        time.sleep(random.uniform(1, 3))
    
    # Step 3: Save the data
    save_to_csv(all_data)

# Example usage
if __name__ == "__main__":
    # Define your search query
    search_query = "organizations offering private grants"
    
    # Define your API keys
    search_api_key = 'your-bing-search-api-key-here'  # Replace with your Bing Search API key
    openai_api_key = 'your-openai-api-key-here'      # Replace with your OpenAI API key
    
    # Scrape grants
    scrape_grants_with_search_and_llm(search_query, search_api_key, openai_api_key, num_search_results=20)
