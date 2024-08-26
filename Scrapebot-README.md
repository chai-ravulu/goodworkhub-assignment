# goodworkhub-assignment
## Initial Comments from Chai Ravulu

I started to get my hands dirty with some run of the mill python code about scraping. I was going in a direction of having the code scrape the relevant information when you supply it relevant websites. Some of the interesting challenges I faced when developing the psuedo code are:
   - Most of the example usecases tend to be static in a sense that if you supply websites then it will scrape the relevant information. then I realized that this approach wouldn't scale in real life.
   - I refined my idea to have the code to get integrate with Search API like Bing to get the relevant websites dynamically.
   - Once we have that, I realized that all the websites will not have a standardized formating in having their grant information in their websites. We need something better and interesting to be intelligent   
     about scraping the relavant text from the websites that we hit.
   - I played around with some refinements in the code to have a Chat GPT 4 integration to extract the relevant information from the parsed website text to make our bot more intelligent. I didn't do this before 
     and this is something new and very exciting for me.
   - I tried to have the code documented and included the explanation. Please comment/reachout for any questions. 

## Objective

To enhance the web scraper by enabling it to automatically discover relevant websites offering private grants, we'll integrate a search functionality using a search API. This approach eliminates the need to manually supply URLs and allows the scraper to dynamically find and process relevant websites.

Here's a comprehensive guide, including the refined Python code, comments, and explanations.

## Project Overview

The objective is to develop an intelligent web scraper that:

1. **Automatically Identifies Relevant Websites**: Uses a search API to find websites offering private grants based on specific keywords.
2. **Extracts Grant Information**: Parses the content of these websites to extract relevant grant details.
3. **Processes Data with an LLM**: Utilizes a Large Language Model (LLM) to intelligently interpret and structure the extracted information.
4. **Stores Data**: Saves the structured data in a CSV file for further analysis.

## Flow Chart
![image](https://github.com/user-attachments/assets/d935c3b3-c8be-49b4-8066-117dee782587)


## Solution Overview

### Steps to Automatically Identify and Scrape Relevant Websites

1. **Search for Relevant Websites**:
    - **Use a Search API**: Utilize a search API (e.g., Bing Search API) to perform a search query related to private grants.
    - **Extract URLs**: Parse the search results to extract the URLs of relevant websites.

2. **Scrape and Process Data from Identified Websites**:
    - **Fetch Webpage Content**: Retrieve the HTML content of each identified website.
    - **LLM-Assisted Parsing**: Use an LLM (e.g., OpenAI's GPT-4) to intelligently parse the content and extract structured grant information.

3. **Store the Extracted Data**:
    - **Structured Storage**: Save the extracted information in a structured format (e.g., CSV) for easy access and analysis.

### Data Structure

The extracted data will be structured as follows:

```json
{
    "organization_name": "string",
    "website_url": "string",
    "grant_details": {
        "grant_name": "string",
        "description": "string",
        "eligibility": "string",
        "application_deadline": "date",
        "contact_info": "string"
    }
}
```

This structure ensures that each grant's information is neatly organized under the respective organization.

### Tools and Libraries

1. **Python**: Chosen for its simplicity and the rich ecosystem of libraries for web scraping and data processing.
2. **Requests**: To send HTTP requests to both the search API and the target websites.
3. **BeautifulSoup**: For parsing HTML content and extracting text.
4. **Pandas**: For handling and storing the extracted data in CSV format.
5. **Fake-UserAgent**: To rotate User-Agent headers, reducing the risk of being blocked by websites.
6. **OpenAI's GPT-4**: As the LLM for intelligent parsing and structuring of the scraped content.
7. **Bing Search API**: To programmatically search for relevant websites based on a query.

### Python Code

Below is the refined Python code integrating automatic website discovery using the Bing Search API and intelligent content parsing using OpenAI's GPT-4.

```python
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
```

### Explanation of the Code

1. **Search for Relevant Websites (`search_relevant_websites`)**:
    - **Purpose**: Automatically discovers websites relevant to the search query using the Bing Search API.
    - **Functionality**:
        - Sends a GET request to the Bing Search API with the specified query.
        - Parses the JSON response to extract URLs from the search results.
    - **Parameters**:
        - `query`: The search term (e.g., "organizations offering private grants").
        - `api_key`: Your Bing Search API subscription key.
        - `num_results`: Number of search results to retrieve.

2. **Fetch Webpage Content (`fetch_page_content`)**:
    - **Purpose**: Retrieves the HTML content of a given URL.
    - **Functionality**:
        - Sends a GET request to the target URL with a randomized User-Agent header to mimic different browsers.
        - Handles exceptions and non-200 HTTP status codes gracefully.
    - **Parameters**:
        - `url`: The URL of the webpage to fetch.

3. **LLM-Assisted Parsing (`llm_parse_grant_information`)**:
    - **Purpose**: Uses an LLM (e.g., GPT-4) to intelligently parse and extract structured grant information from raw HTML content.
    - **Functionality**:
        - Extracts raw text from the HTML using BeautifulSoup.
        - Constructs a prompt instructing the LLM to extract specific grant-related information and return it in JSON format.
        - Sends the prompt to the LLM and processes the response.
        - Attempts to parse the LLM's response as JSON; if unsuccessful, returns the raw text.
    - **Parameters**:
        - `html_content`: The HTML content of the webpage.

4. **Save Extracted Data (`save_to_csv`)**:
    - **Purpose**: Saves the structured grant data to a CSV file.
    - **Functionality**:
        - Converts the list of dictionaries into a Pandas DataFrame.
        - Writes the DataFrame to a CSV file.
    - **Parameters**:
        - `data`: The list of extracted grant information.
        - `filename`: The name of the CSV file to save.

5. **Main Orchestration Function (`scrape_grants_with_search_and_llm`)**:
    - **Purpose**: Coordinates the entire scraping process, from searching for websites to saving extracted data.
    - **Functionality**:
        - Sets the OpenAI API key.
        - Searches for relevant websites using the specified query and search API.
        - Iterates over the retrieved URLs, fetching and processing each one.
        - Introduces random delays between requests to respect server load and avoid detection.
        - Saves the aggregated data to a CSV file.
    - **Parameters**:
        - `search_query`: The search term to use for finding relevant websites.
        - `search_api_key`: Your Bing Search API subscription key.
        - `openai_api_key`: Your OpenAI API key.
        - `num_search_results`: Number of search results to retrieve.

### Justification for Tool Selection

1. **Python**: Its extensive library support and readability make it ideal for web scraping and data processing tasks.
2. **Requests**: A simple and efficient HTTP library for sending requests to APIs and websites.
3. **BeautifulSoup**: Provides powerful tools for parsing and navigating HTML content.
4. **Pandas**: Facilitates easy data manipulation and storage, especially for CSV files.
5. **Fake-UserAgent**: Helps in rotating User-Agent headers, reducing the risk of being blocked by websites.
6. **OpenAI's GPT-4**: Offers advanced natural language understanding, enabling intelligent parsing and structuring of unstructured web content.
7. **Bing Search API**: Allows programmatic access to search results, enabling the scraper to dynamically find relevant websites based on queries.

### Additional Considerations

- **API Keys**: Ensure you have valid API keys for both the Bing Search API and OpenAI. Replace `'your-bing-search-api-key-here'` and `'your-openai-api-key-here'` with your actual keys.
- **Rate Limiting**: Be mindful of the rate limits imposed by both the search API and OpenAI. The code includes random delays between requests to help mitigate this.
- **Error Handling**: The code includes basic error handling for HTTP requests and API interactions. Depending on your use case, you might want to implement more robust error handling and logging mechanisms.
- **LLM Response Parsing**: The LLM might not always return perfectly formatted JSON. Additional processing or prompt tuning might be necessary to improve the consistency of the responses.
