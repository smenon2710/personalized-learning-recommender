# app.py

# --- NEW IMPORTS for Google Custom Search ---
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError as GoogleApiHttpError # For catching specific Google API errors

# --- Existing Imports ---
import gradio as gr
from datetime import datetime
import os
import webbrowser
import logging
import time
import re # Already present, but needed for JSON parsing fix
import random
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import json

from anthropic import Anthropic, APIStatusError, RateLimitError as AnthropicRateLimitError
import sqlite3 # Import sqlite3 for caching

# --- 0. Configuration and Initialization ---

# Load environment variables from .env file
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it in a .env file.")

# Initialize Anthropic client
client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Define the Claude model to use (Haiku is most cost-effective for these tasks)
CLAUDE_MODEL = "claude-3-haiku-20240307" 

# Load Google API Key and CSE ID
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    raise ValueError("Google API Key or Custom Search Engine ID not found in environment variables. Please set them in a .env file.")

# Configure Logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_name = datetime.now().strftime("gradio_recommender_%Y%m%d_%H%M%S.log")
log_file_path = os.path.join(log_dir, log_file_name)

logging.basicConfig(
    level=logging.INFO, # Set overall logging level (e.g., INFO, DEBUG)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path), # Log to a file
        logging.StreamHandler() # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# --- Database Setup for Caching ---
DATABASE_NAME = 'cache_v2.db'
CACHE_EXPIRATION_DAYS = 7 # Cache entries expire after 7 days

def init_db():
    """Initializes the SQLite database and creates the cache table if it doesn't exist."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resource_cache (
            url TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            resource_type TEXT,
            llm_assessment_json TEXT, -- Store LLM assessment as JSON string
            llm_summary TEXT,
            category TEXT,          
            reasoning TEXT,         
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Database '{DATABASE_NAME}' initialized.")

init_db() # Initialize the database when the application starts

def get_cached_resource(url):
    """Retrieves a resource from cache if it's not expired."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT url, title, description, resource_type, llm_assessment_json, llm_summary, category, reasoning, timestamp FROM resource_cache WHERE url = ?", (url,))
    row = cursor.fetchone()
    conn.close()

    if row:
        url, title, description, resource_type, llm_assessment_json, llm_summary, category, reasoning, timestamp_str = row
        cached_time = datetime.fromisoformat(timestamp_str)
        if (datetime.now() - cached_time).days < CACHE_EXPIRATION_DAYS:
            logger.info(f"Cache hit for {url}. Using cached data.")
            llm_assessment = json.loads(llm_assessment_json) if llm_assessment_json else {}
            return {
                "title": title,
                "description": description,
                "url": url,
                "type": resource_type,
                "llm_assessment": llm_assessment,
                "llm_summary": llm_summary,
                "category": category,
                "reasoning": reasoning
            }
        else:
            logger.info(f"Cache expired for {url}. Will re-scrape and re-process LLM.")
    return None

def save_resource_to_cache(resource_data):
    """Saves a resource's details and LLM assessment to cache."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    llm_assessment_json = json.dumps(resource_data.get('llm_assessment', {}))

    try:
        cursor.execute("""
            INSERT OR REPLACE INTO resource_cache 
            (url, title, description, resource_type, llm_assessment_json, llm_summary, category, reasoning, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            resource_data['url'],
            resource_data['title'],
            resource_data['description'],
            resource_data.get('type', 'unknown'),
            llm_assessment_json,
            resource_data.get('llm_summary', resource_data['description']),
            resource_data.get('category', 'Other'),
            resource_data.get('reasoning', 'No specific reasoning provided.'),
            datetime.now().isoformat()
        ))
        conn.commit()
        logger.info(f"Resource for {resource_data['url']} saved/updated in cache.")
    except sqlite3.Error as e:
        logger.error(f"Error saving to cache for {resource_data['url']}: {e}", exc_info=True)
    finally:
        conn.close()

# --- 1. Core Helper Functions ---

def fetch_url_content(url, retries=3, backoff_factor=1):
    """
    Fetches URL content with retries and exponential backoff.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    for attempt in range(retries):
        try:
            logger.debug(f"Attempt {attempt + 1} to fetch: {url}")
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError as e:
            if 400 <= e.response.status_code < 500 and e.response.status_code not in [429, 408]:
                logger.error(f"Client error ({e.response.status_code}) fetching {url}: {e}. Not retrying.")
                return None
            logger.warning(f"HTTP error ({e.response.status_code}) fetching {url}: {e}. Retrying...")
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching {url}. Retrying...")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error fetching {url}. Retrying...")
        except requests.exceptions.RequestException as e:
            logger.error(f"General request error fetching {url}: {e}. Not retrying.")
            return None
        time_to_wait = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
        logger.info(f"Waiting {time_to_wait:.2f} seconds before retry...")
        time.sleep(time_to_wait)
    logger.error(f"Failed to fetch {url} after {retries} attempts.")
    return None

def perform_web_search(query, num_results=10, retries=3, backoff_factor=2):
    """
    Performs a web search using Google Custom Search JSON API
    and returns a list of result dictionaries.
    Each dictionary contains 'title', 'href' (URL), and 'body' (snippet).
    """
    logger.info(f"Initiating Google Custom Search for query: '{query}'")
    
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    
    for attempt in range(retries):
        try:
            res = service.cse().list(
                q=query,
                cx=GOOGLE_CSE_ID,
                num=num_results 
            ).execute()
            
            results = []
            for item in res.get('items', []):
                results.append({
                    'title': item.get('title'),
                    'href': item.get('link'),
                    'body': item.get('snippet')
                })
            
            logger.info(f"Found {len(results)} search results for query '{query}'.")
            return results
        except GoogleApiHttpError as e:
            logger.error(f"Google Custom Search API HTTP error (Attempt {attempt + 1}/{retries}): {e.resp.status} - {e.content.decode()}", exc_info=True)
            if e.resp.status == 429: 
                 logger.warning("Google Custom Search API Quota Exceeded or Rate Limited.")
            if attempt < retries - 1:
                time_to_wait = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Waiting {time_to_wait:.2f} seconds before retrying Google search...")
                time.sleep(time_to_wait)
            else:
                logger.error(f"Failed to perform Google search for '{query}' after {retries} attempts.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during Google search for '{query}' (Attempt {attempt + 1}/{retries}): {e}", exc_info=True)
            if attempt < retries - 1:
                time_to_wait = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Waiting {time_to_wait:.2f} seconds before retrying Google search...")
                time.sleep(time_to_wait)
            else:
                logger.error(f"Failed to perform Google search for '{query}' after {retries} attempts.")
    return []

# --- 2. Anthropic LLM Interactions ---

def call_llm_api(system_prompt, user_message, model=CLAUDE_MODEL, temperature=0.5, max_tokens=700, retries=3): # Increased max_tokens for reasoning
    """
    General function to call Anthropic API with a given prompt, including retry logic for rate limits.
    Now correctly uses the `system` parameter.
    """
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt, 
                messages=[
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
            )
            return response.content[0].text.strip() if response.content else None
        except AnthropicRateLimitError as e:
            logger.warning(f"Anthropic Rate Limit Exceeded (Attempt {attempt + 1}/{retries}): {e}")
            time_to_wait = 2 ** attempt + random.uniform(0, 1) 
            logger.info(f"Waiting {time_to_wait:.2f} seconds before retrying LLM call...")
            time.sleep(time_to_wait)
        except APIStatusError as e: 
            logger.error(f"Anthropic API Status Error (Attempt {attempt + 1}/{retries}): {e.response.status_code} - {e.response.text}", exc_info=True)
            if e.response.status_code >= 400 and e.response.status_code < 500 and e.response.status_code != 429:
                break 
            time_to_wait = 2 ** attempt + random.uniform(0, 1)
            logger.info(f"Waiting {time_to_wait:.2f} seconds before retrying LLM call...")
            time.sleep(time_to_wait)
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM API call (Attempt {attempt + 1}/{retries}): {e}", exc_info=True)
            break 
    logger.error(f"Failed to get LLM response after {retries} attempts.")
    return None

# MODIFIED llm_process_resource_metadata for new prompt (category, actionable summary, reasoning)
def llm_process_resource_metadata(learning_goal, skill_level, title, description, url):
    """
    Uses LLM to classify resource type, assess relevance/skill/quality, and provide a summary
    in a single API call for efficiency. Includes new 'category' and 'reasoning' output.
    """
    logger.debug(f"LLM processing metadata for: {title}")

    system_prompt_content = f"""You are an expert learning resource evaluator and summarizer for a recommendation system. Your primary goal is to provide highly accurate and reliable assessments and actionable summaries. You are particularly skilled at evaluating technical documentation, API references, and programming guides.

For the resource provided, perform the following tasks:
1.  **Classify its primary type** (e.g., 'video', 'article', 'hands-on project', 'course', 'documentation', 'unknown'). Be precise.
2.  **Categorize this resource** into one of these broader learning categories: 'Official Documentation/API Reference', 'Hands-on Project/Code Sample', 'In-Depth Tutorial/Guide', 'Video Walkthrough/Series', 'Online Course', 'Community/Forum Discussion', 'Other'. Choose the single best category.
3.  **Assess its relevance** to the user's learning goal on a scale of 1-5 (5 being extremely relevant, directly addressing the core of the goal; 1 being irrelevant or tangential). For API/technical queries, a 5 means it's a primary source or highly detailed, relevant guide.
4.  **Determine its target skill level** for the user's goal: 'perfect' (exactly matches), 'good' (slightly above/below but usable), 'acceptable' (needs significant prior knowledge or is too basic/advanced), 'poor' (completely misaligned). For advanced technical topics, look for deep dives, specific versions, and implementation details.
5.  **Evaluate its overall quality**: 'excellent' (comprehensive, well-structured, authoritative, up-to-date, from official/reputable source), 'good' (solid, useful, clear), 'average' (basic, potentially incomplete or less clear), 'poor' (low quality, misleading, outdated, non-authoritative). Prioritize official documentation or well-known community sites for technical topics.
6.  **Provide a concise, *actionable* 2-3 sentence summary** (max 100 words) highlighting its *key content, what a learner will gain*, and its suitability for the user's goal and skill level. For technical content, explicitly mention APIs, versions, examples, or specific functionalities if present.
7.  **Provide a brief reasoning** (1-2 sentences, max 50 words) for *why* you believe this resource is relevant, high-quality, or matches the skill level, based on its title and description. This helps users trust the recommendation.

Return your assessment as a JSON object with the following keys:
- "type": string (the classified type)
- "category": string (the chosen learning category)
- "relevance_score": integer (1-5)
- "skill_level_match": string ("perfect", "good", "acceptable", "poor")
- "quality_rating": string ("excellent", "good", "average", "poor")
- "summary": string (the 2-3 sentence actionable summary)
- "reasoning": string (the brief explanation for your assessment)

Example JSON Output for "Tableau REST API, documentation, advanced":
{{
  "type": "documentation",
  "category": "Official Documentation/API Reference",
  "relevance_score": 5,
  "skill_level_match": "perfect",
  "quality_rating": "excellent",
  "summary": "This is the official Tableau REST API reference documentation, covering all endpoints, authentication methods, and usage examples for version 3.19. It is perfectly suited for advanced users needing detailed programmatic access to Tableau Server to build robust integrations.",
  "reasoning": "This resource is from Tableau's official domain and directly addresses the REST API with versioned details, indicating high relevance and quality for an advanced user."
}}
"""

    user_message_content = f"""User Learning Goal: {learning_goal}
User Skill Level: {skill_level}

Resource Title: {title}
Resource Description: {description}
Resource URL: {url}

Assessment:"""
    
    llm_response_str = call_llm_api(system_prompt_content, user_message_content, max_tokens=700, temperature=0.5) 
    
    if llm_response_str:
        try:
            llm_assessment = json.loads(llm_response_str)
            logger.debug(f"  LLM full assessment received: {llm_assessment}")
            
            # Validate and normalize assessment values from LLM
            llm_assessment['type'] = llm_assessment.get('type', 'unknown').lower()
            llm_assessment['category'] = llm_assessment.get('category', 'Other') 
            llm_assessment['relevance_score'] = max(1, min(5, int(llm_assessment.get('relevance_score', 1))))
            llm_assessment['skill_level_match'] = llm_assessment.get('skill_level_match', 'poor').lower()
            llm_assessment['quality_rating'] = llm_assessment.get('quality_rating', 'average').lower()
            llm_assessment['summary'] = llm_assessment.get('summary', description)[:400] + "..." if len(llm_assessment.get('summary', '')) > 400 else llm_assessment.get('summary', description)
            llm_assessment['reasoning'] = llm_assessment.get('reasoning', 'No specific reasoning provided.')[:200] + "..." if len(llm_assessment.get('reasoning', '')) > 200 else llm_assessment.get('reasoning', 'No specific reasoning provided.') 
            
            return llm_assessment
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM assessment JSON: {e}. Raw response: {llm_response_str}", exc_info=True)
            return None
        except ValueError as e: 
            logger.error(f"Failed to convert LLM assessment values: {e}. Raw response: {llm_response_str}", exc_info=True)
            return None
    return None

# NEW: LLM Reranker for initial search results (UPDATED for robust JSON parsing and prompt)
def llm_rerank_search_results(user_goal, user_style, user_level, search_results):
    """
    Uses LLM to quickly assess the relevance of raw search results (title, snippet, URL)
    and return a filtered list of promising URLs, intended to be very cheap.
    """
    logger.info(f"LLM reranking initial {len(search_results)} search results.")
    
    # NEW: Make the system prompt even more explicit about JSON ONLY and strictness
    system_prompt = f"""You are a highly efficient and strict search result filter. Your only task is to assess the relevance of provided search results.
    For each search result (title, snippet, URL) given the user's learning goal, determine if it is HIGHLY_RELEVANT or NOT_RELEVANT.
    Focus exclusively on the title and snippet provided. Be very strict: if a result is not directly and clearly related to the learning goal, it is NOT_RELEVANT.

    You MUST return your judgment as a JSON array of objects. Do NOT include any conversational text, introductions, or explanations outside the JSON.
    Each object in the array MUST have:
    - "url": The exact URL of the resource.
    - "relevance": "HIGHLY_RELEVANT" or "NOT_RELEVANT".

    Example JSON Output:
    [
        {{ "url": "https://example.com/highly-relevant", "relevance": "HIGHLY_RELEVANT" }},
        {{ "url": "https://example.com/unrelated-blog", "relevance": "NOT_RELEVANT" }}
    ]
    """
    
    reranked_results = []
    chunk_size = 5 # Process 5 results at a time with the LLM to manage prompt length
    
    for i in range(0, len(search_results), chunk_size):
        chunk = search_results[i:i+chunk_size]
        
        user_messages_for_chunk = []
        for res in chunk:
            user_messages_for_chunk.append(f"""
            ---
            Search Result Title: {res.get('title', 'N/A')}
            Search Result Snippet: {res.get('body', 'N/A')}
            Search Result URL: {res.get('href', 'N/A')}
            ---
            """)
        
        combined_user_message = f"""User Learning Goal: {user_goal}
User Style: {user_style}
User Level: {user_level}

Evaluate the following search results and return a JSON array of judgments based on the instructions in your system prompt. Ensure the JSON is complete and valid:""" + "\n".join(user_messages_for_chunk)
        
        # Increased max_tokens for reranker to allow JSON to complete reliably
        # Target ~100 tokens per judgment object * chunk_size + buffer
        llm_response_str = call_llm_api(system_prompt, combined_user_message, model=CLAUDE_MODEL, max_tokens=chunk_size * 100, temperature=0.1) 

        if llm_response_str:
            # UPDATED JSON Extraction: Find the first and last bracket to isolate the JSON array robustly
            json_start = llm_response_str.find('[')
            json_end = llm_response_str.rfind(']')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_portion = llm_response_str[json_start : json_end + 1]
                try:
                    judgments = json.loads(json_portion)
                    if isinstance(judgments, list):
                        for judgment in judgments:
                            if isinstance(judgment, dict) and judgment.get('relevance') == "HIGHLY_RELEVANT" and judgment.get('url'):
                                # Find the original result object from the chunk to preserve all its data
                                original_result = next((r for r in chunk if r.get('href') == judgment['url']), None)
                                if original_result:
                                    reranked_results.append(original_result)
                            else:
                                logger.debug(f"Skipping malformed or non-relevant judgment: {judgment}")
                    else:
                        logger.warning(f"LLM rerank response parsed but not a list: {judgments}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM rerank JSON (after slice): {e}. Raw: {json_portion}", exc_info=True)
            else:
                logger.warning(f"LLM response did not contain a valid JSON list structure. Raw: {llm_response_str}")
        
        time.sleep(random.uniform(0.5, 1.5)) # Small delay between LLM reranking chunks
        
    logger.info(f"LLM reranking filtered {len(search_results)} to {len(reranked_results)} promising results.")
    return reranked_results


# --- 3. Resource Extraction and Curation Logic ---

def extract_resource_details(url):
    """
    Fetches the content of a URL and attempts to extract a title and a description.
    Returns a dictionary with 'title', 'description', 'url'.
    LLM will classify type and assess quality later.
    """
    logger.info(f"Attempting to extract raw details from: {url}")
    html_content = fetch_url_content(url)
    if not html_content:
        logger.warning(f"Skipping {url} due to failed content fetch.")
        return None

    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        title = soup.find('title').get_text(strip=True) if soup.find('title') else "No Title Found"
        if title == "No Title Found":
            logger.debug(f"No title tag found for {url}")

        description = "No Description Found"
        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description:
            description = meta_description.get('content', '').strip()
        else:
            paragraphs = soup.find_all('p')
            if paragraphs:
                full_text = ' '.join([p.get_text(strip=True) for p in paragraphs[:5]])
                description = full_text[:1000] + "..." if len(full_text) > 1000 else full_text
                description = description.replace('\n', ' ').replace('\r', '').strip()
        
        if description == "No Description Found":
             logger.debug(f"No meta description or paragraphs found for {url}")

        logger.info(f"  Extracted raw -> Title: '{title[:50]}...'")

        return {
            "title": title,
            "description": description, 
            "url": url,
        }

    except Exception as e:
        logger.error(f"An error occurred during parsing {url}: {e}", exc_info=True)
        return None

def curate_resources(resources, preferences):
    """
    Processes resources, calls LLM for assessment/type/summary/reasoning, then filters and ranks.
    Incorporates caching for LLM-derived data.
    """
    logger.info("Starting resource curation with LLM assessments.")
    curated_list = []
    
    for resource in resources:
        if not resource:
            continue
        
        # --- Caching Check for LLM Data ---
        cached_data = get_cached_resource(resource['url'])
        
        if cached_data:
            resource['type'] = cached_data['type']
            resource['llm_assessment'] = cached_data['llm_assessment']
            resource['llm_summary'] = cached_data['llm_summary']
            resource['category'] = cached_data.get('category', 'Other')
            resource['reasoning'] = cached_data.get('reasoning', 'No specific reasoning provided.')
            logger.debug(f"Used cached LLM data for {resource.get('title', 'N/A')}")
        else:
            logger.info(f"Calling LLM for {resource.get('title', 'N/A')}")
            llm_data = llm_process_resource_metadata(
                preferences['learning_goal'],
                preferences['skill_level'],
                resource['title'],
                resource['description'], 
                resource['url']
            )
            
            if llm_data:
                resource['type'] = llm_data.get('type', 'unknown')
                resource['category'] = llm_data.get('category', 'Other') 
                resource['llm_assessment'] = {
                    "relevance_score": llm_data.get('relevance_score', 1),
                    "skill_level_match": llm_data.get('skill_level_match', 'poor'),
                    "quality_rating": llm_data.get('quality_rating', 'average')
                }
                resource['llm_summary'] = llm_data.get('summary', resource['description'])
                resource['reasoning'] = llm_data.get('reasoning', 'No specific reasoning provided.')
                save_resource_to_cache(resource)
            else:
                logger.warning(f"LLM processing failed for {resource.get('title', 'N/A')}. Using fallbacks for LLM data.")
                resource['type'] = "unknown"
                resource['category'] = "Other" 
                resource['llm_assessment'] = {
                    "relevance_score": 1, "skill_level_match": "poor", "quality_rating": "poor"
                }
                resource['llm_summary'] = resource['description']
                resource['reasoning'] = 'LLM assessment failed, using fallback.'


        # --- Calculate Curation Score based heavily on LLM data ---
        resource_score = 0

        resource_score += resource['llm_assessment']['relevance_score'] * 20 

        skill_match_mapping = {"perfect": 15, "good": 7, "acceptable": 3, "poor": -15}
        resource_score += skill_match_mapping.get(resource['llm_assessment']['skill_level_match'], 0)

        quality_mapping = {"excellent": 15, "good": 7, "average": 3, "poor": -10} 
        resource_score += quality_mapping.get(resource['llm_assessment']['quality_rating'], 0)

        if preferences['learning_style'] != 'mixed':
            if resource['type'] == preferences['learning_style']:
                resource_score += 3 
            else:
                resource_score -= 2 
        else:
            if resource['type'] != 'unknown':
                resource_score += 1 

        if "no description found" in resource.get('description', '').lower() or \
           "no title found" in resource.get('title', '').lower():
            resource_score -= 5
        
        resource_score = max(0, resource_score)
        resource['curation_score'] = resource_score
        curated_list.append(resource)

    filtered_list = [res for res in curated_list if res['curation_score'] >= 30] 

    filtered_list.sort(key=lambda x: x.get('curation_score', 0), reverse=True)
    
    logger.info(f"Finished curation. Found {len(filtered_list)} relevant resources.")
    return filtered_list[:7]

# --- 4. Report Generation (UPDATED for Categorization and UI styling) ---

def generate_html_report(resources, preferences):
    """
    Generates a simple HTML report of the curated resources.
    Groups resources by category and presents them, including LLM reasoning.
    """
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    
    report_filename = f"learning_resources_{preferences['learning_goal'].replace(' ', '_').replace('/', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    file_path = os.path.join(output_dir, report_filename)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Personalized Learning Resources for {preferences['learning_goal']}</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Roboto', sans-serif; line-height: 1.7; color: #333; margin: 0; background-color: #eef1f5; }}
            .header {{ background-color: #2c3e50; padding: 25px 0; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }}
            .header h1 {{ margin: 0; font-size: 2.8em; color: #FFFFFF; }}
            .header p {{ margin: 0; font-size: 1.2em; color: #E0E0E0; }}
            .container {{ max-width: 960px; margin: 30px auto; background: #fff; padding: 30px 40px; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
            h2 {{ color: #2c3e50; border-bottom: 2px solid #e0e0e0; padding-bottom: 12px; margin-bottom: 25px; font-size: 1.8em; }}
            .preferences p {{ margin: 8px 0; font-size: 1.05em; line-height: 1.5; }}
            .preferences strong {{ color: #34495e; }}
            .resource {{ background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin-bottom: 25px; transition: transform 0.2s ease, box-shadow 0.2s ease; }}
            .resource:hover {{ transform: translateY(-5px); box-shadow: 0 8px 16px rgba(0,0,0,0.15); }}
            .resource h3 {{ margin-top: 0; color: #3498db; font-size: 1.5em; }}
            .resource a {{ color: #3498db; text-decoration: none; font-weight: 500; }}
            .resource a:hover {{ text-decoration: underline; }}
            .resource p {{ font-size: 0.95em; color: #555; margin-bottom: 10px; }}
            .resource .meta {{ display: flex; flex-wrap: wrap; margin-bottom: 15px; }}
            .resource .meta span {{ background-color: #ecf0f1; color: #555; padding: 5px 12px; border-radius: 20px; font-size: 0.85em; margin-right: 10px; margin-bottom: 8px; font-weight: 500; }}
            .resource span.type {{ background-color: #d4edda; color: #155724; }}
            .resource span.score {{ background-color: #cce5ff; color: #004085; }}
            .llm-relevance, .llm-skill, .llm-quality {{ background-color: #f0e6fa; color: #6a0dad; }}
            .resource .reasoning {{ font-style: italic; font-size: 0.85em; color: #777; margin-top: 10px; border-left: 3px solid #ccc; padding-left: 10px; }}
            .category-heading {{ color: #2c3e50; font-size: 1.6em; margin-top: 40px; margin-bottom: 15px; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
            .category-separator {{ border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.25), rgba(0, 0, 0, 0)); margin-top: 15px; margin-bottom: 25px; }}

            .footer {{ text-align: center; margin-top: 50px; padding: 20px; font-size: 0.85em; color: #777; border-top: 1px solid #eee; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Personalized Learning Resources</h1>
            <p>Your AI-powered guide to knowledge.</p>
        </div>
        <div class="container">
            <h2>Your Learning Quest:</h2>
            <div class="preferences">
                <p><strong>Goal:</strong> {preferences['learning_goal']}</p>
                <p><strong>Style:</strong> {preferences['learning_style'].title()}</p>
                <p><strong>Level:</strong> {preferences['skill_level'].title()}</p>
            </div>

            <h2>Top Recommended Resources:</h2>
    """
    # Group resources by category
    categorized_resources = {}
    for res in resources:
        category = res.get('category', 'Other')
        if category not in categorized_resources:
            categorized_resources[category] = []
        categorized_resources[category].append(res)
    
    # Define a desired order for categories
    category_order = [
        'Official Documentation/API Reference',
        'Hands-on Project/Code Sample',
        'In-Depth Tutorial/Guide',
        'Video Walkthrough/Series',
        'Online Course',
        'Community/Forum Discussion',
        'Other'
    ]

    if not resources:
        html_content += "<p>No highly relevant resources found for your query after careful curation. Please try a different learning goal or broaden your preferences!</p>"
        logger.info("No resources to display in HTML report after curation.")
    else:
        # Iterate through categories in desired order
        for category_name in category_order:
            if category_name in categorized_resources and categorized_resources[category_name]:
                html_content += f"<h3 class='category-heading'>{category_name}</h3>"

                for i, res in enumerate(categorized_resources[category_name]):
                    display_description = res.get('llm_summary') or res.get('description', 'N/A')
                    display_reasoning = res.get('reasoning', 'No specific reasoning provided.')

                    html_content += f"""
                    <div class="resource">
                        <h3><a href="{res.get('url', '#')}" target="_blank">{res.get('title', 'N/A')}</a></h3>
                        <div class="meta">
                            <span class="type">{res.get('type', 'N/A').title()}</span>
                            <span class="score">Curation Score: {res.get('curation_score', 'N/A')}</span>
                            <span class="llm-relevance">Relevance: {res.get('llm_assessment', {}).get('relevance_score', 'N/A')}/5</span>
                            <span class="llm-skill">Skill Match: {res.get('llm_assessment', {}).get('skill_level_match', 'N/A').title()}</span>
                            <span class="llm-quality">Quality: {res.get('llm_assessment', {}).get('quality_rating', 'N/A').title()}</span>
                        </div>
                        <p>{display_description}</p>
                        <p class="reasoning">**AI Reasoning:** {display_reasoning}</p>
                    </div>
                    """
    
    html_content += """
        </div>
        <div class="footer">
            <p>© 2025 Agentic AI Learning Recommender. All rights reserved.</p>
            <p>Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
    </body>
    </html>
    """

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated at: {os.path.abspath(file_path)}")
    return file_path

# --- 5. Gradio Interface Integration ---

def run_recommender(learning_goal: str, learning_style: str, skill_level: str) -> str:
    """
    Main function to run the agentic learning resource recommender,
    designed to be called by Gradio.
    Returns the HTML content of the generated report.
    """
    logger.info(f"Gradio function called with: Goal='{learning_goal}', Style='{learning_style}', Level='{skill_level}'")

    if not learning_goal.strip():
        logger.warning("Learning goal not provided by user.")
        return "<p>Please provide a learning goal to get recommendations.</p>"
    
    # Standardize inputs
    learning_style = learning_style.strip().lower()
    skill_level = skill_level.strip().lower()

    user_prefs = {
        "learning_goal": learning_goal.strip(),
        "learning_style": learning_style,
        "skill_level": skill_level
    }

    # --- Phase 2: Advanced Search Query Generation with LLMs ---
    search_query_system_prompt = """You are a helpful assistant that generates highly effective search queries for learning resources, specifically optimized for a Google-like search engine.
Given a user's learning goal, preferred style, and skill level, generate 3-5 diverse and highly relevant search queries.

**For highly technical or API-related queries, prioritize terms like 'official documentation', 'API reference', 'developer guide', 'SDK', 'examples', 'tutorial'.**
Aim for variety, including terms like 'tutorial', 'guide', 'course', 'project', 'documentation', 'best practices', 'examples', 'troubleshooting', 'reference'.
Use double quotes for exact phrases where appropriate (e.g., "Tableau REST API").
Avoid conversational filler in the queries.
Return the queries as a JSON list of strings.

Example for 'Tableau REST API, documentation, advanced':
["Tableau REST API official documentation", "Tableau Server REST API reference advanced", "Tableau API developer guide examples", "Tableau REST API tutorial advanced", "Tableau SDK documentation"]

Example for 'Python data science tutorial for beginners':
["Python data science tutorial for beginners", "introduction to data science with Python course", "beginner data science projects Python"]
"""
    search_query_user_message = f"""User Learning Goal: {user_prefs['learning_goal']}
User Preferred Style: {user_prefs['learning_style']}
User Skill Level: {user_prefs['skill_level']}

Generate search queries:"""

    llm_generated_queries_str = call_llm_api(search_query_system_prompt, search_query_user_message, max_tokens=200, temperature=0.7)
    
    generated_queries = []
    if llm_generated_queries_str:
        # Use regex to find the JSON array part robustly (Fix for LLM conversational filler)
        json_match = re.search(r'\[\s*".*?"(?:,\s*".*?")*\s*\]', llm_generated_queries_str, re.DOTALL)
        if json_match:
            json_portion = json_match.group(0)
            try:
                generated_queries = json.loads(json_portion)
                if not isinstance(generated_queries, list):
                    logger.warning(f"LLM generated non-list queries: {generated_queries}. Falling back to default.")
                    generated_queries = []
                else:
                    logger.info(f"LLM generated queries: {generated_queries}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM generated queries JSON (after regex): {e}. Raw: {json_portion}", exc_info=True)
                generated_queries = []
        else:
            logger.warning(f"LLM response did not contain a valid JSON list. Raw: {llm_generated_queries_str}")
            generated_queries = []


    # Fallback to a default query if LLM fails or returns empty
    if not generated_queries:
        logger.warning("LLM query generation failed or returned empty. Using fallback query.")
        base_query = f"{user_prefs['learning_goal']} {user_prefs['learning_style']} {user_prefs['skill_level']} tutorial"
        if user_prefs['learning_style'] == 'video':
            base_query += " youtube course"
        elif user_prefs['learning_style'] == 'hands-on projects':
            base_query += " github project tutorial example"
        elif user_prefs['learning_style'] == 'articles':
            base_query += " blog guide medium"
        elif user_prefs['learning_style'] == 'documentation':
            base_query += " official docs reference"
        generated_queries = [base_query]
        logger.info(f"Using fallback query: {generated_queries[0]}")

    all_search_results = []
    for query in generated_queries:
        results = perform_web_search(query, num_results=10) # Get 10 results per query (max for Google CSE)
        all_search_results.extend(results)
    
    # Remove duplicate URLs from search results to avoid redundant scraping/LLM calls
    unique_search_results = []
    seen_urls = set()
    for res in all_search_results:
        url = res.get('href')
        if url and url not in seen_urls:
            unique_search_results.append(res)
            seen_urls.add(url)
    
    logger.info(f"Collected {len(unique_search_results)} unique search results after advanced query generation.")

    if not unique_search_results:
        logger.warning("No search results found after advanced query generation. Cannot proceed with reranking.")
        return "<p>No initial search results found for your query. Please check your internet connection or try a different learning goal.</p>"

    # --- NEW STEP: LLM-Powered Initial Filtering/Reranking ---
    filtered_promising_results = llm_rerank_search_results(
        user_prefs['learning_goal'],
        user_prefs['learning_style'],
        user_prefs['skill_level'],
        unique_search_results # Pass all unique search results for reranking
    )
    
    logger.info(f"After LLM reranking, {len(filtered_promising_results)} promising results remain.")

    if not filtered_promising_results:
        logger.warning("No highly promising resources found after intelligent filtering.")
        return "<p>No highly promising resources found after intelligent filtering. Try a different query or broader preferences!</p>"

    logger.info("Starting detailed resource extraction and LLM processing for promising links.")
    collected_resources = []
    # Cap detailed scraping at a reasonable number (e.g., top 10 promising unique results)
    max_scrape = min(len(filtered_promising_results), 10) 
    
    for i, result in enumerate(filtered_promising_results[:max_scrape]):
        url = result.get('href')
        if url and url.startswith("http"):
            # --- Caching Check for full Resource Data (including LLM output) ---
            cached_resource = get_cached_resource(url)
            if cached_resource:
                collected_resources.append(cached_resource)
                logger.debug(f"Using cached data for URL: {url}")
            else:
                resource_details = extract_resource_details(url)
                if resource_details:
                    collected_resources.append(resource_details)
                time.sleep(random.uniform(0.7, 2.5)) 
        else:
            logger.debug(f"Skipping invalid URL after rerank: {url}")
    
    if not collected_resources:
        logger.warning("No resources could be extracted after detailed analysis of promising links.")
        return "<p>No resources could be extracted after detailed analysis of promising links. This might be due to website blocking or content structure.</p>"

    final_curated_resources = curate_resources(collected_resources, user_prefs)

    html_file_path = generate_html_report(final_curated_resources, user_prefs)
    
    with open(html_file_path, "r", encoding="utf-8") as f:
        html_content_to_display = f.read()
    
    logger.info("Successfully generated and returning HTML content.")
    return html_content_to_display


# --- 6. Gradio Interface Definition ---

iface = gr.Interface(
    fn=run_recommender,
    inputs=[
        gr.Textbox(label="What do you want to learn?", placeholder="e.g., Python for Data Science, React hooks, Kubernetes"),
        gr.Radio(
            ["video", "articles", "hands-on projects", "course", "documentation", "mixed"],
            label="Preferred Learning Style",
            value="mixed"
        ),
        gr.Dropdown(
            ["beginner", "intermediate", "advanced"],
            label="Current Skill Level",
            value="beginner"
        )
    ],
    outputs=gr.HTML(label="Your Personalized Learning Resources"), # Gradio HTML component to display our report
    title="📚 Personalized Learning AI Recommender (Powered by Anthropic Claude)",
    description=(
        "Enter your learning goal and preferences, and let the AI find the best resources for you!<br>"
        "This agent autonomously searches the web, extracts details, and uses Anthropic Claude for enhanced accuracy and efficiency."
    ),
    #allow_flagging=False, # Disable Gradio's default "Flag" button
    examples=[
        ["Generative AI basics", "video", "beginner"],
        ["Fine-tuning LLMs with PyTorch", "documentation", "advanced"],
        ["Building a Full Stack App with Next.js", "hands-on projects", "intermediate"],
        ["Deep Reinforcement Learning", "articles", "advanced"],
        ["Data Structures and Algorithms", "course", "intermediate"]
    ]
)

# --- 7. Launch Application ---

if __name__ == "__main__":
    # Initialize the database when the application starts
    #init_db() 
    logger.info("Launching Gradio application.")
    iface.launch(share=True, inbrowser=True)