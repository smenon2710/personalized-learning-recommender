# main.py (Final Stage - incorporating logging and other refinements)

from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import time
import re
import random
import os
import webbrowser
import logging # Import the logging module
from datetime import datetime # For unique log file names and reports

# --- 1. Configure Logging ---
# Create a logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_name = datetime.now().strftime("recommender_%Y%m%d_%H%M%S.log")
log_file_path = os.path.join(log_dir, log_file_name)

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO, # Set overall logging level (e.g., INFO, DEBUG)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path), # Log to a file
        logging.StreamHandler() # Also log to console
    ]
)
logger = logging.getLogger(__name__) # Get a logger for this module

# --- Helper function for robust HTTP requests with retries ---
def fetch_url_content(url, retries=3, backoff_factor=1):
    """
    Fetches URL content with retries and exponential backoff.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    for attempt in range(retries):
        try:
            logger.debug(f"Attempt {attempt + 1} to fetch: {url}")
            response = requests.get(url, headers=headers, timeout=15) # Increased timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.text
        except requests.exceptions.HTTPError as e:
            if 400 <= e.response.status_code < 500 and e.response.status_code not in [429, 408]:
                logger.error(f"Client error ({e.response.status_code}) fetching {url}: {e}. Not retrying.")
                return None # Don't retry for client errors unless specific ones
            logger.warning(f"HTTP error ({e.response.status_code}) fetching {url}: {e}. Retrying...")
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching {url}. Retrying...")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error fetching {url}. Retrying...")
        except requests.exceptions.RequestException as e:
            logger.error(f"General request error fetching {url}: {e}. Not retrying.")
            return None # Other general request errors, don't retry

        time_to_wait = backoff_factor * (2 ** attempt) + random.uniform(0, 1) # Exponential backoff with jitter
        logger.info(f"Waiting {time_to_wait:.2f} seconds before retry...")
        time.sleep(time_to_wait)
    logger.error(f"Failed to fetch {url} after {retries} attempts.")
    return None


def get_user_learning_preferences():
    """
    Prompts the user for their learning goal and preferences,
    and returns them as a dictionary.
    """
    logger.info("Starting user preference collection.")
    print("Welcome to your Personalized Learning Resource Recommender!")
    print("-" * 50)

    learning_goal = input("What do you want to learn? (e.g., 'Python for Data Science', 'React hooks', 'Machine Learning fundamentals'):\n> ")
    if not learning_goal.strip():
        logger.warning("Learning goal not provided, setting a default.")
        learning_goal = "Python programming" # Default if empty

    learning_style = input("\nWhat's your preferred learning style? (e.g., 'video', 'articles', 'hands-on projects', 'mixed'):\n> ").lower()
    if learning_style not in ['video', 'articles', 'hands-on projects', 'mixed']:
        logger.warning(f"Invalid learning style '{learning_style}'. Defaulting to 'mixed'.")
        learning_style = 'mixed'
    
    skill_level = input("\nWhat's your current skill level for this topic? (e.g., 'beginner', 'intermediate', 'advanced'):\n> ").lower()
    if skill_level not in ['beginner', 'intermediate', 'advanced']:
        logger.warning(f"Invalid skill level '{skill_level}'. Defaulting to 'beginner'.")
        skill_level = 'beginner'

    preferences = {
        "learning_goal": learning_goal.strip(),
        "learning_style": learning_style.strip().lower(),
        "skill_level": skill_level.strip().lower()
    }
    logger.info(f"User preferences collected: {preferences}")
    print("\nGot it! Let's find some resources for you.")
    print("-" * 50)
    return preferences

def perform_web_search(query, num_results=20):
    """
    Performs a web search using DuckDuckGo and returns a list of result dictionaries.
    Each dictionary contains 'title', 'href' (URL), and 'body' (snippet).
    """
    logger.info(f"Initiating web search for query: '{query}'")
    print(f"Searching the web for: '{query}'...")
    try:
        results = DDGS().text(keywords=query, max_results=num_results)
        logger.info(f"Found {len(results)} search results.")
        return results
    except Exception as e:
        logger.error(f"An error occurred during web search for '{query}': {e}", exc_info=True)
        return []

def extract_resource_details(url):
    """
    Fetches the content of a URL and attempts to extract a title and a description.
    Also attempts to detect resource type and calculate a basic quality score.
    Returns a dictionary with 'title', 'description', 'url', 'type', 'quality_score'.
    """
    logger.info(f"Attempting to extract details from: {url}")
    print(f"Attempting to extract details from: {url}")

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
                full_text = ' '.join([p.get_text(strip=True) for p in paragraphs[:3]])
                description = full_text[:300] + "..." if len(full_text) > 300 else full_text
                description = description.replace('\n', ' ').replace('\r', '').strip()
        
        if description == "No Description Found":
             logger.debug(f"No meta description or paragraphs found for {url}")


        resource_type = "article" # Default
        if "youtube.com/watch" in url or "vimeo.com" in url: # More robust video check
            resource_type = "video"
        elif any(domain in url for domain in ["github.com", "codepen.io", "repl.it", "jupyter.org"]):
            resource_type = "hands-on projects"
        elif any(tag in url for tag in ["blog", "article", "tutorial", "guide", "medium.com"]):
            resource_type = "article"
        elif any(domain in url for domain in ["udemy.com", "coursera.org", "edx.org", "khanacademy.org", "pluralsight.com"]):
             resource_type = "course"
        elif "docs." in url or "documentation" in url:
             resource_type = "documentation"
        else:
            logger.debug(f"Could not definitively determine type for {url}, defaulting to article.")


        quality_score = 0
        text_content = (title + " " + description).lower()
        if any(keyword in text_content for keyword in ["official", "documentation", "guide", "tutorial", "best practices", "deep dive", "comprehensive", "free", "beginner", "introduction"]):
            quality_score += 1
        if any(domain in url for domain in ["udemy.com", "coursera.org", "edx.org", "freecodecamp.org", "w3schools.com", "developer.mozilla.org", "learn.microsoft.com", "developers.google.com", "python.org"]):
            quality_score += 2 # Higher score for known reputable domains
        
        logger.info(f"  Extracted -> Title: '{title[:50]}...', Type: {resource_type}, Quality Score: {quality_score}")

        return {
            "title": title,
            "description": description,
            "url": url,
            "type": resource_type,
            "quality_score": quality_score
        }

    except Exception as e:
        logger.error(f"An error occurred during parsing {url}: {e}", exc_info=True)
        return None

def curate_resources(resources, preferences):
    """
    Filters and ranks resources based on user preferences and basic quality heuristics.
    """
    logger.info("Starting resource curation.")
    print("\n--- Curating Resources ---")
    curated_list = []
    
    goal_keywords = set(re.findall(r'\b\w+\b', preferences['learning_goal'].lower()))
    
    for resource in resources:
        if not resource:
            continue

        resource_score = 0
        
        # 1. Relevance Score (based on learning goal keywords)
        resource_text = (resource['title'] + " " + resource['description']).lower()
        found_keywords = [kw for kw in goal_keywords if kw in resource_text]
        resource_score += len(found_keywords) * 3 # Increased weight for keyword match

        # 2. Learning Style Preference
        if preferences['learning_style'] != 'mixed':
            if resource['type'] == preferences['learning_style']:
                resource_score += 5 # Stronger preference match
            elif resource['type'] != 'unknown':
                resource_score -= 3 # Penalize non-matching types
        else:
            if resource['type'] != 'unknown':
                resource_score += 2


        # 3. Skill Level Match (improved keywords)
        if preferences['skill_level'] == 'beginner':
            if any(level_kw in resource_text for level_kw in ["introduction", "beginner", "getting started", "basics", "for beginners", "what is", "fundamental"]):
                resource_score += 4
            if any(level_kw in resource_text for level_kw in ["advanced", "expert", "deep dive", "optimizing", "architecting", "production"]):
                resource_score -= 4
        elif preferences['skill_level'] == 'intermediate':
            if any(level_kw in resource_text for level_kw in ["intermediate", "practical", "applied", "projects", "use cases", "building", "advanced concepts"]):
                resource_score += 4
            if any(level_kw in resource_text for level_kw in ["introduction", "beginner", "basics"]): # Slightly penalize if too basic
                resource_score -= 1
        elif preferences['skill_level'] == 'advanced':
            if any(level_kw in resource_text for level_kw in ["advanced", "expert", "master", "deep dive", "optimization", "scalability", "performance", "design patterns", "architecture"]):
                resource_score += 4
            if any(level_kw in resource_text for level_kw in ["introduction", "beginner", "getting started", "basics"]):
                resource_score -= 3

        # 4. Add the basic quality score from extraction
        resource_score += resource.get('quality_score', 0) * 3 # Weight quality higher

        # 5. Penalize very short descriptions or generic titles
        if len(resource.get('description', '')) < 70 or "no description found" in resource.get('description', '').lower():
            resource_score -= 2
        if "no title found" in resource.get('title', '').lower():
            resource_score -= 2
        
        # Ensure score doesn't go too low (optional, but can help with sorting behavior)
        resource_score = max(0, resource_score)

        resource['curation_score'] = resource_score
        curated_list.append(resource)

    # Filter out resources with very low scores if desired, or just sort
    filtered_list = [res for res in curated_list if res['curation_score'] > 0] # Example: only keep resources with score > 0

    # Sort resources by their curation score in descending order
    filtered_list.sort(key=lambda x: x.get('curation_score', 0), reverse=True)
    
    logger.info(f"Finished curation. Found {len(filtered_list)} relevant resources.")
    return filtered_list[:7] # Show top N resources

def generate_html_report(resources, preferences):
    """
    Generates a simple HTML report of the curated resources.
    """
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a unique filename for each report
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
            .header {{ background-color: #2c3e50; color: #fff; padding: 25px 0; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }}
            .header h1 {{ margin: 0; font-size: 2.8em; }}
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
            .resource span.type {{ background-color: #d4edda; color: #155724; }} /* Green for Type */
            .resource span.score {{ background-color: #cce5ff; color: #004085; }} /* Blue for Score */
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

    if not resources:
        html_content += "<p>No highly relevant resources found for your query after careful curation. Please try a different learning goal or broaden your preferences!</p>"
        logger.info("No resources to display in HTML report after curation.")
    else:
        for i, res in enumerate(resources):
            html_content += f"""
            <div class="resource">
                <h3>{i+1}. <a href="{res.get('url', '#')}" target="_blank">{res.get('title', 'N/A')}</a></h3>
                <div class="meta">
                    <span class="type">{res.get('type', 'N/A').title()}</span>
                    <span class="score">Curation Score: {res.get('curation_score', 'N/A')}</span>
                </div>
                <p>{res.get('description', 'N/A')}</p>
            </div>
            """
    
    html_content += """
        </div>
        <div class="footer">
            <p>&copy; 2025 Agentic AI Learning Recommender. All rights reserved.</p>
            <p>Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
    </body>
    </html>
    """

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated at: {os.path.abspath(file_path)}")
    return file_path


if __name__ == "__main__":
    logger.info("Application started.")
    try:
        user_prefs = get_user_learning_preferences()

        search_query = (
            f"{user_prefs['learning_goal']} {user_prefs['learning_style']} "
            f"{user_prefs['skill_level']} tutorial"
        )
        
        # Further refine search query based on preferences
        if user_prefs['learning_style'] == 'video':
            search_query += " youtube course"
        elif user_prefs['learning_style'] == 'hands-on projects':
            search_query += " github project tutorial example"
        elif user_prefs['learning_style'] == 'articles':
            search_query += " blog guide medium"
        
        # Add skill level keywords to the search query as well
        search_query += f" {user_prefs['skill_level']}"


        search_results = perform_web_search(search_query, num_results=25) # Increased results again

        if search_results:
            logger.info("Starting detailed resource extraction.")
            print("\nExtracting details from search results:")
            collected_resources = []
            # Cap detailed scraping at 15-20 results for practical purposes
            max_scrape = min(len(search_results), 15) # Scrape max 15, or fewer if less results
            for i, result in enumerate(search_results[:max_scrape]):
                url = result.get('href')
                if url and url.startswith("http"): # Basic URL validation
                    resource_details = extract_resource_details(url)
                    if resource_details:
                        collected_resources.append(resource_details)
                    time.sleep(random.uniform(0.7, 2.5)) # Random delay
                else:
                    logger.debug(f"Skipping invalid URL from search results: {url}")
                print("-" * 30)

            if collected_resources:
                final_curated_resources = curate_resources(collected_resources, user_prefs)

                # --- Console Output ---
                print("\n" + "="*70)
                print(f"         ✨ Your Personalized Learning Resources for: {user_prefs['learning_goal']} ✨")
                print("="*70)
                if not final_curated_resources:
                    print("\nNo highly relevant resources found after curation. Try a different query or broader preferences!")
                else:
                    for i, res in enumerate(final_curated_resources):
                        print(f"\n--- Resource {i+1} (Curation Score: {res.get('curation_score', 'N/A')}) ---")
                        print(f"  Title: {res.get('title', 'N/A')}")
                        print(f"  URL: {res.get('url', 'N/A')}")
                        print(f"  Type: {res.get('type', 'N/A').title()}")
                        print(f"  Description: {res.get('description', 'N/A')[:250]}...")
                        print("-" * 50)
                
                # --- HTML Report Generation ---
                html_report_path = generate_html_report(final_curated_resources, user_prefs)
                
                open_html = input("\nDo you want to open the HTML report in your browser? (yes/no): ").lower()
                if open_html == 'yes':
                    try:
                        webbrowser.open('file://' + os.path.abspath(html_report_path))
                        logger.info(f"Opened HTML report: {os.path.abspath(html_report_path)}")
                    except Exception as e:
                        logger.error(f"Could not open browser automatically: {e}", exc_info=True)
                        print(f"Could not open browser automatically. Please open this file manually: {os.path.abspath(html_report_path)}")

            else:
                print("\nNo resources could be extracted after detailed analysis. This might be due to website blocking or content structure.")
                logger.warning("No resources collected after detailed analysis.")
        else:
            print("No initial search results found. Please check your internet connection or try a different learning goal.")
            logger.warning("No initial search results found from DuckDuckGo.")

    except Exception as overall_e:
        logger.critical(f"An unhandled critical error occurred: {overall_e}", exc_info=True)
        print(f"\nAn unexpected error occurred. Please check the '{log_file_path}' for details.")
    finally:
        logger.info("Application finished.")