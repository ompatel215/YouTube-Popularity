import os
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.webdriver.common.by import By

def scrape_search_results(query="popular videos 2024", max_results=3000, headless=True):
    """
    Scrapes YouTube search results for video metadata.
    """
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    # Use search with filters for recent, popular videos
    # sp=CAMSAhAB filters by upload date
    search_url = f"https://www.youtube.com/results?search_query={query}&sp=CAMSAhAB"
    driver.get(search_url)
    
    time.sleep(8)
    
    # Accept cookies
    try:
        accept_button = driver.find_element(By.XPATH, "//button[@aria-label='Accept all']")
        accept_button.click()
        time.sleep(2)
    except:
        pass

    data = []
    last_height = driver.execute_script("return document.documentElement.scrollHeight")
    
    # Keep scrolling until we have enough videos
    while len(data) < max_results:
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(3)
        
        # Check if we've reached the bottom
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        
        # Get videos
        videos = driver.find_elements(By.CSS_SELECTOR, "ytd-video-renderer")
        print(f"Found {len(videos)} videos so far...")
        
        for video in videos[len(data):]:  # Only process new videos
            try:
                # Title and link
                title_el = video.find_element(By.CSS_SELECTOR, "#video-title")
                title = title_el.text.strip()
                link = title_el.get_attribute("href")
                
                if not title or not link:
                    continue
                
                # Channel
                channel = ""
                try:
                    channel = video.find_element(By.CSS_SELECTOR, "#channel-name").text.strip()
                except:
                    pass
                
                # Metadata (views, upload date)
                views = upload_date = ""
                try:
                    metadata_el = video.find_element(By.CSS_SELECTOR, "#metadata-line")
                    metadata_text = metadata_el.text
                    lines = metadata_text.split('\n')
                    views = lines[0] if len(lines) > 0 else ""
                    upload_date = lines[1] if len(lines) > 1 else ""
                except:
                    pass
                
                # Duration (optional but useful)
                duration = ""
                try:
                    duration = video.find_element(By.CSS_SELECTOR, "span.style-scope.ytd-thumbnail-overlay-time-status-renderer").text.strip()
                except:
                    pass
                
                data.append({
                    "title": title,
                    "channel": channel,
                    "views": views,
                    "upload_date": upload_date,
                    "duration": duration,
                    "link": link
                })
                
                if len(data) >= max_results:
                    break
                    
            except Exception as e:
                continue
        
        print(f"Collected {len(data)}/{max_results} videos")

    driver.quit()
    print(f"Successfully scraped {len(data)} videos.")
    return data


def save_to_csv(data, filename="data/raw/scraped_videos.csv"):
    """
    Save scraped video data to CSV file.
    """
    if not data:
        print("Warning: No data to save. Scraping returned 0 results.")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    keys = data[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved {len(data)} videos to {filename}")

def collect_diverse_dataset(target_count=3000, output_file="data/raw/scraped_videos.csv", headless=True):
    """
    Collect a diverse dataset of 3000+ videos using multiple search queries
    """
    print("="*60)
    print("WEB SCRAPING DATA COLLECTION")
    print("="*60)
    print(f"Target: {target_count} videos\n")
    
    # Diverse search queries
    search_queries = [
        "trending videos 2024",
        "popular music videos",
        "gaming highlights",
        "movie trailers 2024",
        "tech reviews",
        "cooking recipes",
        "fitness workout",
        "educational videos",
        "comedy videos",
        "travel vlogs"
    ]
    
    # Calculate videos per query
    results_per_query = (target_count // len(search_queries)) + 50
    
    print(f"Scraping across {len(search_queries)} different topics...")
    print(f"Target: ~{results_per_query} videos per query\n")
    
    all_scraped_data = []
    
    for i, query in enumerate(search_queries, 1):
        print(f"\n[{i}/{len(search_queries)}] Scraping: '{query}'...")
        try:
            data = scrape_search_results(
                query=query,
                max_results=results_per_query,
                headless=headless
            )
            all_scraped_data.extend(data)
            print(f"  Collected {len(data)} videos")
            print(f"  Total so far: {len(all_scraped_data)} videos")
            
            # Small delay between queries
            if i < len(search_queries):
                print(f"  Waiting 5 seconds before next query...")
                time.sleep(5)
                
        except Exception as e:
            print(f"  Error scraping '{query}': {e}")
            continue
    
    print(f"\nScraping Complete: {len(all_scraped_data)} videos collected")
    
    # Remove duplicates based on link
    unique_data = []
    seen_links = set()
    for video in all_scraped_data:
        if video['link'] not in seen_links:
            unique_data.append(video)
            seen_links.add(video['link'])
    
    print(f"After removing duplicates: {len(unique_data)} unique videos")
    
    # Save to CSV
    save_to_csv(unique_data, output_file)
    
    return unique_data