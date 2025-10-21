# src/youtube_api.py
"""
YouTube Data API Module
Collects structured video metadata using API v3
Requires: google-api-python-client
"""
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import pandas as pd
import time

# Load environment variables
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")  # Set your API key as env variable

if not API_KEY:
    print("WARNING: YOUTUBE_API_KEY not found in environment variables!")
    print("Please create a .env file with YOUTUBE_API_KEY=your_key_here")

youtube = build("youtube", "v3", developerKey=API_KEY) if API_KEY else None

def get_video_ids_from_search(query="trending", max_results=50, region_code="US"):
    """
    Search for videos and return their IDs
    Note: Each API call returns max 50 results, so we need to paginate
    """
    if not youtube:
        print("ERROR: YouTube API not initialized. Check your API key.")
        return []
    
    video_ids = []
    next_page_token = None
    
    try:
        while len(video_ids) < max_results:
            request = youtube.search().list(
                q=query,
                part="id",
                maxResults=min(50, max_results - len(video_ids)),
                type="video",
                regionCode=region_code,
                pageToken=next_page_token,
                order="relevance"  # Can be: relevance, date, viewCount, rating
            )
            response = request.execute()
            
            # Extract video IDs
            for item in response.get("items", []):
                if item["id"]["kind"] == "youtube#video":
                    video_ids.append(item["id"]["videoId"])
            
            # Check if there are more pages
            next_page_token = response.get("nextPageToken")
            if not next_page_token or len(video_ids) >= max_results:
                break
                
            time.sleep(0.5)  # Small delay to avoid rate limiting
            
    except HttpError as e:
        print(f"An HTTP error occurred: {e}")
    
    return video_ids[:max_results]

def get_video_ids_from_multiple_searches(queries, results_per_query=400):
    """
    Collect video IDs from multiple search queries
    """
    all_video_ids = set()  # Use set to avoid duplicates
    
    for query in queries:
        print(f"Searching for: '{query}'...")
        video_ids = get_video_ids_from_search(query, max_results=results_per_query)
        all_video_ids.update(video_ids)
        print(f"  Found {len(video_ids)} videos (Total unique: {len(all_video_ids)})")
        time.sleep(1)  # Delay between queries
    
    return list(all_video_ids)

def get_videos(video_ids):
    """
    Get detailed video information for a list of video IDs
    """
    if not youtube:
        print("ERROR: YouTube API not initialized. Check your API key.")
        return []
    
    all_videos = []
    total = len(video_ids)
    
    try:
        # Process in batches of 50 (API limit)
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            print(f"Fetching video details: {i+1}-{min(i+50, total)} of {total}...")
            
            response = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=",".join(batch)
            ).execute()
            
            for item in response.get("items", []):
                all_videos.append({
                    "videoId": item["id"],
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "publish_date": item["snippet"]["publishedAt"],
                    "duration": item["contentDetails"]["duration"],
                    "tags": item["snippet"].get("tags", []),
                    "categoryId": item["snippet"].get("categoryId"),
                    "viewCount": item["statistics"].get("viewCount", 0),
                    "likeCount": item["statistics"].get("likeCount", 0),
                    "commentCount": item["statistics"].get("commentCount", 0)
                })
            
            time.sleep(0.5)  # Small delay to avoid rate limiting
            
    except HttpError as e:
        print(f"An HTTP error occurred: {e}")
    
    return all_videos

def save_to_csv(data, filename="data/raw/api_videos.csv"):
    """
    Save video data to CSV file
    """
    if not data:
        print("Warning: No data to save.")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} videos to {filename}")

def collect_diverse_dataset(target_count=3000, output_file="data/raw/api_videos.csv"):
    """
    Collect a diverse dataset of 3000+ videos using multiple search queries
    """
    print("="*60)
    print("YOUTUBE API DATA COLLECTION")
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
        "comedy sketches",
        "travel vlogs",
        "news highlights",
        "sports highlights",
        "DIY tutorials",
        "product reviews"
    ]
    
    # Calculate videos per query
    results_per_query = (target_count // len(search_queries)) + 50
    
    print(f"Searching across {len(search_queries)} different topics...")
    print(f"Target: ~{results_per_query} videos per query\n")
    
    # Collect video IDs
    video_ids = get_video_ids_from_multiple_searches(search_queries, results_per_query)
    print(f"\nCollected {len(video_ids)} unique video IDs")
    
    # Get detailed video information
    print(f"\nFetching detailed information...")
    videos = get_videos(video_ids)
    print(f"\nRetrieved details for {len(videos)} videos")
    
    # Save to CSV
    save_to_csv(videos, output_file)
    
    return videos

if __name__ == "__main__":
    collect_diverse_dataset(target_count=3000)
