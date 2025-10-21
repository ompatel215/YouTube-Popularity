# src/preprocess.py
"""
Data preprocessing and feature engineering
Normalizes numeric fields, computes engagement rate, and extracts useful features
"""

import pandas as pd
import numpy as np
import isodate
import re
from datetime import datetime, timedelta
import os

def iso_duration_to_minutes(duration):
    """Convert ISO 8601 duration to minutes (for API data)"""
    try:
        return isodate.parse_duration(duration).total_seconds() / 60
    except:
        return np.nan

def parse_views_text(views_text):
    """Parse view count from text like '8.8B views', '1.5M views', '1,234 views'"""
    if pd.isna(views_text) or views_text == '':
        return np.nan
    
    views_text = str(views_text).strip().upper()
    views_text = views_text.replace(' VIEWS', '').replace('VIEWS', '').strip()
    
    # Handle abbreviations
    multipliers = {'K': 1000, 'M': 1_000_000, 'B': 1_000_000_000}
    
    for suffix, multiplier in multipliers.items():
        if suffix in views_text:
            try:
                number = float(views_text.replace(suffix, '').replace(',', '').strip())
                return int(number * multiplier)
            except:
                return np.nan
    
    # Handle regular numbers with commas
    try:
        return int(views_text.replace(',', ''))
    except:
        return np.nan

def parse_upload_date_text(date_text):
    """Parse upload date from text like '8 years ago', '3 months ago', '5 days ago'"""
    if pd.isna(date_text) or date_text == '':
        return np.nan
    
    date_text = str(date_text).strip().lower()
    now = datetime.now()
    
    # Try to match patterns
    patterns = [
        (r'(\d+)\s*year', 'years'),
        (r'(\d+)\s*month', 'months'),
        (r'(\d+)\s*week', 'weeks'),
        (r'(\d+)\s*day', 'days'),
        (r'(\d+)\s*hour', 'hours'),
        (r'(\d+)\s*minute', 'minutes')
    ]
    
    for pattern, unit in patterns:
        match = re.search(pattern, date_text)
        if match:
            value = int(match.group(1))
            if unit == 'years':
                return now - timedelta(days=value*365)
            elif unit == 'months':
                return now - timedelta(days=value*30)
            elif unit == 'weeks':
                return now - timedelta(weeks=value)
            elif unit == 'days':
                return now - timedelta(days=value)
            elif unit == 'hours':
                return now - timedelta(hours=value)
            elif unit == 'minutes':
                return now - timedelta(minutes=value)
    
    # Try parsing as actual date (e.g., "2009. 10. 24.")
    try:
        # Handle format like "2009. 10. 24."
        cleaned = date_text.replace('.', '-').strip('-').strip()
        return pd.to_datetime(cleaned, errors='coerce')
    except:
        pass
    
    return np.nan

def parse_duration_text(duration_text):
    """Parse duration from text like '3:45', '1:23:45' to minutes"""
    if pd.isna(duration_text) or duration_text == '':
        return np.nan
    
    duration_text = str(duration_text).strip()
    parts = duration_text.split(':')
    
    try:
        if len(parts) == 2:  # MM:SS
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes + seconds / 60
        elif len(parts) == 3:  # HH:MM:SS
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 60 + minutes + seconds / 60
        elif len(parts) == 1:  # Just seconds
            return int(parts[0]) / 60
    except:
        pass
    
    return np.nan

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    if pd.isna(url):
        return np.nan
    
    # Match pattern like watch?v=VIDEO_ID
    match = re.search(r'[?&]v=([^&]+)', url)
    if match:
        return match.group(1)
    
    # Match pattern like youtu.be/VIDEO_ID
    match = re.search(r'youtu\.be/([^?&]+)', url)
    if match:
        return match.group(1)
    
    return np.nan

def preprocess_scraped_data(filename="data/raw/scraped_videos.csv", output="data/processed/scraped_processed.csv"):
    """
    Preprocess scraped YouTube data
    Input: title, channel, views, upload_date, duration, link
    Output: Cleaned and feature-engineered dataset
    """
    print(f"Loading scraped data from {filename}...")
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} videos")
    
    # Parse text fields to numeric/datetime
    print("Parsing views...")
    df['views'] = df['views'].apply(parse_views_text)
    
    print("Parsing upload dates...")
    df['publish_date'] = df['upload_date'].apply(parse_upload_date_text)
    
    print("Parsing durations...")
    df['duration_minutes'] = df['duration'].apply(parse_duration_text)
    
    print("Extracting video IDs...")
    df['videoId'] = df['link'].apply(extract_video_id)
    
    # Feature engineering
    print("Engineering features...")
    
    # Time since upload
    df['time_since_upload_days'] = (datetime.now() - df['publish_date']).dt.days
    
    # Title features
    df['title_len'] = df['title'].fillna('').str.split().apply(len)
    df['title_char_len'] = df['title'].fillna('').str.len()
    df['title_upper_ratio'] = df['title'].fillna('').apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    
    # Channel features
    df['has_channel'] = df['channel'].notna() & (df['channel'] != '')
    
    # View velocity (views per day)
    df['views_per_day'] = df['views'] / df['time_since_upload_days'].replace(0, np.nan)
    
    # Category prediction (we don't have this for scraped, so we'll infer basic categories)
    df['is_music'] = df['title'].fillna('').str.lower().str.contains('music|song|official video|ft\.|feat\.')
    df['is_gaming'] = df['title'].fillna('').str.lower().str.contains('gameplay|gaming|game|minecraft|fortnite')
    df['is_educational'] = df['title'].fillna('').str.lower().str.contains('how to|tutorial|learn|course|lesson')
    
    # Fill missing duration BEFORE cleaning (many scraped videos don't have duration)
    print("\nFilling missing durations...")
    median_duration = df['duration_minutes'].median()
    if pd.isna(median_duration) or median_duration == 0:
        median_duration = 5.0  # Default to 5 minutes if all are missing
    df['duration_minutes'] = df['duration_minutes'].fillna(median_duration)
    print(f"Filled {df['duration_minutes'].isna().sum()} missing durations with {median_duration:.1f} minutes")
    
    # Remove rows with critical missing values (but NOT duration anymore)
    print("\nCleaning data...")
    original_len = len(df)
    df = df.dropna(subset=['views', 'time_since_upload_days'])  # Removed duration_minutes from required fields
    df = df[df['views'] > 0]  # Remove videos with 0 views
    df = df[df['time_since_upload_days'] > 0]  # Remove future dates
    print(f"Removed {original_len - len(df)} rows with missing critical data")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    # Save processed data
    df.to_csv(output, index=False)
    print(f"\nPreprocessed {len(df)} videos saved to {output}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample statistics:")
    print(df[['views', 'duration_minutes', 'time_since_upload_days', 'title_len']].describe())
    
    return df

def preprocess_api_data(filename="data/raw/api_videos.csv", output="data/processed/api_processed.csv"):
    """
    Preprocess API YouTube data
    Input: videoId, title, description, publish_date, duration, tags, categoryId, viewCount, likeCount, commentCount
    Output: Cleaned and feature-engineered dataset
    """
    print(f"Loading API data from {filename}...")
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} videos")
    
    # Convert duration from ISO 8601
    print("Parsing durations...")
    df['duration_minutes'] = df['duration'].apply(iso_duration_to_minutes)
    
    # Convert numeric fields
    print("Converting numeric fields...")
    df['views'] = pd.to_numeric(df['viewCount'], errors='coerce')
    df['likes'] = pd.to_numeric(df['likeCount'], errors='coerce')
    df['comments'] = pd.to_numeric(df['commentCount'], errors='coerce')
    
    # Calculate engagement metrics
    print("Calculating engagement metrics...")
    df['engagement_rate'] = (df['likes'].fillna(0) + df['comments'].fillna(0)) / df['views'].replace(0, np.nan)
    df['like_rate'] = df['likes'].fillna(0) / df['views'].replace(0, np.nan)
    df['comment_rate'] = df['comments'].fillna(0) / df['views'].replace(0, np.nan)
    
    # Parse dates
    print("Parsing dates...")
    df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
    df['time_since_upload_days'] = (pd.Timestamp.now(tz='UTC') - df['publish_date']).dt.days
    
    # Title features
    print("Engineering title features...")
    df['title_len'] = df['title'].fillna('').str.split().apply(len)
    df['title_char_len'] = df['title'].fillna('').str.len()
    df['title_upper_ratio'] = df['title'].fillna('').apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    
    # Description features
    print("Engineering description features...")
    df['desc_len'] = df['description'].fillna('').str.split().apply(len)
    df['desc_char_len'] = df['description'].fillna('').str.len()
    
    # Tags features
    print("Engineering tags features...")
    df['num_tags'] = df['tags'].apply(lambda x: len(eval(x)) if pd.notna(x) and x != '[]' else 0)
    
    # Category features
    df['categoryId'] = df['categoryId'].fillna('0')
    
    # View velocity
    df['views_per_day'] = df['views'] / df['time_since_upload_days'].replace(0, np.nan)
    
    # Remove rows with critical missing values
    print("\nCleaning data...")
    original_len = len(df)
    df = df.dropna(subset=['views', 'duration_minutes', 'time_since_upload_days', 'engagement_rate'])
    df = df[df['views'] > 0]
    df = df[df['time_since_upload_days'] > 0]
    print(f"Removed {original_len - len(df)} rows with missing critical data")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    # Save processed data
    df.to_csv(output, index=False)
    print(f"\nPreprocessed {len(df)} videos saved to {output}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample statistics:")
    print(df[['views', 'likes', 'comments', 'engagement_rate', 'duration_minutes']].describe())
    
    return df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "scraped":
        preprocess_scraped_data()
    elif len(sys.argv) > 1 and sys.argv[1] == "api":
        preprocess_api_data()
    else:
        print("Usage: python preprocess.py [scraped|api]")
        print("Processing both datasets...")
        preprocess_scraped_data()
        preprocess_api_data()
