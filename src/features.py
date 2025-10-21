# src/features.py
"""
Feature Engineering Utilities

This module provides helper functions for computing video engagement metrics
and analyzing text content in video metadata.
"""

def compute_engagement_rate(likes, comments, views):
    """
    Calculate engagement rate as proportion of interactions to views.
    
    Args:
        likes: Number of likes
        comments: Number of comments
        views: Total view count
        
    Returns:
        Engagement rate (float between 0 and 1)
    """
    if views == 0:
        return 0
    return (likes + comments) / views

def count_keywords(text, keywords):
    """
    Count occurrences of keywords in text.
    
    Args:
        text: Text to search
        keywords: List of keywords to count
        
    Returns:
        Total count of all keyword occurrences
    """
    text = text.lower()
    return sum(text.count(k) for k in keywords)
