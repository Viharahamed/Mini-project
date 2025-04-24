import instaloader
import os
import json
from pathlib import Path
import sys
import time
import random
import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
import re
import tensorflow as tf # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from PIL import Image
import numpy as np
import cv2
import requests
from urllib.parse import urlparse

try:
    import getpass
except ImportError:
    # Handle cases where the `getpass` module is unavailable.
    getpass = None

def masked_input(prompt: str) -> str:
    """
    Cross-platform password input with masked characters
    (stars displayed for each entered character).
    """
    password = ""
    if sys.platform == "win32":
        # For Windows, use input with manual masking of password.
        import msvcrt
        print(prompt, end='', flush=True)
        while True:
            char = msvcrt.getch()  # Capture each keypress
            if char == b'\r':  # Enter key
                print()  # Move to a new line
                break
            elif char == b'\x08':  # Backspace key
                if len(password) > 0:
                    password = password[:-1]
                    sys.stdout.write('\b \b')  # Remove the last star
                    sys.stdout.flush()
            else:
                password += char.decode('utf-8')
                sys.stdout.write('*')  # Display '*' for each character
                sys.stdout.flush()
    else:
        # For Unix-based systems (Linux/macOS), use getpass for password input.
        password = getpass.getpass(prompt)

    return password

def create_sentiment_analyzer():
    """Create a sentiment analysis pipeline using transformers library"""
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        print(f"Error initializing sentiment analyzer: {str(e)}")
        print("Using fallback sentiment analysis method")
        return None

def create_image_analyzer():
    """Create a placeholder image analyzer"""
    print("TensorFlow not available, image analysis will be limited")
    return True  # Just return a placeholder

def analyze_profile_image(image_path, model):
    """Simplified image analysis with basic metrics only"""
    if not os.path.exists(image_path):
        return {
            "risk_score": 0.0,
            "categories": {},
            "analysis_performed": False
        }
    
    try:
        # Use only OpenCV for basic image analysis
        cv_img = cv2.imread(image_path)
        
        # Basic metrics
        average_pixel = np.mean(cv_img)
        risk_score = min(average_pixel / 255.0, 0.5)  # Simple placeholder calculation
        
        return {
            "risk_score": risk_score,
            "categories": {"basic_analysis": True},
            "analysis_performed": True
        }
    except Exception as e:
        print(f"Error in simplified image analysis: {str(e)}")
        return {
            "risk_score": 0.0,
            "categories": {},
            "analysis_performed": False
        }

def analyze_image_content(img_array, image_path):
    """
    Enhanced image content analysis including emotional content detection
    """
    try:
        # Convert to OpenCV format for analysis
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        # Get image dimensions
        height, width = cv_img.shape[:2]
        
        # Convert to different color spaces for analysis
        hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate basic metrics
        mean_intensity = np.mean(cv_img)
        std_intensity = np.std(cv_img)
        red_ratio = np.mean(rgb_img[:,:,0]) / (np.mean(rgb_img[:,:,1]) + np.mean(rgb_img[:,:,2]) + 1)
        
        # Color distribution (can indicate mood)
        mean_hsv = np.mean(hsv_img, axis=(0,1))
        color_saturation = mean_hsv[1] / 255.0  # More saturated colors can indicate intensity
        color_brightness = mean_hsv[2] / 255.0  # Brightness can indicate mood
        
        # Edge detection (more edges can mean more complex/busy image)
        edges = cv2.Canny(gray_img, 100, 200)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Calculate a comprehensive risk score
        risk_score = 0.0
        categories = {}
        
        # Analyze color emotion
        # Red can indicate anger, danger, or passion
        if red_ratio > 1.5:
            risk_score += 0.3
            categories["high_red_content"] = red_ratio
            
        # Very dark images (can indicate negative mood)
        if color_brightness < 0.3:
            risk_score += 0.15
            categories["dark_imagery"] = color_brightness
            
        # Very bright/washed out images
        if color_brightness > 0.9 and color_saturation < 0.2:
            risk_score += 0.1
            categories["washed_out_imagery"] = color_brightness
            
        # High contrast can indicate strong emotional content
        if std_intensity > 80:
            risk_score += 0.1
            categories["high_contrast"] = std_intensity
            
        # Very busy/complex images
        if edge_density > 0.3:
            risk_score += 0.1
            categories["complex_imagery"] = edge_density
            
        # Extreme saturation can indicate artificially enhanced images
        if color_saturation > 0.8:
            risk_score += 0.15
            categories["highly_saturated"] = color_saturation
        
        # Cap the risk score at 1.0
        risk_score = min(risk_score, 1.0)
        
        return {
            "risk_score": risk_score,
            "categories": categories,
            "metrics": {
                "mean_intensity": float(mean_intensity),
                "color_brightness": float(color_brightness),
                "color_saturation": float(color_saturation),
                "red_ratio": float(red_ratio),
                "edge_density": float(edge_density),
                "contrast": float(std_intensity)
            },
            "analysis_performed": True
        }
        
    except Exception as e:
        print(f"Error in detailed image analysis: {str(e)}")
        return {
            "risk_score": 0.0,
            "categories": {},
            "analysis_performed": False,
            "error": str(e)
        }

def fallback_sentiment_analysis(text):
    """
    Enhanced rule-based sentiment analysis as fallback when transformer model fails
    """
    if not text or len(text.strip()) == 0:
        return {"label": "NEUTRAL", "score": 0.5}

    # Comprehensive list of positive words
    positive_words = [
        # General Positive
        "good", "great", "excellent", "awesome", "amazing", "wonderful", "fantastic",
        "terrific", "outstanding", "superb", "brilliant", "spectacular", "fabulous",
        
        # Joy/Happiness
        "happy", "joy", "delighted", "blessed", "blissful", "cheerful", "content",
        "ecstatic", "elated", "glad", "jolly", "joyful", "joyous", "pleased",
        "thrilled", "upbeat", "satisfied", "radiant", "peaceful", "zen",
        
        # Love/Affection
        "love", "adore", "cherish", "passionate", "fond", "caring", "devoted",
        "admire", "appreciate", "treasure", "embrace", "romantic", "heartfelt",
        
        # Success/Achievement
        "success", "achieve", "accomplish", "victory", "win", "triumph", "excel",
        "prosper", "thrive", "flourish", "succeed", "accomplish", "achieve",
        
        # Beauty/Quality
        "beautiful", "gorgeous", "stunning", "elegant", "graceful", "lovely",
        "attractive", "charming", "delightful", "magnificent", "perfect", "refined",
        
        # Strength/Power
        "strong", "powerful", "confident", "capable", "determined", "resilient",
        "brave", "bold", "courageous", "fearless", "dynamic", "energetic",
        
        # Growth/Improvement
        "grow", "improve", "enhance", "advance", "progress", "develop", "evolve",
        "upgrade", "better", "superior", "exceptional", "outstanding",
        
        # Support/Encouragement
        "support", "encourage", "inspire", "motivate", "help", "assist", "guide",
        "mentor", "nurture", "uplift", "empower", "strengthen"
    ]

    # Comprehensive list of negative words with added profanity
    negative_words = [
        # General Negative
        "bad", "terrible", "horrible", "awful", "poor", "unpleasant", "negative",
        "worse", "worst", "inferior", "mediocre", "inadequate", "disappointing",
        
        # Anger/Rage
        "angry", "rage", "furious", "hostile", "outraged", "bitter", "hateful",
        "irritated", "annoyed", "frustrated", "agitated", "mad", "enraged","blood"
        
        # Sadness/Depression
        "sad", "depressed", "miserable", "unhappy", "gloomy", "heartbroken",
        "melancholy", "despair", "grief", "sorrow", "hopeless", "devastated",
        
        # Fear/Anxiety
        "afraid", "scared", "terrified", "anxious", "nervous", "worried", "panic",
        "dread", "frightened", "horrified", "paranoid", "threatened", "fearful",
        
        # Disgust/Hate
        "hate", "disgust", "despise", "loathe", "detest", "revolting", "offensive",
        "repulsive", "nasty", "gross", "hideous", "vile", "repugnant",
        
        # Violence/Harm
        "violent", "brutal", "cruel", "vicious", "savage", "aggressive", "hostile",
        "abusive", "dangerous", "deadly", "lethal", "threatening", "harmful",
        
        # Failure/Loss
        "fail", "failure", "lose", "lost", "defeat", "rejected", "worthless",
        "useless", "incompetent", "pathetic", "weak", "helpless", "powerless",
        
        # Crime/Illegal
        "illegal", "criminal", "corrupt", "fraud", "theft", "stolen", "cheat",
        "deceive", "manipulate", "exploit", "abuse", "violation", "wrongdoing",
        
        # Disaster/Crisis
        "disaster", "catastrophe", "crisis", "emergency", "tragedy", "calamity",
        "destruction", "chaos", "havoc", "ruin", "damage", "devastation",
        
        # Social Issues
        "racist", "sexist", "discrimination", "prejudice", "bias", "harassment",
        "bullying", "oppression", "inequality", "injustice", "unfair", "toxic",
        
        # Profanity/Curse Words
        "damn", "hell", "crap", "shit", "fuck", "asshole", "bitch", "bastard", 
        "douchebag", "piss", "dick", "cock", "pussy", "whore", "slut", "ass",
        "jackass", "bullshit", "goddamn", "motherfucker", "wtf", "stfu", "lmao",
        "idiot", "moron", "stupid", "dumb", "retard", "dumbass", "jerk",
        
        # Insulting Terms
        "stupid", "idiot", "dumb", "moron", "imbecile", "fool", "ridiculous",
        "lame", "absurd", "obnoxious", "ignorant", "incompetent", "worthless",
        "loser", "creep", "weirdo", "psycho", "nutcase", "freak", "scumbag",
        
        # Dismissive/Belittling
        "whatever", "nonsense", "rubbish", "garbage", "trash", "worthless",
        "pathetic", "joke", "laughable", "meh", "boring", "lame", "stupid"
    ]

    # Convert text to lowercase
    text_lower = text.lower()
    
    # Count positive and negative words
    pos_count = 0
    neg_count = 0
    
    # Process positive words
    for word in positive_words:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            pos_count += 1
    
    # Process negative words
    for word in negative_words:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            neg_count += 1
    
    # List of profanity for extra weighting
    profanity_words = ["fuck", "shit", "asshole", "bitch", "bastard", "motherfucker"]
    
    # Add extra weight for profanity (count them twice)
    for word in profanity_words:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            neg_count += 1  # Count them twice for stronger weight
    
    # Calculate sentiment score with more granular scaling
    total_words = pos_count + neg_count
    if total_words == 0:
        return {"label": "NEUTRAL", "score": 0.5}
    
    # Calculate normalized sentiment score
    if pos_count > neg_count:
        score = 0.5 + min(0.49, (pos_count / (total_words + 1)) * 0.5)
        return {"label": "POSITIVE", "score": score}
    elif neg_count > pos_count:
        score = 0.5 - min(0.49, (neg_count / (total_words + 1)) * 0.5)
        return {"label": "NEGATIVE", "score": score}
    else:
        return {"label": "NEUTRAL", "score": 0.5}

def analyze_text_sentiment(text, analyzer):
    """Analyze sentiment of a single text"""
    if not text or len(text.strip()) == 0:
        return {"label": "NEUTRAL", "score": 0.5}

    try:
        if analyzer:
            return analyzer(text)[0]
        else:
            return fallback_sentiment_analysis(text)
    except Exception as e:
        print(f"Error in sentiment analysis, using fallback: {str(e)}")
        return fallback_sentiment_analysis(text)

def analyze_instagram_profile(username, analyzer, image_analyzer, insta_loader, retry_count=3, retry_delay=5):
    """
    Analyze an Instagram profile for sentiment risks with retry mechanism

    Args:
        username (str): Instagram username
        analyzer: Sentiment analysis pipeline
        image_analyzer: CNN model for image analysis
        insta_loader: Instaloader instance
        retry_count (int): Number of retries on failure
        retry_delay (int): Base delay between retries in seconds

    Returns:
        dict: Profile analysis results
    """
    for attempt in range(retry_count):
        try:
            profile = instaloader.Profile.from_username(insta_loader.context, username)

            # Extract profile data
            profile_data = {
                "username": username,
                "fullname": profile.full_name,
                "bio": profile.biography,
                "followers": profile.followers,
                "following": profile.followees,
                "is_private": profile.is_private,
                "external_url": profile.external_url,
                "business_category": profile.business_category_name,
            }

            # Download profile picture based on version
            profile_pic_path = None
            try:
                # Create a temporary directory for the profile picture
                temp_dir = os.path.join(os.path.expanduser("~"), "temp_instagram_analysis", username)
                os.makedirs(temp_dir, exist_ok=True)
                
                # Change working directory temporarily
                original_dir = os.getcwd()
                os.chdir(temp_dir)
                
                try:
                    # Try the method that may work for your version
                    if hasattr(insta_loader, 'download_pic'):
                        insta_loader.download_pic(filename=f"{username}_profile_pic.jpg", 
                                                  url=profile.profile_pic_url,
                                                  mtime=None)
                    else:
                        # Use the profile download method instead
                        insta_loader.download_profile(profile.username, profile_pic_only=True)
                    
                    # Find the downloaded profile picture
                    os.chdir(original_dir)
                    for file in os.listdir(temp_dir):
                        if file.endswith((".jpg", ".jpeg", ".png")):
                            profile_pic_path = os.path.join(temp_dir, file)
                            break
                finally:
                    os.chdir(original_dir)
            except Exception as e:
                print(f"Error downloading profile picture: {str(e)}")
                profile_pic_path = None

            # Analyze username
            username_sentiment = analyze_text_sentiment(username, analyzer)

            # Analyze full name
            fullname_sentiment = analyze_text_sentiment(profile.full_name, analyzer)

            # Analyze bio
            bio_sentiment = analyze_text_sentiment(profile.biography, analyzer)

            # Analyze profile picture
            image_analysis = None
            if profile_pic_path and os.path.exists(profile_pic_path) and image_analyzer:
                image_analysis = analyze_profile_image(profile_pic_path, image_analyzer)

            # Calculate risk scores based on text analysis
            text_risk_scores = {
                "username": 1 if username_sentiment["label"] == "NEGATIVE" else 0,
                "fullname": 1 if fullname_sentiment["label"] == "NEGATIVE" else 0,
                "bio": 1 if bio_sentiment["label"] == "NEGATIVE" else 0,
                "bio_confidence": bio_sentiment["score"]
            }

            # Calculate overall risk including image analysis if available
            if image_analysis and image_analysis.get("analysis_performed", False):
                # Combine text and image risk scores (30% for text, 70% for image)
                text_risk = (
                    (text_risk_scores["username"] * 0.2) +
                    (text_risk_scores["fullname"] * 0.2) +
                    (text_risk_scores["bio"] * 0.6)
                )
                overall_risk = (text_risk * 0.3) + (image_analysis["risk_score"] * 0.7)
                
                # Include image risk in the risk scores
                text_risk_scores["profile_image"] = image_analysis["risk_score"]
            else:
                # Use only text risk if image analysis is not available
                overall_risk = (
                    (text_risk_scores["username"] * 0.2) +
                    (text_risk_scores["fullname"] * 0.2) +
                    (text_risk_scores["bio"] * 0.6)
                )

            # Additional sentiment analysis for bio text
            additional_risk_factors = detect_risk_keywords(profile.biography)

            return {
                "profile_data": profile_data,
                "sentiment_analysis": {
                    "username": username_sentiment,
                    "fullname": fullname_sentiment,
                    "bio": bio_sentiment,
                },
                "image_analysis": image_analysis,
                "profile_pic_path": profile_pic_path,
                "risk_assessment": {
                    "individual_risks": text_risk_scores,
                    "overall_risk": overall_risk,
                    "risk_level": get_risk_level(overall_risk),
                    "risk_keywords_detected": additional_risk_factors,
                }
            }

        except instaloader.exceptions.ConnectionException as e:
            if "429" in str(e):
                # Rate limiting error - implement exponential backoff
                wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Waiting for {wait_time:.1f} seconds before retry {attempt+1}/{retry_count}")
                time.sleep(wait_time)
            elif "401" in str(e):
                # Authentication error
                print("Authentication error. Might need to wait or use a different account.")
                if attempt < retry_count - 1:
                    wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Waiting for {wait_time:.1f} seconds before retry {attempt+1}/{retry_count}")
                    time.sleep(wait_time)
                else:
                    return {"error": "Instagram authentication error. Try again later or use a different account."}
            else:
                # Other connection errors
                print(f"Connection error: {str(e)}")
                if attempt < retry_count - 1:
                    wait_time = retry_delay * (attempt + 1) + random.uniform(0, 1)
                    print(f"Waiting for {wait_time:.1f} seconds before retry {attempt+1}/{retry_count}")
                    time.sleep(wait_time)
                else:
                    return {"error": f"Failed to connect to Instagram after {retry_count} attempts: {str(e)}"}

        except instaloader.exceptions.ProfileNotExistsException:
            return {"error": f"Profile '{username}' does not exist"}

        except Exception as e:
            print(f"Error analyzing profile (attempt {attempt+1}/{retry_count}): {str(e)}")
            if attempt < retry_count - 1:
                wait_time = retry_delay * (attempt + 1) + random.uniform(0, 1)
                print(f"Waiting for {wait_time:.1f} seconds before retry")
                time.sleep(wait_time)
            else:
                return {"error": f"Failed to analyze profile after {retry_count} attempts: {str(e)}"}

    return {"error": "Failed to analyze profile after multiple attempts"}

def detect_risk_keywords(text):
    """
    Detect potential risk keywords in text

    Args:
        text (str): Text to analyze

    Returns:
        list: List of detected risk keywords
    """
    if not text:
        return []

    # List of potentially concerning keywords
    risk_keywords = [
        "hate", "anger", "violent", "kill", "threat", "weapon", "gun",
        "suicide", "harm", "danger", "illegal", "drug", "abuse",
        "extremist", "radical", "attack", "revenge"
    ]

    detected = []
    text_lower = text.lower()

    for keyword in risk_keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text_lower):
            detected.append(keyword)

    return detected

def get_risk_level(risk_score):
    """Convert numerical risk score to categorical level"""
    if risk_score < 0.2:
        return "Low"
    elif risk_score < 0.5:
        return "Medium"
    else:
        return "High"

def analyze_post_sentiment(post_data, analyzer):
    """
    Analyze sentiment of a post's caption

    Args:
        post_data (dict): Post data
        analyzer: Sentiment analysis pipeline

    Returns:
        dict: Post with sentiment analysis
    """
    caption = post_data.get("caption", "")

    sentiment_result = analyze_text_sentiment(caption, analyzer)

    result = post_data.copy()
    result["sentiment"] = sentiment_result["label"]
    result["sentiment_score"] = sentiment_result["score"]
    result["risk_keywords"] = detect_risk_keywords(caption)

    return result

def download_profile_posts(username, save_folder=None, post_limit=5, analyzer=None, image_analyzer=None, username_login=None, password=None, retry_count=3):
    """
    Download recent posts from a public Instagram profile and analyze sentiments

    Args:
        username (str): Instagram username
        save_folder (str): Base path to save the data
        post_limit (int): Number of recent posts to download
        analyzer: Sentiment analysis pipeline
        image_analyzer: CNN model for image analysis
        username_login (str): Instagram login username (optional)
        password (str): Instagram login password (optional)
        retry_count (int): Number of retries on failure

    Returns:
        dict: Status message and analysis results
    """
    # Set the save folder to the specified path
    save_folder = r"C:\Users\vihar\Downloads\Mini pro"

    try:
        # Initialize the image analyzer if not provided
        if image_analyzer is None:
            image_analyzer = create_image_analyzer()

        # Create user-specific folder
        user_folder = os.path.join(save_folder, username)
        posts_folder = os.path.join(user_folder, 'posts')
        analysis_folder = os.path.join(user_folder, 'analysis')
        os.makedirs(posts_folder, exist_ok=True)
        os.makedirs(analysis_folder, exist_ok=True)

        # Initialize Instaloader with more conservative settings to avoid rate limiting
        L = instaloader.Instaloader(
            download_pictures=True,
            download_videos=True,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=True,
            compress_json=False,
            post_metadata_txt_pattern='',
            max_connection_attempts=retry_count,
            request_timeout=60)

        # Ask for Instagram credentials if not provided
        if username_login is None or password is None:
            print("\nInstagram login required to download posts")
            username_login = input("Enter your Instagram username: ")
            password = masked_input("Enter your Instagram password: ")  # Password input is masked with stars

        login_success = False
        for attempt in range(retry_count):
            try:
                L.login(username_login, password)
                print("Login successful!")
                login_success = True
                break
            except instaloader.exceptions.ConnectionException as e:
                if "429" in str(e):
                    wait_time = 5 * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limited during login. Waiting for {wait_time:.1f} seconds before retry {attempt+1}/{retry_count}")
                    time.sleep(wait_time)
                elif "401" in str(e):
                    print("Authentication error. Please check your username and password.")
                    return {"status": f"Login failed: {str(e)}"}
                else:
                    print(f"Connection error during login: {str(e)}")
                    if attempt < retry_count - 1:
                        wait_time = 5 * (attempt + 1) + random.uniform(0, 1)
                        print(f"Waiting for {wait_time:.1f} seconds before retry {attempt+1}/{retry_count}")
                        time.sleep(wait_time)
            except Exception as e:
                print(f"Login error (attempt {attempt+1}/{retry_count}): {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(5)

        if not login_success:
            return {"status": "Login failed after multiple attempts. Please try again later."}

        # First, analyze the profile itself
        profile_analysis = analyze_instagram_profile(username, analyzer, image_analyzer, L, retry_count)

        if "error" in profile_analysis:
            return {"status": profile_analysis["error"]}

        # Save profile analysis
        profile_analysis_file = os.path.join(analysis_folder, 'profile_analysis.json')
        with open(profile_analysis_file, 'w', encoding='utf-8') as f:
            json.dump(profile_analysis, f, indent=4, ensure_ascii=False)

        # Get profile and posts
        try:
            profile = instaloader.Profile.from_username(L.context, username)
        except Exception as e:
            return {"status": f"Error retrieving profile: {str(e)}", "profile_analysis": profile_analysis}

        if profile.mediacount > 0:
            print(f"\nDownloading up to {post_limit} recent posts...")
            posts_data = []
            analyzed_posts = []

            # Get iterator for posts
            try:
                posts_iterator = profile.get_posts()
            except Exception as e:
                return {
                    "status": f"Error retrieving posts: {str(e)}",
                    "profile_analysis": profile_analysis
                }

            for index, post in enumerate(posts_iterator):
                if index >= post_limit:
                    break

                try:
                    print(f"\nDownloading post {index + 1}/{post_limit}")

                    # Create post-specific folder
                    post_date = post.date_local.strftime("%Y%m%d")
                    post_folder = os.path.join(posts_folder, f"post_{post_date}_{post.shortcode}")
                    os.makedirs(post_folder, exist_ok=True)

                    # Slow down to avoid rate limiting
                    if index > 0:
                        time.sleep(random.uniform(1.0, 3.0))

                    # Download the post (with retry)
                    download_success = False
                    for attempt in range(retry_count):
                        try:
                            L.download_post(post, target=post_folder)
                            download_success = True
                            break
                        except instaloader.exceptions.ConnectionException as e:
                            if "429" in str(e):
                                wait_time = 5 * (2 ** attempt) + random.uniform(0, 1)
                                print(f"Rate limited. Waiting for {wait_time:.1f} seconds before retry {attempt+1}/{retry_count}")
                                time.sleep(wait_time)
                            else:
                                print(f"Connection error: {str(e)}")
                                if attempt < retry_count - 1:
                                    time.sleep(5)
                        except Exception as e:
                            print(f"Download error: {str(e)}")
                            if attempt < retry_count - 1:
                                time.sleep(5)

                    if not download_success:
                        print(f"Failed to download post {index + 1} after multiple attempts. Skipping.")
                        continue

                    # Save post information
                    post_info = {
                        "shortcode": post.shortcode,
                        "date": post.date_local.strftime("%Y-%m-%d %H:%M:%S"),
                        "caption": post.caption if post.caption else "",
                        "likes": post.likes,
                        "is_video": post.is_video,
                        "url": f"https://www.instagram.com/p/{post.shortcode}/",
                    }

                    # Analyze sentiment if analyzer provided
                    if analyzer is not None:
                        analyzed_post = analyze_post_sentiment(post_info, analyzer)
                        analyzed_posts.append(analyzed_post)
                    else:
                        analyzed_posts.append(post_info)

                    # Save post info to JSON
                    info_file = os.path.join(post_folder, 'post_info.json')
                    with open(info_file, 'w', encoding='utf-8') as f:
                        json.dump(post_info, f, indent=4, ensure_ascii=False)

                    posts_data.append(post_info)

                except Exception as e:
                    print(f"Error downloading post {index + 1}: {str(e)}")
                    continue

            # Save summary of all posts
            summary_file = os.path.join(posts_folder, 'posts_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(posts_data, f, indent=4, ensure_ascii=False)

            # Save sentiment analysis of all posts
            analysis_file = os.path.join(analysis_folder, 'posts_sentiment_analysis.json')
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analyzed_posts, f, indent=4, ensure_ascii=False)

            # Calculate overall sentiment statistics
            sentiment_stats = calculate_sentiment_stats(analyzed_posts)
            stats_file = os.path.join(analysis_folder, 'sentiment_statistics.json')
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(sentiment_stats, f, indent=4, ensure_ascii=False)

            return {
                "status": f"""Downloads and analysis completed successfully:
- Posts saved to: {posts_folder}
- Analysis saved to: {analysis_folder}
- Total posts downloaded: {len(posts_data)}""",
                "profile_analysis": profile_analysis,
                "posts_analysis": analyzed_posts,
                "sentiment_stats": sentiment_stats
            }
        else:
            return {
                "status": "This profile has no posts to download.",
                "profile_analysis": profile_analysis
            }

    except instaloader.exceptions.ProfileNotExistsException:
        return {"status": f"Error: Profile '{username}' does not exist"}
    except Exception as e:
        return {"status": f"Error occurred: {str(e)}"}

def calculate_sentiment_stats(analyzed_posts):
    """Calculate overall sentiment statistics from analyzed posts"""
    if not analyzed_posts:
        return {
            "total_posts": 0,
            "sentiment_distribution": {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0},
            "average_sentiment_score": 0,
            "risk_keywords_frequency": {}
        }

    total = len(analyzed_posts)
    sentiments = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    total_score = 0
    keywords_freq = {}

    for post in analyzed_posts:
        sentiment = post.get("sentiment", "NEUTRAL")
        sentiments[sentiment] = sentiments.get(sentiment, 0) + 1

        score = post.get("sentiment_score", 0.5)
        total_score += score

        for keyword in post.get("risk_keywords", []):
            keywords_freq[keyword] = keywords_freq.get(keyword, 0) + 1

    # Calculate percentages
    sentiment_distribution = {}
    for sentiment, count in sentiments.items():
        if count > 0:
            sentiment_distribution[sentiment] = f"{count} ({(count/total)*100:.1f}%)"

    return {
        "total_posts": total,
        "sentiment_distribution": sentiment_distribution,
        "average_sentiment_score": total_score / total if total > 0 else 0,
        "risk_keywords_frequency": keywords_freq
    }

def main():
    # Initialize analyzers
    analyzer = create_sentiment_analyzer()
    image_analyzer = create_image_analyzer()
    
    username = input("Enter Instagram username to download posts from: ")
    custom_location = input("Enter custom save location (press Enter to use default Pictures folder): ").strip()
    post_limit = input("Enter number of recent posts to download (default is 5): ").strip()

    save_folder = custom_location if custom_location else None
    post_limit = int(post_limit) if post_limit.isdigit() else 5

    result = download_profile_posts(username, save_folder, post_limit, analyzer, image_analyzer)
    print("\n" + result["status"])

    # Print profile risk assessment summary
    if "profile_analysis" in result and "risk_assessment" in result["profile_analysis"]:
        risk = result["profile_analysis"]["risk_assessment"]
        print("\nProfile Risk Assessment:")
        print(f"Risk Level: {risk['risk_level']}")
        if risk.get("risk_keywords_detected"):
            print(f"Risk Keywords Detected: {', '.join(risk['risk_keywords_detected'])}")

    # Print post sentiment summary
    if "sentiment_stats" in result:
        stats = result["sentiment_stats"]
        print("\nPost Sentiment Analysis Summary:")
        print(f"Total Posts Analyzed: {stats['total_posts']}")
        print("Sentiment Distribution:")
        for sentiment, count in stats.get("sentiment_distribution", {}).items():
            print(f"  - {sentiment}: {count}")

        if stats.get("risk_keywords_frequency"):
            print("Risk Keywords Detected in Posts:")
            for keyword, freq in stats["risk_keywords_frequency"].items():
                print(f"  - {keyword}: {freq} occurrences")

def streamlit_app():
    st.set_page_config(page_title="Instagram Profile Risk Analysis", layout="wide")

    st.title("Instagram Profile Risk Analysis Dashboard")
    st.write("Analyze Instagram profiles and posts for sentiment and potential risk factors")

    # Initialize analyzers
    analyzer = create_sentiment_analyzer()
    image_analyzer = create_image_analyzer()

    # Add session state for storing analysis results
    if 'profile_analysis' not in st.session_state:
        st.session_state.profile_analysis = None
    if 'posts_analysis' not in st.session_state:
        st.session_state.posts_analysis = []
    if 'login_status' not in st.session_state:
        st.session_state.login_status = None

    # Sidebar for login credentials
    with st.sidebar:
        st.header("Instagram Credentials")
        st.warning("Note: Instagram may rate-limit your requests. If you encounter errors, try again later or use a different account.")

        username_login = st.text_input("Your Instagram Username")
        password = st.text_input("Your Instagram Password", type="password")

        if st.button("Test Login"):
            if not username_login or not password:
                st.error("Please provide username and password")
            else:
                with st.spinner("Testing login..."):
                    # Initialize Instaloader
                    L = instaloader.Instaloader()
                    try:
                        L.login(username_login, password)
                        st.session_state.login_status = "success"
                        st.success("Login successful!")
                    except Exception as e:
                        st.session_state.login_status = "failed"
                        st.error(f"Login failed: {str(e)}")

        st.header("Analysis Settings")
        target_username = st.text_input("Target Instagram Profile to Analyze")
        post_limit = st.number_input("Number of Posts to Analyze", min_value=1, max_value=20, value=5)
        retry_count = st.number_input("Number of Retry Attempts", min_value=1, max_value=5, value=3)

        analyze_button = st.button("Analyze Profile")

    # Main content area
    if analyze_button:
        if not username_login or not password or not target_username:
            st.error("Please provide all required information")
        else:
            with st.spinner(f"Analyzing Instagram profile: @{target_username}"):
                # Setup temporary directory for downloads
                temp_dir = os.path.join(os.path.expanduser("~"), "temp_instagram_analysis")

                try:
                    # Perform analysis with enhanced error handling
                    result = download_profile_posts(
                        target_username,
                        save_folder=temp_dir,
                        post_limit=post_limit,
                        analyzer=analyzer,
                        image_analyzer=image_analyzer,
                        username_login=username_login,
                        password=password,
                        retry_count=retry_count
                    )

                    if "profile_analysis" in result and "error" not in result["profile_analysis"]:
                        st.session_state.profile_analysis = result["profile_analysis"]
                        st.session_state.posts_analysis = result.get("posts_analysis", [])
                        st.success("Analysis completed!")
                    else:
                        st.error(result["status"])

                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")

    # Display analysis results if available
    if st.session_state.profile_analysis:
        profile_analysis = st.session_state.profile_analysis

        # Display profile information
        col1, col2 = st.columns(2)

        with col1:
            st.header(f"Profile: @{profile_analysis['profile_data']['username']}")
            st.write(f"Full Name: {profile_analysis['profile_data']['fullname']}")
            st.write(f"Followers: {profile_analysis['profile_data']['followers']:,}")
            st.write(f"Following: {profile_analysis['profile_data']['following']:,}")
            st.write(f"Private: {'Yes' if profile_analysis['profile_data']['is_private'] else 'No'}")

            # New section for profile picture
            st.subheader("Profile Picture")
            # Find the profile picture path (we'll need to update the analyze_instagram_profile function to return this)
            profile_pic_path = profile_analysis.get('profile_pic_path')
            if profile_pic_path and os.path.exists(profile_pic_path):
                try:
                    # Display the profile picture
                    img = Image.open(profile_pic_path)
                    st.image(img, width=250, caption=f"@{profile_analysis['profile_data']['username']}'s Profile Picture")
                except Exception as e:
                    st.error(f"Error displaying profile picture: {str(e)}")
            else:
                st.write("Profile picture not available")

            st.subheader("Biography")
            st.write(profile_analysis['profile_data']['bio'])

        with col2:
            st.header("Profile Risk Assessment")

            risk_level = profile_analysis['risk_assessment']['risk_level']
            risk_color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"

            st.markdown(f"<h3 style='color:{risk_color}'>Overall Risk Level: {risk_level}</h3>", unsafe_allow_html=True)

            # Display profile image analysis in a prominent way
            if 'image_analysis' in profile_analysis and profile_analysis['image_analysis'] and profile_analysis['image_analysis'].get('analysis_performed', False):
                image_analysis = profile_analysis['image_analysis']
                
                # Create a visual indicator for image risk
                image_risk = image_analysis['risk_score']
                img_risk_color = "green" if image_risk < 0.3 else "orange" if image_risk < 0.6 else "red"
                img_risk_level = "Low" if image_risk < 0.3 else "Medium" if image_risk < 0.6 else "High"
                
                st.markdown("<h4>Profile Picture Analysis</h4>", unsafe_allow_html=True)
                st.markdown(f"<h5>Risk Level: <span style='color:{img_risk_color}'>{img_risk_level} ({image_risk:.2f})</span></h5>", 
                           unsafe_allow_html=True)
                
                # Make a simple gauge-like visualization
                img_risk_percentage = int(image_risk * 100)
                st.progress(image_risk)
                
                # Display detected categories
                if image_analysis.get('categories'):
                    st.markdown("<h5>Risk Factors Detected:</h5>", unsafe_allow_html=True)
                    for category, value in image_analysis['categories'].items():
                        st.write(f"• {category.replace('_', ' ').title()}: {value:.2f}")
                
                # Display image metrics
                if 'metrics' in image_analysis:
                    with st.expander("View Technical Image Metrics"):
                        for metric, value in image_analysis['metrics'].items():
                            st.write(f"{metric.replace('_', ' ').title()}: {value:.2f}")
            else:
                st.write("Profile picture analysis not available")

            # Display sentiment analysis results
            st.subheader("Text Content Analysis")

            sentiment_df = pd.DataFrame({
                "Element": ["Username", "Full Name", "Biography"],
                "Sentiment": [
                    profile_analysis['sentiment_analysis']['username']['label'],
                    profile_analysis['sentiment_analysis']['fullname']['label'],
                    profile_analysis['sentiment_analysis']['bio']['label']
                ],
                "Confidence": [
                    profile_analysis['sentiment_analysis']['username']['score'] * 100,
                    profile_analysis['sentiment_analysis']['fullname']['score'] * 100,
                    profile_analysis['sentiment_analysis']['bio']['score'] * 100
                ]
            })

            st.dataframe(sentiment_df)

            # Display risk keywords if any
            if profile_analysis['risk_assessment']['risk_keywords_detected']:
                st.subheader("Risk Keywords Detected")
                st.write(", ".join(profile_analysis['risk_assessment']['risk_keywords_detected']))

        # Display posts analysis
        posts_analysis = st.session_state.posts_analysis

        if posts_analysis:
            st.header("Posts Analysis")

            # Create DataFrame for analysis
            posts_df = pd.DataFrame([
                {
                    'date': post.get('date', 'Unknown'),
                    'sentiment': post.get('sentiment', 'NEUTRAL'),
                    'sentiment_score': post.get('sentiment_score', 0.5),
                    'likes': post.get('likes', 0),
                    'has_risk_keywords': len(post.get('risk_keywords', [])) > 0
                }
                for post in posts_analysis
            ])

            # Display sentiment distribution
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Posts Sentiment Distribution")
                sentiment_counts = posts_df['sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']

                fig = px.pie(
                    sentiment_counts,
                    values='Count',
                    names='Sentiment',
                    color='Sentiment',
                    color_discrete_map={
                        'POSITIVE': 'green',
                        'NEGATIVE': 'red',
                        'NEUTRAL': 'gray'
                    }
                )
                st.plotly_chart(fig)

            # Calculate and display risk keywords
            all_keywords = []
            for post in posts_analysis:
                all_keywords.extend(post.get('risk_keywords', []))

            keyword_counts = {}
            for keyword in all_keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

            if keyword_counts:
                with col2:
                    st.subheader("Risk Keywords in Posts")
                    keywords_df = pd.DataFrame({
                        'Keyword': list(keyword_counts.keys()),
                        'Occurrences': list(keyword_counts.values())
                    })
                    fig = px.bar(
                        keywords_df,
                        x='Keyword',
                        y='Occurrences',
                        color='Occurrences',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig)

            # Display individual posts with sentiment
            st.subheader("Individual Post Analysis")
            for idx, post in enumerate(posts_analysis):
                with st.expander(f"Post {idx+1} - {post['date']} ({post['sentiment']})"):
                    st.write(f"Caption: {post['caption'][:200]}..." if len(post['caption']) > 200 else f"Caption: {post['caption']}")
                    st.write(f"Sentiment: {post['sentiment']} (Confidence: {post['sentiment_score']*100:.1f}%)")
                    st.write(f"Likes: {post['likes']}")
                    st.write(f"URL: {post['url']}")

                    if post.get('risk_keywords'):
                        st.write(f"Risk Keywords: {', '.join(post['risk_keywords'])}")

    # Add a button to analyze another account
    if st.button("Analyze Another Account"):
        st.session_state.profile_analysis = None
        st.session_state.posts_analysis = []
        st.experimental_rerun()

if __name__ == "__main__":
    # Check if running in Streamlit
    if 'streamlit' in sys.modules:
        streamlit_app()
    else:
        main()
