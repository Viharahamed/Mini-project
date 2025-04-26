import json
import os.path
from datetime import datetime
import instaloader
import sys
import time
import random
import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
import re
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

try:
    import getpass
except ImportError:
    getpass = None

def masked_input(prompt: str) -> str:
    """
    Cross-platform password input with masked characters
    (stars displayed for each entered character).
    """
    password = ""
    if sys.platform == "win32":
        import msvcrt
        print(prompt, end='', flush=True)
        while True:
            char = msvcrt.getch()
            if char == b'\r':
                print()
                break
            elif char == b'\x08':
                if len(password) > 0:
                    password = password[:-1]
                    sys.stdout.write('\b \b')
                    sys.stdout.flush()
            else:
                password += char.decode('utf-8')
                sys.stdout.write('*')
                sys.stdout.flush()
    else:
        password = getpass.getpass(prompt)
    return password

def create_sentiment_analyzer():
    """Create a sentiment analysis pipeline using transformers library"""
    try:
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt"
        )
    except Exception as e:
        print(f"Error initializing primary analyzer: {str(e)}")
        try:
            return pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                framework="pt"
            )
        except Exception as e:
            print(f"Error initializing secondary analyzer: {str(e)}")
            return None

def analyze_text_sentiment(text, analyzer):
    """Analyze sentiment and toxicity of a single text"""
    if not text or len(text.strip()) == 0:
        return {"label": "NEUTRAL", "score": 0.5, "risk_score": 0.0}

    try:
        if analyzer:
            result = analyzer(text)
            if isinstance(result, list) and isinstance(result[0], dict):
                sentiment_scores = result[0]
                if 'label' in sentiment_scores:
                    return {
                        "label": sentiment_scores['label'].upper(),
                        "score": sentiment_scores['score'],
                        "risk_score": 1 - sentiment_scores['score'] if 'POSITIVE' in sentiment_scores['label'].upper() else sentiment_scores['score'],
                        "detailed_scores": result
                    }
                else:
                    max_sentiment = max(sentiment_scores, key=lambda x: x['score'])
                    return {
                        "label": max_sentiment['label'].upper(),
                        "score": max_sentiment['score'],
                        "risk_score": max_sentiment['score'] if 'NEGATIVE' in max_sentiment['label'].upper() else 0.0,
                        "detailed_scores": sentiment_scores
                    }
        return fallback_sentiment_analysis(text)
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return fallback_sentiment_analysis(text)

def create_image_analyzer():
    """Initialize the image analysis model"""
    try:
        model = ResNet50(weights='imagenet', include_top=True)
        return model
    except Exception as e:
        print(f"Error initializing image analyzer: {str(e)}")
        return None

def analyze_profile_picture(image_path, model):
    """
    Analyze profile picture for potential risks
    
    Args:
        image_path: URL or local path to the profile picture
        model: Pre-trained image analysis model
    
    Returns:
        dict: Analysis results including risk score and detected features
    """
    try:
        if isinstance(image_path, str) and image_path.startswith('http'):
            response = requests.get(image_path)
            if response.status_code != 200:
                return {"risk_score": 0.0, "error": "Failed to download image"}
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize and preprocess
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Get predictions
        predictions = model.predict(img_array)
        from tensorflow.keras.applications.resnet50 import decode_predictions
        decoded_predictions = decode_predictions(predictions, top=5)[0]

        # Define risky categories and their risk weights
        risk_categories = {
            'weapon': 0.9,
            'knife': 0.8,
            'gun': 0.9,
            'drugs': 0.8,
            'alcohol': 0.6,
            'cigarette': 0.6,
            'violence': 0.8,
            'blood': 0.7,
            'danger': 0.7,
            'terrorism':0.9,
            'abuse':0.7,
            'hate':0.6,
            'harm':0.9,
            'illegal':0.9,
            'drug':0.9,
            'suicide':0.9,
            'harm':0.9,
            'attack':0.8,
            'revenge':0.7,
            'extremist':0.4,
            'radical':0.6,
            

        }

        # Calculate risk score based on detected categories
        max_risk_score = 0.0
        detected_categories = {}

        for _, category, confidence in decoded_predictions:
            category = category.lower()
            for risk_key, risk_weight in risk_categories.items():
                if risk_key in category:
                    risk_score = confidence * risk_weight
                    max_risk_score = max(max_risk_score, risk_score)
                    detected_categories[category] = confidence

        return {
            "risk_score": max_risk_score,
            "detected_categories": detected_categories,
            "raw_predictions": [(cat, float(conf)) for _, cat, conf in decoded_predictions]
        }

    except Exception as e:
        print(f"Error analyzing profile picture: {str(e)}")
        return {"risk_score": 0.0, "error": str(e)}

def analyze_instagram_profile(username, analyzer, insta_context, retry_count=3, retry_delay=5):
    """
    Analyze an Instagram profile for sentiment risks with retry mechanism

    Args:
        username (str): Instagram username
        analyzer: Sentiment analysis pipeline
        insta_context: Instaloader context
        retry_count (int): Number of retries on failure
        retry_delay (int): Base delay between retries in seconds

    Returns:
        dict: Profile analysis results
    """
    for attempt in range(retry_count):
        try:
            profile = instaloader.Profile.from_username(insta_context, username)
            profile_data = {
                "username": username,
                "fullname": profile.full_name,
                "bio": profile.biography,
                "followers": profile.followers,
                "following": profile.followees,
                "is_private": profile.is_private,
                "external_url": profile.external_url,
                "business_category": profile.business_category_name,
                "profile_pic_url": profile.profile_pic_url
            }

            # Text analysis
            username_sentiment = analyze_text_sentiment(username, analyzer)
            fullname_sentiment = analyze_text_sentiment(profile.full_name, analyzer)
            bio_sentiment = analyze_text_sentiment(profile.biography, analyzer)

            # Profile picture analysis
            image_analyzer = create_image_analyzer()
            if image_analyzer and profile.profile_pic_url:
                pfp_analysis = analyze_profile_picture(profile.profile_pic_url, image_analyzer)
            else:
                pfp_analysis = {"risk_score": 0.0, "error": "Image analysis not available"}

            # Calculate risk scores with new weights
            text_risk_scores = {
                "username": username_sentiment.get("risk_score", 0.0),
                "fullname": fullname_sentiment.get("risk_score", 0.0),
                "bio": bio_sentiment.get("risk_score", 0.0),
                "profile_picture": pfp_analysis.get("risk_score", 0.0),
                "detailed_scores": {
                    "username": username_sentiment.get("detailed_scores", []),
                    "fullname": fullname_sentiment.get("detailed_scores", []),
                    "bio": bio_sentiment.get("detailed_scores", []),
                    "profile_picture": pfp_analysis.get("detected_categories", {})
                }
            }

            # Calculate weighted risk score
            # Weights: Username (15%), Full Name (15%), Bio (40%), Profile Picture (30%)
            weighted_risk = (
                text_risk_scores["username"] * 0.15 +
                text_risk_scores["fullname"] * 0.15 +
                text_risk_scores["bio"] * 0.40 +
                text_risk_scores["profile_picture"] * 0.30
            )

            # Additional risk factors from bio
            additional_risk_factors = detect_risk_keywords(profile.biography)
            keyword_risk = len(additional_risk_factors) * 0.1
            final_risk = min(1.0, weighted_risk + keyword_risk)

            return {
                "profile_data": profile_data,
                "sentiment_analysis": {
                    "username": username_sentiment,
                    "fullname": fullname_sentiment,
                    "bio": bio_sentiment,
                },
                "profile_picture_analysis": pfp_analysis,
                "risk_assessment": {
                    "individual_risks": text_risk_scores,
                    "overall_risk": final_risk,
                    "risk_level": get_risk_level(final_risk),
                    "risk_keywords_detected": additional_risk_factors,
                }
            }
        except instaloader.exceptions.ConnectionException as e:
            if "429" in str(e):
                wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Waiting for {wait_time:.1f} seconds before retry {attempt+1}/{retry_count}")
                time.sleep(wait_time)
            elif "401" in str(e):
                print("Authentication error. Might need to wait or use a different account.")
                if attempt < retry_count - 1:
                    wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Waiting for {wait_time:.1f} seconds before retry {attempt+1}/{retry_count}")
                    time.sleep(wait_time)
            else:
                print(f"Connection error: {str(e)}")
                if attempt < retry_count - 1:
                    wait_time = retry_delay * (attempt + 1) + random.uniform(0, 1)
                    print(f"Waiting for {wait_time:.1f} seconds before retry {attempt+1}/{retry_count}")
                    time.sleep(wait_time)
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

def download_profile_posts(username, save_folder=None, post_limit=5, analyzer=None, username_login=None, password=None, retry_count=3):
    """
    Download recent posts from a public Instagram profile and analyze sentiments

    Args:
        username (str): Instagram username
        save_folder (str): Base path to save the data
        post_limit (int): Number of recent posts to download
        analyzer: Sentiment analysis pipeline
        username_login (str): Instagram login username (optional)
        password (str): Instagram login password (optional)
        retry_count (int): Number of retries on failure

    Returns:
        dict: Status message and analysis results
    """
    save_folder = r"C:\Users\vihar\Downloads\Mini pro"
    try:
        user_folder = os.path.join(save_folder, username)
        posts_folder = os.path.join(user_folder, 'posts')
        analysis_folder = os.path.join(user_folder, 'analysis')
        os.makedirs(posts_folder, exist_ok=True)
        os.makedirs(analysis_folder, exist_ok=True)
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
            request_timeout=60
        )
        if username_login is None or password is None:
            print("\nInstagram login required to download posts")
            username_login = input("Enter your Instagram username: ")
            password = masked_input("Enter your Instagram password: ")
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
        profile_analysis = analyze_instagram_profile(username, analyzer, L.context, retry_count)
        if "error" in profile_analysis:
            return {"status": profile_analysis["error"]}
        profile_analysis_file = os.path.join(analysis_folder, 'profile_analysis.json')
        with open(profile_analysis_file, 'w', encoding='utf-8') as f:
            json.dump(profile_analysis, f, indent=4, ensure_ascii=False)
        try:
            profile = instaloader.Profile.from_username(L.context, username)
        except Exception as e:
            return {"status": f"Error retrieving profile: {str(e)}", "profile_analysis": profile_analysis}
        if profile.mediacount > 0:
            print(f"\nDownloading up to {post_limit} recent posts...")
            posts_data = []
            analyzed_posts = []
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
                    post_date = post.date_local.strftime("%Y%m%d")
                    post_folder = os.path.join(posts_folder, f"post_{post_date}_{post.shortcode}")
                    os.makedirs(post_folder, exist_ok=True)
                    if index > 0:
                        time.sleep(random.uniform(1.0, 3.0))
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
                    post_info = {
                        "shortcode": post.shortcode,
                        "date": post.date_local.strftime("%Y-%m-%d %H:%M:%S"),
                        "caption": post.caption if post.caption else "",
                        "likes": post.likes,
                        "is_video": post.is_video,
                        "url": f"https://www.instagram.com/p/{post.shortcode}/",
                    }
                    if analyzer is not None:
                        analyzed_post = analyze_post_sentiment(post_info, analyzer)
                        analyzed_posts.append(analyzed_post)
                    else:
                        analyzed_posts.append(post_info)
                    info_file = os.path.join(post_folder, 'post_info.json')
                    with open(info_file, 'w', encoding='utf-8') as f:
                        json.dump(post_info, f, indent=4, ensure_ascii=False)
                    posts_data.append(post_info)
                except Exception as e:
                    print(f"Error downloading post {index + 1}: {str(e)}")
                    continue
            summary_file = os.path.join(posts_folder, 'posts_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(posts_data, f, indent=4, ensure_ascii=False)
            analysis_file = os.path.join(analysis_folder, 'posts_sentiment_analysis.json')
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analyzed_posts, f, indent=4, ensure_ascii=False)
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

def fallback_sentiment_analysis(text):
    """
    Simple rule-based sentiment analysis as fallback when transformer model fails
    """
    if not text or len(text.strip()) == 0:
        return {"label": "NEUTRAL", "score": 0.5, "risk_score": 0.0}
    positive_words = ["good", "great", "excellent", "awesome", "happy", "love", "best", "amazing",
                     "wonderful", "beautiful", "enjoy", "nice", "perfect", "exciting", "joy", "positive"]
    negative_words = ["bad", "terrible", "awful", "horrible", "hate", "sad", "worst", "disappointing",
                     "poor", "negative", "anger", "angry", "annoying", "disaster", "failure", "blood"]
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower.split())
    neg_count = sum(1 for word in negative_words if word in text_lower.split())
    if pos_count > neg_count:
        score = 0.5 + min(0.49, (pos_count - neg_count) * 0.1)
        return {
            "label": "POSITIVE",
            "score": score,
            "risk_score": 1 - score,
            "detailed_scores": []
        }
    elif neg_count > pos_count:
        score = 0.5 + min(0.49, (neg_count - pos_count) * 0.1)
        return {
            "label": "NEGATIVE",
            "score": score,
            "risk_score": score,
            "detailed_scores": []
        }
    else:
        return {
            "label": "NEUTRAL",
            "score": 0.5,
            "risk_score": 0.25,
            "detailed_scores": []
        }

def log_profile_risk(username, risk_score, risk_level, log_file="log.json"):
    """
    Log profile risk assessment to JSON file

    Args:
        username (str): Instagram username
        risk_score (float): Calculated risk score (0-1)
        risk_level (str): Risk level (Low/Medium/High)
        log_file (str): Path to the log file

    Returns:
        bool: True if logging was successful
    """
    try:
        log_entry = {
            "username": username,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            with open(log_file, 'r', encoding='utf-8') as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = {"profiles": []}
        else:
            logs = {"profiles": []}
        usernames = [profile["username"] for profile in logs["profiles"]]
        if username in usernames:
            for profile in logs["profiles"]:
                if profile["username"] == username:
                    profile["risk_score"] = risk_score
                    profile["risk_level"] = risk_level
                    profile["timestamp"] = log_entry["timestamp"]
        else:
            logs["profiles"].append(log_entry)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=4)
        return True
    except Exception as e:
        print(f"Error logging profile risk: {str(e)}")
        return False

def display_priority_log(log_file="log.json"):
    """
    Display priority log sorted by risk score with enhanced visualization and delete functionality

    Args:
        log_file (str): Path to the log file
    """
    try:
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            with open(log_file, 'r', encoding='utf-8') as f:
                try:
                    logs = json.load(f)
                    if not isinstance(logs, dict):
                        logs = {"profiles": []}
                    if "profiles" not in logs:
                        logs["profiles"] = []
                except json.JSONDecodeError:
                    logs = {"profiles": []}
        else:
            logs = {"profiles": []}
            
        sorted_profiles = sorted(
            logs["profiles"],
            key=lambda x: x["risk_score"],
            reverse=True
        )
        
        if sorted_profiles:
            # Create DataFrame with additional columns
            df = pd.DataFrame(sorted_profiles)
            
            # Add priority indicator column
            def get_priority_indicator(risk_level):
                if risk_level == 'High':
                    return '🔴'  # Red circle
                elif risk_level == 'Medium':
                    return '🟡'  # Yellow circle
                else:
                    return '🟢'  # Green circle
            
            df.insert(0, 'Priority', df['risk_level'].apply(get_priority_indicator))
            
            # Display the DataFrame with custom styling
            st.dataframe(
                df[['Priority', 'username', 'risk_score', 'risk_level', 'timestamp']],
                column_config={
                    'Priority': st.column_config.TextColumn(
                        'Priority',
                        help='Risk level indicator',
                        width='small'
                    ),
                    'username': st.column_config.TextColumn(
                        'Username',
                        help='Instagram username'
                    ),
                    'risk_score': st.column_config.NumberColumn(
                        'Risk Score',
                        help='Calculated risk score (0-1)',
                        format='%.2f'
                    ),
                    'risk_level': st.column_config.TextColumn(
                        'Risk Level',
                        help='Risk level category'
                    ),
                    'timestamp': st.column_config.DatetimeColumn(
                        'Timestamp',
                        help='Analysis timestamp'
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Add individual delete buttons for each entry
            st.subheader("Delete Entries")
            for idx, profile in enumerate(sorted_profiles):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"Username: {profile['username']} (Risk Level: {profile['risk_level']})")
                with col2:
                    st.write(f"Score: {profile['risk_score']:.2f}")
                with col3:
                    if st.button("Delete", key=f"delete_{idx}"):
                        # Remove the entry from the logs
                        logs["profiles"] = [p for p in logs["profiles"] if p["username"] != profile["username"]]
                        
                        # Save updated logs to file
                        with open(log_file, 'w', encoding='utf-8') as f:
                            json.dump(logs, f, indent=4)
                        
                        st.success(f"Entry for {profile['username']} deleted successfully!")
                        st.experimental_rerun()
            
            # Add download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="instagram_risk_log.csv",
                mime="text/csv",
            )
        else:
            st.info("No profiles have been analyzed yet.")
    except Exception as e:
        st.error(f"Error displaying priority log: {str(e)}")

def streamlit_app():
    st.set_page_config(page_title="Instagram Profile Risk Analysis", layout="wide")
    st.title("Instagram Profile Risk Analysis Dashboard")
    st.write("Analyze Instagram profiles and posts for sentiment and potential risk factors")
    analyzer = create_sentiment_analyzer()
    if 'profile_analysis' not in st.session_state:
        st.session_state.profile_analysis = None
    if 'posts_analysis' not in st.session_state:
        st.session_state.posts_analysis = []
    if 'login_status' not in st.session_state:
        st.session_state.login_status = None
    if 'show_priority_log' not in st.session_state:
        st.session_state.show_priority_log = False
    if 'selected_rows' not in st.session_state:
        st.session_state.selected_rows = []
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
        col1, col2 = st.columns(2)
        with col1:
            analyze_button = st.button("Analyze Profile")
        with col2:
            priority_log_button = st.button("Priority Log")
        if priority_log_button:
            st.session_state.show_priority_log = True
    if st.session_state.show_priority_log:
        st.header("Priority Risk Log")
        st.subheader("Profiles sorted by risk score (highest risk first)")
        if st.button("Close Priority Log"):
            st.session_state.show_priority_log = False
        display_priority_log()
    elif analyze_button:
        if not username_login or not password or not target_username:
            st.error("Please provide all required information")
        else:
            with st.spinner(f"Analyzing Instagram profile: @{target_username}"):
                temp_dir = os.path.join(os.path.expanduser("~"), "temp_instagram_analysis")
                try:
                    result = download_profile_posts(
                        target_username,
                        save_folder=temp_dir,
                        post_limit=post_limit,
                        analyzer=analyzer,
                        username_login=username_login,
                        password=password,
                        retry_count=retry_count
                    )
                    if "profile_analysis" in result and "error" not in result["profile_analysis"]:
                        st.session_state.profile_analysis = result["profile_analysis"]
                        st.session_state.posts_analysis = result.get("posts_analysis", [])
                        risk_assessment = result["profile_analysis"]["risk_assessment"]
                        log_success = log_profile_risk(
                            target_username,
                            risk_assessment["overall_risk"],
                            risk_assessment["risk_level"]
                        )
                        st.success("Analysis completed and logged!")
                    else:
                        st.error(result["status"])
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
    if not st.session_state.show_priority_log and st.session_state.profile_analysis:
        profile_analysis = st.session_state.profile_analysis
        col1, col2 = st.columns(2)
        with col1:
            st.header(f"Profile: @{profile_analysis['profile_data']['username']}")
            
            # Display profile picture and analysis
            if profile_analysis['profile_data'].get('profile_pic_url'):
                try:
                    # Download and display the profile picture
                    response = requests.get(profile_analysis['profile_data']['profile_pic_url'])
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        st.image(img, width=200, caption=f"@{profile_analysis['profile_data']['username']}'s Profile Picture")
                        
                        # Display profile picture analysis if available
                        if 'profile_picture_analysis' in profile_analysis:
                            pfp_analysis = profile_analysis['profile_picture_analysis']
                            st.subheader("Profile Picture Analysis")
                            
                            # Display risk score with color coding
                            risk_score = pfp_analysis.get('risk_score', 0.0)
                            risk_color = "green" if risk_score < 0.3 else "orange" if risk_score < 0.6 else "red"
                            st.markdown(f"Risk Score: <span style='color:{risk_color}'>{risk_score:.2f}</span>", unsafe_allow_html=True)
                            
                            # Display detected categories if any
                            if pfp_analysis.get('detected_categories'):
                                st.write("Detected Categories:")
                                for category, confidence in pfp_analysis['detected_categories'].items():
                                    st.write(f"- {category.title()}: {confidence:.2%}")
                    else:
                        st.warning("Could not load profile picture")
                except Exception as e:
                    st.warning(f"Error loading profile picture: {str(e)}")
            
            st.write(f"Full Name: {profile_analysis['profile_data']['fullname']}")
            st.write(f"Followers: {profile_analysis['profile_data']['followers']:,}")
            st.write(f"Following: {profile_analysis['profile_data']['following']:,}")
            st.write(f"Private: {'Yes' if profile_analysis['profile_data']['is_private'] else 'No'}")
            st.subheader("Biography")
            st.write(profile_analysis['profile_data']['bio'])
        with col2:
            st.header("Profile Risk Assessment")
            risk_level = profile_analysis['risk_assessment']['risk_level']
            risk_color = "green" if risk_level == "Low" else "orange" if risk_level == "Medium" else "red"
            st.markdown(f"<h3 style='color:{risk_color}'>Risk Level: {risk_level}</h3>", unsafe_allow_html=True)
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
            if profile_analysis['risk_assessment']['risk_keywords_detected']:
                st.subheader("Risk Keywords Detected")
                st.write(", ".join(profile_analysis['risk_assessment']['risk_keywords_detected']))
        posts_analysis = st.session_state.posts_analysis
        if posts_analysis:
            st.header("Posts Analysis")
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
            st.subheader("Individual Post Analysis")
            for idx, post in enumerate(posts_analysis):
                with st.expander(f"Post {idx+1} - {post['date']} ({post['sentiment']})"):
                    st.write(f"Caption: {post['caption'][:200]}..." if len(post['caption']) > 200 else f"Caption: {post['caption']}")
                    st.write(f"Sentiment: {post['sentiment']} (Confidence: {post['sentiment_score']*100:.1f}%)")
                    st.write(f"Likes: {post['likes']}")
                    st.write(f"URL: {post['url']}")
                    if post.get('risk_keywords'):
                        st.write(f"Risk Keywords: {', '.join(post['risk_keywords'])}")

if __name__ == "__main__":
    if 'streamlit' in sys.modules:
        streamlit_app()
    else:
        main()
