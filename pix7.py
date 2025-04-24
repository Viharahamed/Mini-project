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
from textblob import TextBlob
import numpy as np

try:
    import getpass
except ImportError:
    getpass = None

def masked_input(prompt: str) -> str:
    """Cross-platform password input with masked characters"""
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
    """Create a more comprehensive sentiment analysis pipeline"""
    try:
        return pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
    except Exception as e:
        print(f"Error initializing sentiment analyzer: {str(e)}")
        return None

def create_instagram_session(username, password, retry_count=3):
    """Create an authenticated Instagram session"""
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

    for attempt in range(retry_count):
        try:
            # Try to load session from cache first
            session_file = f"{username}_instagram_session"
            try:
                L.load_session_from_file(username, session_file)
                print("Loaded session from cache")
                return L
            except FileNotFoundError:
                pass

            # If no cached session, perform login
            L.login(username, password)
            
            # Save session for future use
            L.save_session_to_file(session_file)
            
            print("Login successful!")
            return L

        except instaloader.exceptions.ConnectionException as e:
            if "429" in str(e):  # Rate limiting
                wait_time = 5 * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            elif "401" in str(e):  # Authentication error
                print("Authentication failed. Please check your credentials.")
                break
            else:
                print(f"Connection error: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(5)
        except Exception as e:
            print(f"Login error: {str(e)}")
            if attempt < retry_count - 1:
                time.sleep(5)

    raise Exception("Failed to create Instagram session")

def analyze_text_patterns(text):
    """Analyze text for concerning patterns dynamically"""
    if not text:
        return {}

    patterns = {
        'aggressive_language': r'\b(threat|kill|attack|fight|destroy|violent|angry|aggressive)\b',
        'hate_speech': r'\b(hate|racist|discriminat\w+|bigot|prejudice)\b',
        'suspicious_activities': r'\b(illegal|drug|weapon|scam|fraud|hack)\b',
        'emotional_distress': r'\b(suicide|depressed|anxiety|hurt|sad|lonely|desperate)\b',
        'spam_patterns': r'(\$\d+|buy now|dm for|check link|follow me|check bio)',
        'excessive_symbols': r'([!?]{3,}|\$+|@{2,})',
        'inappropriate_content': r'\b(nsfw|adult|xxx|porn|sex)\b'
    }
    
    detected_patterns = {}
    text_lower = text.lower()
    
    for pattern_type, pattern in patterns.items():
        matches = re.findall(pattern, text_lower)
        if matches:
            detected_patterns[pattern_type] = list(set(matches))
    
    return detected_patterns

def calculate_dynamic_risk_score(profile_data, pattern_analysis, sentiment_scores):
    """Calculate comprehensive risk score based on multiple factors"""
    risk_factors = {
        'profile_completeness': 0,
        'engagement_ratio': 0,
        'content_risk': 0,
        'sentiment_risk': 0,
        'pattern_risk': 0
    }
    
    # Profile completeness (inverse - less complete = higher risk)
    completeness = 0
    if profile_data.get('bio'): completeness += 0.3
    if profile_data.get('external_url'): completeness += 0.2
    if profile_data.get('is_verified'): completeness += 0.5
    risk_factors['profile_completeness'] = 1 - completeness

    # Engagement ratio analysis
    followers = profile_data.get('followers', 0)
    following = profile_data.get('following', 0)
    if followers > 0:
        ratio = following / followers
        risk_factors['engagement_ratio'] = min(1.0, ratio / 10)
    
    # Content risk from pattern analysis
    pattern_count = sum(len(patterns) for patterns in pattern_analysis.values())
    risk_factors['pattern_risk'] = min(1.0, pattern_count / 5)
    
    # Sentiment risk
    negative_sentiment_count = sum(1 for score in sentiment_scores if score < 0.4)
    risk_factors['sentiment_risk'] = negative_sentiment_count / len(sentiment_scores) if sentiment_scores else 0
    
    # Calculate weighted risk score
    weights = {
        'profile_completeness': 0.15,
        'engagement_ratio': 0.25,
        'pattern_risk': 0.35,
        'sentiment_risk': 0.25
    }
    
    total_risk = sum(score * weights[factor] for factor, score in risk_factors.items())
    
    return total_risk, risk_factors

def analyze_text_sentiment(text, analyzer):
    """Enhanced sentiment analysis with fallback"""
    if not text or len(text.strip()) == 0:
        return {"label": "NEUTRAL", "score": 0.5}

    try:
        if analyzer:
            result = analyzer(text)[0]
            score = result['score']
            if result['label'] == 'NEGATIVE':
                score = 1 - score
            return {"label": result['label'], "score": score}
        else:
            blob = TextBlob(text)
            score = (blob.sentiment.polarity + 1) / 2
            label = "POSITIVE" if score > 0.6 else "NEGATIVE" if score < 0.4 else "NEUTRAL"
            return {"label": label, "score": score}
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return {"label": "NEUTRAL", "score": 0.5}
    
def analyze_instagram_profile(username, analyzer, insta_context, retry_count=3, retry_delay=5):
    """Enhanced profile analysis with dynamic risk assessment"""
    for attempt in range(retry_count):
        try:
            time.sleep(random.uniform(1, 2))  # Add delay to avoid rate limiting
            profile = instaloader.Profile.from_username(insta_context, username)
            
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
                "is_verified": profile.is_verified,
                "media_count": profile.mediacount
            }

            # Analyze patterns
            bio_patterns = analyze_text_patterns(profile.biography)
            username_patterns = analyze_text_patterns(username)
            
            # Sentiment analysis
            sentiment_results = {
                "username": analyze_text_sentiment(username, analyzer),
                "fullname": analyze_text_sentiment(profile.full_name, analyzer),
                "bio": analyze_text_sentiment(profile.biography, analyzer)
            }
            
            # Calculate risk score
            sentiment_scores = [result['score'] for result in sentiment_results.values()]
            risk_score, risk_factors = calculate_dynamic_risk_score(
                profile_data, 
                bio_patterns,
                sentiment_scores
            )

            return {
                "profile_data": profile_data,
                "pattern_analysis": {
                    "bio_patterns": bio_patterns,
                    "username_patterns": username_patterns
                },
                "sentiment_analysis": sentiment_results,
                "risk_assessment": {
                    "risk_score": risk_score,
                    "risk_factors": risk_factors,
                    "risk_level": get_risk_level(risk_score),
                    "detected_patterns": bio_patterns
                }
            }

        except instaloader.exceptions.ConnectionException as e:
            if attempt < retry_count - 1:
                wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying... ({attempt + 1}/{retry_count})")
                time.sleep(wait_time)
            else:
                return {"error": f"Failed to analyze profile: {str(e)}"}
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(retry_delay)
            else:
                return {"error": f"Failed to analyze profile: {str(e)}"}

    return {"error": "Failed to analyze profile after multiple attempts"}

def get_risk_level(risk_score):
    """Enhanced risk level determination with detailed description"""
    if risk_score < 0.3:
        return {
            "level": "Low",
            "description": "No significant risk factors detected",
            "color": "green",
            "recommendations": "Regular monitoring recommended"
        }
    elif risk_score < 0.6:
        return {
            "level": "Medium",
            "description": "Some concerning patterns detected",
            "color": "orange",
            "recommendations": "Increased monitoring and further investigation recommended"
        }
    else:
        return {
            "level": "High",
            "description": "Multiple risk factors present",
            "color": "red",
            "recommendations": "Immediate attention and detailed investigation required"
        }

def analyze_post_sentiment(post_data, analyzer):
    """Enhanced post sentiment analysis"""
    caption = post_data.get("caption", "")
    
    sentiment_result = analyze_text_sentiment(caption, analyzer)
    patterns = analyze_text_patterns(caption)
    
    result = post_data.copy()
    result["sentiment"] = sentiment_result["label"]
    result["sentiment_score"] = sentiment_result["score"]
    result["detected_patterns"] = patterns
    result["risk_level"] = "High" if patterns else "Low"
    
    return result

def download_profile_posts(username, save_folder=None, post_limit=5, analyzer=None, username_login=None, password=None, retry_count=3):
    """Enhanced profile and post analysis with robust authentication"""
    save_folder = save_folder or os.path.join(os.path.expanduser("~"), "instagram_analysis")
    
    try:
        # Create folders
        user_folder = os.path.join(save_folder, username)
        posts_folder = os.path.join(user_folder, 'posts')
        analysis_folder = os.path.join(user_folder, 'analysis')
        os.makedirs(posts_folder, exist_ok=True)
        os.makedirs(analysis_folder, exist_ok=True)

        # Get Instagram session
        try:
            L = create_instagram_session(username_login, password, retry_count)
        except Exception as e:
            return {"status": f"Authentication failed: {str(e)}"}

        # Analyze profile with delay to avoid rate limiting
        time.sleep(random.uniform(1, 2))
        profile_analysis = analyze_instagram_profile(username, analyzer, L.context, retry_count)
        
        if "error" in profile_analysis:
            return {"status": profile_analysis["error"]}

        # Save profile analysis
        with open(os.path.join(analysis_folder, 'profile_analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(profile_analysis, f, indent=4, ensure_ascii=False)

        # Get posts with rate limiting protection
        try:
            profile = instaloader.Profile.from_username(L.context, username)
            posts_data = []
            analyzed_posts = []

            for index, post in enumerate(profile.get_posts()):
                if index >= post_limit:
                    break

                try:
                    # Add delay between post downloads
                    if index > 0:
                        time.sleep(random.uniform(1, 2))

                    post_info = {
                        "shortcode": post.shortcode,
                        "date": post.date_local.strftime("%Y-%m-%d %H:%M:%S"),
                        "caption": post.caption if post.caption else "",
                        "likes": post.likes,
                        "comments": post.comments,
                        "is_video": post.is_video,
                        "url": f"https://www.instagram.com/p/{post.shortcode}/",
                    }

                    # Analyze post
                    analyzed_post = analyze_post_sentiment(post_info, analyzer)
                    analyzed_posts.append(analyzed_post)
                    posts_data.append(post_info)

                    # Save post data
                    post_folder = os.path.join(posts_folder, f"post_{post.date_local.strftime('%Y%m%d')}_{post.shortcode}")
                    os.makedirs(post_folder, exist_ok=True)
                    
                    with open(os.path.join(post_folder, 'post_analysis.json'), 'w', encoding='utf-8') as f:
                        json.dump(analyzed_post, f, indent=4, ensure_ascii=False)

                except Exception as e:
                    print(f"Error analyzing post {index + 1}: {str(e)}")
                    continue

            # Calculate and save overall analysis
            analysis_results = {
                "profile_analysis": profile_analysis,
                "posts_analysis": analyzed_posts,
                "summary": calculate_profile_summary(profile_analysis, analyzed_posts)
            }

            with open(os.path.join(analysis_folder, 'complete_analysis.json'), 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=4, ensure_ascii=False)

            return analysis_results

        except Exception as e:
            return {"status": f"Error analyzing posts: {str(e)}", "profile_analysis": profile_analysis}

    except Exception as e:
        return {"status": f"Error: {str(e)}"}

def calculate_profile_summary(profile_analysis, posts_analysis):
    """Calculate overall profile summary and risk assessment"""
    summary = {
        "total_posts_analyzed": len(posts_analysis),
        "sentiment_distribution": {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0},
        "risk_patterns_found": {},
        "overall_risk_score": profile_analysis["risk_assessment"]["risk_score"],
        "risk_level": profile_analysis["risk_assessment"]["risk_level"]
    }

    for post in posts_analysis:
        summary["sentiment_distribution"][post["sentiment"]] += 1
        
        for pattern_type, patterns in post.get("detected_patterns", {}).items():
            if pattern_type not in summary["risk_patterns_found"]:
                summary["risk_patterns_found"][pattern_type] = set()
            summary["risk_patterns_found"][pattern_type].update(patterns)

    summary["risk_patterns_found"] = {k: list(v) for k, v in summary["risk_patterns_found"].items()}
    return summary

def streamlit_app():
    st.set_page_config(page_title="Instagram Profile Risk Analysis", layout="wide")
    st.title("Instagram Profile Risk Analysis Dashboard")
    st.write("Analyze Instagram profiles and posts for potential risks and sentiment patterns")

    # Initialize sentiment analyzer
    analyzer = create_sentiment_analyzer()

    # Session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'login_status' not in st.session_state:
        st.session_state.login_status = None

    # Sidebar
    with st.sidebar:
        st.header("Instagram Credentials")
        st.warning("Note: Your credentials are used only for API access and are not stored.")
        
        username_login = st.text_input("Your Instagram Username")
        password = st.text_input("Your Instagram Password", type="password")
        
        if st.button("Test Login"):
            try:
                L = create_instagram_session(username_login, password)
                st.session_state.login_status = "success"
                st.success("Login successful!")
            except Exception as e:
                st.session_state.login_status = "failed"
                st.error(f"Login failed: {str(e)}")
        
        st.header("Analysis Settings")
        target_username = st.text_input("Target Instagram Profile to Analyze")
        post_limit = st.number_input("Number of Posts to Analyze", min_value=1, max_value=20, value=5)
        
        analyze_button = st.button("Analyze Profile")

    if analyze_button and target_username and username_login and password:
        with st.spinner(f"Analyzing Instagram profile: @{target_username}"):
            try:
                results = download_profile_posts(
                    target_username,
                    post_limit=post_limit,
                    analyzer=analyzer,
                    username_login=username_login,
                    password=password
                )
                
                if isinstance(results, dict) and "status" in results and "error" in results.get("status", "").lower():
                    st.error(results["status"])
                else:
                    st.session_state.analysis_results = results
                    st.success("Analysis completed successfully!")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Display results
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        profile_analysis = results["profile_analysis"]
        posts_analysis = results.get("posts_analysis", [])

        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Profile Overview")
            profile_data = profile_analysis["profile_data"]
            st.write(f"Username: @{profile_data['username']}")
            st.write(f"Full Name: {profile_data['fullname']}")
            st.write(f"Followers: {profile_data['followers']:,}")
            st.write(f"Following: {profile_data['following']:,}")
            st.write(f"Posts: {profile_data['media_count']:,}")
            st.write(f"Verified: {'Yes' if profile_data['is_verified'] else 'No'}")
            
            if profile_data['bio']:
                st.subheader("Biography")
                st.write(profile_data['bio'])

        with col2:
            st.header("Risk Assessment")
            risk_assessment = profile_analysis["risk_assessment"]
            risk_level = risk_assessment["risk_level"]
            
            st.markdown(
                f"### Risk Level: <span style='color:{risk_level['color']}'>{risk_level['level']}</span>", 
                unsafe_allow_html=True
            )
            st.write(f"**Description:** {risk_level['description']}")
            st.write(f"**Recommendations:** {risk_level['recommendations']}")

            # Risk Factors
            st.subheader("Risk Factors")
            risk_df = pd.DataFrame({
                'Factor': [k.replace('_', ' ').title() for k in risk_assessment["risk_factors"].keys()],
                'Score': list(risk_assessment["risk_factors"].values())
            })
            fig = px.bar(risk_df, x='Factor', y='Score', color='Score',
                        color_continuous_scale=['green', 'yellow', 'red'])
            st.plotly_chart(fig)

        if posts_analysis:
            st.header("Posts Analysis")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Sentiment Distribution")
                sentiment_counts = pd.DataFrame([
                    {'sentiment': post['sentiment'], 'count': 1}
                    for post in posts_analysis
                ]).groupby('sentiment').sum().reset_index()
                
                fig = px.pie(
                    sentiment_counts, 
                    values='count', 
                    names='sentiment',
                    color='sentiment',
                    color_discrete_map={
                        'POSITIVE': 'green',
                        'NEUTRAL': 'gray',
                        'NEGATIVE': 'red'
                    }
                )
                st.plotly_chart(fig)
            
            with col4:
                st.subheader("Risk Patterns in Posts")
                all_patterns = {}
                for post in posts_analysis:
                    for pattern_type, patterns in post.get("detected_patterns", {}).items():
                        if pattern_type not in all_patterns:
                            all_patterns[pattern_type] = set()
                        all_patterns[pattern_type].update(patterns)
                
                if all_patterns:
                    for pattern_type, patterns in all_patterns.items():
                        st.write(f"**{pattern_type.replace('_', ' ').title()}:**")
                        st.write(", ".join(patterns))
                else:
                    st.write("No risk patterns detected in posts")

            # Individual Posts
            st.subheader("Individual Posts Analysis")
            for idx, post in enumerate(posts_analysis):
                with st.expander(f"Post {idx+1} - {post['date']}"):
                    st.write(f"**Caption:** {post['caption'][:200]}..." if len(post['caption']) > 200 
                            else f"**Caption:** {post['caption']}")
                    st.write(f"**Sentiment:** {post['sentiment']} (Score: {post['sentiment_score']:.2f})")
                    st.write(f"**Likes:** {post['likes']}")
                    st.write(f"**URL:** {post['url']}")
                    
                    if post.get("detected_patterns"):
                        st.write("**Detected Patterns:**")
                        for pattern_type, patterns in post["detected_patterns"].items():
                            st.write(f"- {pattern_type.replace('_', ' ').title()}: {', '.join(patterns)}")

def main():
    if 'streamlit' in sys.modules:
        streamlit_app()
    else:
        analyzer = create_sentiment_analyzer()
        username = input("Enter Instagram username to analyze: ")
        username_login = input("Enter your Instagram username: ")
        password = masked_input("Enter your Instagram password: ")
        post_limit = int(input("Enter number of posts to analyze (default 5): ") or 5)
        
        results = download_profile_posts(
            username, 
            post_limit=post_limit, 
            analyzer=analyzer,
            username_login=username_login,
            password=password
        )
        
        if "error" in results:
            print(f"\nError: {results['error']}")
        else:
            print("\nAnalysis completed successfully!")
            risk_level = results["profile_analysis"]["risk_assessment"]["risk_level"]
            print(f"Risk Level: {risk_level['level']}")
            print(f"Description: {risk_level['description']}")
            print(f"Recommendations: {risk_level['recommendations']}")

if __name__ == "__main__":
    main()