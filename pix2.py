import instaloader
import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def download_profile_posts(username, save_folder=None, post_limit=5):
    """
    Download recent posts from a public Instagram profile
    
    Args:
        username (str): Instagram username
        save_folder (str): Base path to save the data
        post_limit (int): Number of recent posts to download
    
    Returns:
        str: Status message
    """
    try:
        if save_folder is None:
            save_folder = os.path.join(str(Path.home()), "Pictures", "Instagram_Downloads")
        
        # Create user-specific folder
        user_folder = os.path.join(save_folder, username)
        posts_folder = os.path.join(user_folder, 'posts')
        os.makedirs(posts_folder, exist_ok=True)
        
        # Initialize Instaloader
        L = instaloader.Instaloader(
            download_pictures=True,
            download_videos=True,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=True,
            compress_json=False
        )
        
        # Ask for Instagram credentials
        print("\nInstagram login required to download posts")
        username_login = input("Enter your Instagram username: ")
        password = input("Enter your Instagram password: ")
        
        try:
            L.login(username_login, password)
            print("Login successful!")
        except Exception as e:
            return f"Login failed: {str(e)}"
        
        # Get profile and posts
        profile = instaloader.Profile.from_username(L.context, username)
        
        if profile.mediacount > 0:
            print(f"\nDownloading up to {post_limit} recent posts...")
            posts_data = []
            
            for index, post in enumerate(profile.get_posts()):
                if index >= post_limit:
                    break
                
                try:
                    print(f"\nDownloading post {index + 1}/{post_limit}")
                    
                    # Create post-specific folder
                    post_date = post.date_local.strftime("%Y%m%d")
                    post_folder = os.path.join(posts_folder, f"post_{post_date}_{post.shortcode}")
                    os.makedirs(post_folder, exist_ok=True)
                    
                    # Download the post
                    L.download_post(post, target=post_folder)
                    
                    # Save post information
                    post_info = {
                        "shortcode": post.shortcode,
                        "date": post.date_local.strftime("%Y-%m-%d %H:%M:%S"),
                        "caption": post.caption if post.caption else "",
                        "likes": post.likes,
                        "is_video": post.is_video,
                        "url": f"https://www.instagram.com/p/{post.shortcode}/",
                    }
                    
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
            
            return f"""Downloads completed successfully:
- Posts saved to: {posts_folder}
- Total posts downloaded: {len(posts_data)}"""
        else:
            return "This profile has no posts to download."
                
    except instaloader.exceptions.ProfileNotExistsException:
        return f"Error: Profile '{username}' does not exist"
    except Exception as e:
        return f"Error occurred: {str(e)}"

def main():
    username = input("Enter Instagram username to download posts from: ")
    custom_location = input("Enter custom save location (press Enter to use default Pictures folder): ").strip()
    post_limit = input("Enter number of recent posts to download (default is 5): ").strip()
    
    save_folder = custom_location if custom_location else None
    post_limit = int(post_limit) if post_limit.isdigit() else 5
    
    result = download_profile_posts(username, save_folder, post_limit)
    print("\n" + result)

if __name__ == "__main__":
    main()
