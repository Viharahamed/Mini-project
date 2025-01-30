import instaloader
import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import time

def download_profile_posts(username, save_folder=None, post_limit=5):
    """
    Download recent posts from either public or private Instagram profiles
    Automatically follows private accounts if needed
    
    Args:
        username (str): Instagram username to download from
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
        media_folder = os.path.join(user_folder, 'media')
        os.makedirs(media_folder, exist_ok=True)
        
        # Initialize Instaloader with specific settings
        L = instaloader.Instaloader(
            dirname_pattern=media_folder,
            filename_pattern='{date:%Y%m%d}_{shortcode}',
            download_pictures=True,
            download_videos=True,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
            post_metadata_txt_pattern=''
        )
        
        # Login process
        print("\nInstagram login required")
        username_login = input("Enter your Instagram username: ")
        password = input("Enter your Instagram password: ")
        
        try:
            L.login(username_login, password)
            print("Login successful!")
        except instaloader.exceptions.BadCredentialsException:
            return "Error: Invalid login credentials"
        except instaloader.exceptions.TwoFactorAuthRequiredException:
            print("\n2FA detected! Please enter your 2FA code:")
            two_factor_code = input("Enter 2FA code: ")
            try:
                L.two_factor_login(two_factor_code)
                print("2FA authentication successful!")
            except Exception as e:
                return f"2FA authentication failed: {str(e)}"
        except Exception as e:
            return f"Login failed: {str(e)}"
        
        # Get profile and handle private account
        try:
            profile = instaloader.Profile.from_username(L.context, username)
            
            if profile.is_private and not profile.followed_by_viewer:
                print(f"\nProfile '{username}' is private and you're not following them.")
                follow_choice = input("Would you like to follow this account? (y/n): ").lower()
                
                if follow_choice == 'y':
                    try:
                        # Follow the profile
                        L.context.get_and_write_raw(
                            path=f"friendships/{profile.userid}/follow/",
                            params="",
                            post=True
                        )
                        print(f"Successfully followed {username}")
                        
                        # Wait a bit for the follow to take effect
                        print("Waiting for follow request to be processed...")
                        time.sleep(5)
                        
                        # Refresh profile to check if we can now access it
                        profile = instaloader.Profile.from_username(L.context, username)
                        
                        if not profile.followed_by_viewer:
                            return "Follow request sent but pending approval. Please try again after the user accepts your request."
                        
                    except Exception as e:
                        return f"Error following the account: {str(e)}"
                else:
                    return "Download cancelled - private account requires following"
            
        except instaloader.exceptions.ProfileNotExistsException:
            return f"Error: Profile '{username}' does not exist"
        
        if profile.mediacount > 0:
            print(f"\nDownloading up to {post_limit} recent posts...")
            posts_data = []
            
            for index, post in enumerate(profile.get_posts()):
                if index >= post_limit:
                    break
                
                try:
                    print(f"\nDownloading post {index + 1}/{post_limit}")
                    
                    # Download the post using Instaloader
                    L.download_post(post, target=media_folder)
                    
                    # Handle post information
                    post_info = {
                        "shortcode": post.shortcode,
                        "date": post.date_local.strftime("%Y-%m-%d %H:%M:%S"),
                        "caption": post.caption if post.caption else "",
                        "likes": post.likes,
                        "comments": post.comments,
                        "is_video": post.is_video,
                        "media_type": "carousel" if post.typename == "GraphSidecar" else "video" if post.is_video else "photo",
                        "url": f"https://www.instagram.com/p/{post.shortcode}/",
                        "location": str(post.location) if post.location else None,
                        "tagged_users": list(post.tagged_users) if post.tagged_users else []
                    }
                    
                    posts_data.append(post_info)
                    
                    # Add option to unfollow after download if it was a new follow
                    if index == post_limit - 1 and profile.is_private and follow_choice == 'y':
                        unfollow = input("\nWould you like to unfollow this account now? (y/n): ").lower()
                        if unfollow == 'y':
                            try:
                                L.context.get_and_write_raw(
                                    path=f"friendships/{profile.userid}/unfollow/",
                                    params="",
                                    post=True
                                )
                                print(f"Successfully unfollowed {username}")
                            except Exception as e:
                                print(f"Error unfollowing account: {str(e)}")
                    
                except Exception as e:
                    print(f"Error downloading post {index + 1}: {str(e)}")
                    continue
            
            # Save summary with profile info and all posts
            summary_data = {
                "profile_info": {
                    "username": username,
                    "full_name": profile.full_name,
                    "biography": profile.biography,
                    "is_private": profile.is_private,
                    "is_verified": profile.is_verified,
                    "follower_count": profile.followers,
                    "following_count": profile.followees,
                    "media_count": profile.mediacount,
                    "download_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "downloaded_posts": posts_data
            }
            
            summary_file = os.path.join(user_folder, 'downloads_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=4, ensure_ascii=False)
            
            return f"""Downloads completed successfully:
- Media files saved to: {media_folder}
- Summary saved to: {summary_file}
- Total posts downloaded: {len(posts_data)}"""
        else:
            return "This profile has no posts to download."
                
    except Exception as e:
        return f"Error occurred: {str(e)}"

def main():
    print("Instagram Post Downloader (Public & Private Profiles)")
    username = input("Enter Instagram username to download posts from: ")
    custom_location = input("Enter custom save location (press Enter to use default Pictures folder): ").strip()
    post_limit = input("Enter number of recent posts to download (default is 5): ").strip()
    
    save_folder = custom_location if custom_location else None
    post_limit = int(post_limit) if post_limit.isdigit() else 5
    
    result = download_profile_posts(username, save_folder, post_limit)
    print("\n" + result)

if __name__ == "__main__":
    main()
