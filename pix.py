import instaloader
import os
import shutil

def download_profile_picture(username):
    """
    Download profile picture of a given Instagram username
    
    Args:
        username (str): Instagram username
    
    Returns:
        str: Path to downloaded file or error message
    """
    try:
        # Create an instance of Instaloader
        L = instaloader.Instaloader(dirname_pattern='profile_pics')
        
        # Get profile information
        profile = instaloader.Profile.from_username(L.context, username)
        
        # Create directory if it doesn't exist
        if not os.path.exists('profile_pics'):
            os.makedirs('profile_pics')
        
        # Download profile picture
        L.download_profilepic(profile)
        
        # Rename the file to something more user-friendly
        # Find the downloaded file (it will be the most recent file in the directory)
        downloaded_files = [f for f in os.listdir('profile_pics') if f.endswith('.jpg')]
        if downloaded_files:
            old_filepath = os.path.join('profile_pics', downloaded_files[0])
            new_filepath = os.path.join('profile_pics', f'{username}_profile_pic.jpg')
            
            # If a file with the new name already exists, remove it
            if os.path.exists(new_filepath):
                os.remove(new_filepath)
                
            # Rename the file
            shutil.move(old_filepath, new_filepath)
            
            return f"Profile picture downloaded successfully to {new_filepath}"
        else:
            return "Error: No profile picture was downloaded"
        
    except instaloader.exceptions.ProfileNotExistsException:
        return f"Error: Profile '{username}' does not exist"
    except Exception as e:
        return f"Error occurred: {str(e)}"

def main():
    # Example usage
    username = input("Enter Instagram username: ")
    result = download_profile_picture(username)
    print(result)

main()