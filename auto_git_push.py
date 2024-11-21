import subprocess
import os
import time

# Define the paths relative to the script's location
repo_path = os.path.dirname(os.path.abspath(__file__))  # Get the repo directory
file_path = os.path.join(repo_path, "Data_10.04.24.xlsx")  # Excel file path

# Monitor the file for changes
last_modified_time = os.path.getmtime(file_path)

while True:
    try:
        current_modified_time = os.path.getmtime(file_path)
        if current_modified_time != last_modified_time:
            print("File updated, pushing to GitHub...")
            os.chdir(repo_path)
            subprocess.run(["git", "add", file_path], check=True)
            subprocess.run(["git", "commit", "-m", "Automated update for Excel file"], check=True)
            subprocess.run(["git", "push", "origin", "main"], check=True)
            last_modified_time = current_modified_time
        time.sleep(10)  # Check every 10 seconds
    except Exception as e:
        print(f"Error occurred: {e}. Retrying in 10 seconds...")
        time.sleep(10)