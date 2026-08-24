import os
import re
import json
import threading
import urllib.request
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://data.darts.isas.jaxa.jp/pub/pds3/sln-l-tc-4-dtm-ortho-v3.0/"
OUTPUT_DIR = "./selene_metadata"
CONFIG_FILE = "config.txt"

# Thread lock for safely updating config.txt across 20 parallel workers
config_lock = threading.Lock()

def load_config():
    """Load the progress state from config.txt"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_config(config):
    """Save the progress state to config.txt (thread-safe)"""
    with config_lock:
        with open(CONFIG_FILE, 'w') as f:
            # We save it as JSON inside the .txt file for robust key-value pairing
            json.dump(config, f, indent=4)

def get_links(url):
    """Scrape the Apache directory listing to get href links."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
            links = re.findall(r'href="([^"]+)"', html)
            links = [l for l in links if not l.startswith('?') and not l.startswith('/') and l != '../']
            return links
    except Exception as e:
        return []

def download_file(url, dest_path):
    """Download a file if it doesn't already exist."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with open(dest_path, 'wb') as f:
                f.write(response.read())
    except Exception as e:
        pass

def process_folder(folder, config):
    # If the folder isn't in config, we assume 20 files were already downloaded based on your previous script
    current_offset = config.get(folder, 20)
    chunk_size = 50 
    
    folder_url = urljoin(BASE_URL, folder)
    data_url = urljoin(folder_url, "data/")
    data_links = get_links(data_url)
    
    lbl_files = [l for l in data_links if l.endswith('.lbl')]
    
    # Get the NEXT 50 files based on the current offset
    lbl_files_to_download = lbl_files[current_offset : current_offset + chunk_size]
    
    if not lbl_files_to_download:
        return folder, 0 # Nothing more to download
        
    for lbl in lbl_files_to_download:
        file_url = urljoin(data_url, lbl)
        dest_path = os.path.join(OUTPUT_DIR, folder, "data", lbl)
        download_file(file_url, dest_path)
        
    # Successfully downloaded the chunk, update the state in config.txt
    new_offset = current_offset + len(lbl_files_to_download)
    config[folder] = new_offset
    save_config(config)
    
    return folder, len(lbl_files_to_download)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Load the state
    config = load_config()
    
    print(f"Fetching root directory: {BASE_URL}")
    folders = get_links(BASE_URL)
    
    # Filter for date folders starting from 20071213
    date_folders = [f for f in folders if f.endswith('/') and f[:-1].isdigit()]
    start_date = '20071213'
    date_folders = [f for f in date_folders if f[:-1] >= start_date]
    date_folders.sort()
    
    print(f"Found {len(date_folders)} folders starting from {start_date}...")
    print("Starting parallel download of NEXT 50 files with 20 concurrent workers...")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_folder, folder, config) for folder in date_folders]
        for future in as_completed(futures):
            folder_name, count = future.result()
            print(f"Finished {folder_name}: Downloaded {count} new files. Total files processed is now {config.get(folder_name)}.")

if __name__ == '__main__':
    main()
