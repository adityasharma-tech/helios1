import os
import re
import urllib.request
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://data.darts.isas.jaxa.jp/pub/pds3/sln-l-tc-4-dtm-ortho-v3.0/"
OUTPUT_DIR = "./selene_metadata"

def get_links(url):
    """Scrape the Apache directory listing to get href links."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
            # Extract href attributes
            links = re.findall(r'href="([^"]+)"', html)
            # Filter out generic parent links and query params
            links = [l for l in links if not l.startswith('?') and not l.startswith('/') and l != '../']
            return links
    except Exception as e:
        # Expected if a directory like data/ doesn't exist for some reason
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
        print(f"Error downloading {url}: {e}")

def process_folder(folder):
    folder_url = urljoin(BASE_URL, folder)
    
    # 1. Process the data/ directory (limit to 20 files max)
    data_url = urljoin(folder_url, "data/")
    data_links = get_links(data_url)
    
    lbl_files = [l for l in data_links if l.endswith('.lbl')]
    lbl_files_to_download = lbl_files[:20]
    
    for lbl in lbl_files_to_download:
        file_url = urljoin(data_url, lbl)
        dest_path = os.path.join(OUTPUT_DIR, folder, "data", lbl)
        download_file(file_url, dest_path)
        
    # 2. Process the index/ directory (these are tiny)
    index_url = urljoin(folder_url, "index/")
    index_links = get_links(index_url)
    for idx_file in index_links:
        if idx_file.endswith('.lbl') or idx_file.endswith('.tab') or idx_file.endswith('.txt'):
            file_url = urljoin(index_url, idx_file)
            dest_path = os.path.join(OUTPUT_DIR, folder, "index", idx_file)
            download_file(file_url, dest_path)
            
    return folder

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Fetching root directory: {BASE_URL}")
    folders = get_links(BASE_URL)
    
    # Filter for date folders (they end with '/' and are all digits)
    date_folders = [f for f in folders if f.endswith('/') and f[:-1].isdigit()]
    
    # Condition: Continue downloading AFTER (or starting from) 20071213
    start_date = '20071213'
    date_folders = [f for f in date_folders if f[:-1] >= start_date]
    date_folders.sort()
    
    print(f"Found {len(date_folders)} folders starting from {start_date}...")
    print("Starting parallel download with 20 concurrent workers...")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_folder, folder) for folder in date_folders]
        for future in as_completed(futures):
            folder_name = future.result()
            print(f"Finished processing folder: {folder_name}")

if __name__ == '__main__':
    main()
