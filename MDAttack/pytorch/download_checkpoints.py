"""Download checkpoint files for defense models"""
import os
import requests
from tqdm import tqdm

def download_file_from_google_drive(file_id, destination):
    """Download file from Google Drive with proper handling of virus scan warning"""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # First request to get the confirmation token
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Check if we need confirmation (for large files)
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            # Need to confirm download
            params = {'id': file_id, 'confirm': value}
            response = session.get(URL, params=params, stream=True)
            break
    
    # If still HTML, extract confirm token from response
    if 'text/html' in response.headers.get('content-type', ''):
        # Try to extract confirm token from HTML
        content = response.text
        if 'confirm=t' in content:
            # Use the new download URL for large files
            download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            response = session.get(download_url, stream=True)
    
    # Save the file
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(desc=os.path.basename(destination), total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Downloaded {destination}")
    
    # Verify it's not HTML
    with open(destination, 'rb') as f:
        first_bytes = f.read(10)
        if first_bytes.startswith(b'<!DOCTYPE'):
            print(f"Error: Downloaded HTML instead of model file for {destination}")
            os.remove(destination)
            return False
    
    return True

# Download RST checkpoint
print("Downloading RST checkpoint...")
success = download_file_from_google_drive('1S3in_jVYJ-YBe5-4D0N70R4bN82kP5U2', 
                                         'checkpoints/RST/cifar10_rst_adv.pt')

if success:
    print("\nCheckpoint downloaded successfully!")
else:
    print("\nFailed to download checkpoint. You may need to download manually from:")
    print("https://drive.google.com/file/d/1S3in_jVYJ-YBe5-4D0N70R4bN82kP5U2/view")