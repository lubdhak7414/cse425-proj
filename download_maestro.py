import urllib.request
import zipfile
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
zip_path = "maestro-v3.0.0-midi.zip"
extract_dir = "data"

os.makedirs(extract_dir, exist_ok=True)
print("Downloading...")
urllib.request.urlretrieve(url, zip_path)
print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
os.remove(zip_path)
print("Done!")
