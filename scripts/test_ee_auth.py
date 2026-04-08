"""Quick smoke test for Earth Engine service account authentication."""
import json
import ee

KEY_FILE = "ee-key.json"
PROJECT = "costal-492719"

# Read service account email from the key file
with open(KEY_FILE) as f:
    service_account = json.load(f)["client_email"]

print(f"Authenticating as: {service_account}")
credentials = ee.ServiceAccountCredentials(service_account, KEY_FILE)
ee.Initialize(credentials, project=PROJECT)

# Trivial computation to confirm round-trip works
result = ee.Number(42).getInfo()
print(f"Earth Engine round-trip OK. Result: {result}")

# Slightly more meaningful test — fetch metadata for a known Sentinel-2 collection
collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").limit(1)
first = collection.first()
info = first.getInfo()
print(f"Sentinel-2 sample image ID: {info['id']}")
print("Earth Engine authentication is fully working.")
