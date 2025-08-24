import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AMADEUS_API_KEY")
API_SECRET = os.getenv("AMADEUS_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("Please set AMADEUS_API_KEY and AMADEUS_API_SECRET in your .env file")

# Get access token
auth_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
auth_data = {
    "grant_type": "client_credentials",
    "client_id": API_KEY,
    "client_secret": API_SECRET
}

auth_response = requests.post(auth_url, data=auth_data)
auth_response.raise_for_status()
access_token = auth_response.json()["access_token"]
print("Access token obtained:", access_token)

# Test request
search_url = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city"
headers = {"Authorization": f"Bearer {access_token}"}

params = {
    "cityCode": "PAR",
    "radius": 1,
    "radiusUnit": "KM",
    "hotelSource": "ALL"
}
headers = {"Authorization": f"Bearer {access_token}"}

response = requests.get(search_url, headers=headers, params=params)
response.raise_for_status()
hotels = response.json()
for hotel in hotels.get("data", []):
    name = hotel.get("name")
    address_lines = hotel.get("address", {}).get("lines", [])
    city = hotel.get("address", {}).get("cityName", "")
    country = hotel.get("address", {}).get("countryCode", "")
    print(f"{name} - {', '.join(address_lines)} - {city}, {country}")
