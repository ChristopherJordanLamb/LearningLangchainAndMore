import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AMADEUS_API_KEY")
API_SECRET = os.getenv("AMADEUS_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("Please set AMADEUS_API_KEY and AMADEUS_API_SECRET in your .env file")


def searchHotels(latitude: float, longitude: float, radius: float):
    """
    Searches for hotels near the given latitude and longitude within the specified radius (km).
    Returns a string listing hotels with addresses.
    """
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

    # Search hotels by geocode
    search_url = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-geocode"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "radius": radius,
        "radiusUnit": "KM",
        "hotelSource": "ALL"
    }

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    hotels = response.json()

    # Build result string
    res = ""
    print(hotels)
    for hotel in hotels.get("data", []):
        hotelID = hotel.get("hotelId", "Unknown")
        name = hotel.get("name", "Unknown")
        address_lines = hotel.get("address", {}).get("lines", [])
        city = hotel.get("address", {}).get("cityName", "")
        country = hotel.get("address", {}).get("countryCode", "")
        res += f"ID:{hotelID} - {name} - {', '.join(address_lines)} - {city}, {country}\n"

    return res
def getRating(hotel_ID: str): #doesn't work with the above - contacted amadeus dev's to ask why it's not compatible
    """
    Retrieves the rating and sentiment details for a specific hotel by its ID.
    """
    auth_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    auth_data = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": API_SECRET
    }

    auth_response = requests.post(auth_url, data=auth_data)
    auth_response.raise_for_status()
    access_token = auth_response.json()["access_token"]

    search_url = f"https://test.api.amadeus.com/v2/e-reputation/hotel-sentiments?hotelIds={hotel_ID}"
    headers = {"Authorization": f"Bearer {access_token}"}

    response = requests.get(search_url, headers=headers)
    response.raise_for_status()
    hotel_data = response.json()
    print(hotel_data)
    if not hotel_data.get("data"):
        return f"No rating information found for hotel ID: {hotel_ID}"
    
    hotel = hotel_data["data"][0]
    hotelID = hotel.get("hotelId", "Unknown")
    overallRating = hotel.get("overallRating", "Unknown")
    numberOfReviews = hotel.get("numberOfReviews", "Unknown")
    numberOfRatings = hotel.get("numberOfRatings", "Unknown")
    sentiments = hotel.get("sentiments", {})

    #Build result string
    res = f"Hotel ID: {hotelID}\n"
    res += f"Overall Rating: {overallRating}/100\n"
    res += f"Number of Reviews: {numberOfReviews}\n"
    res += f"Number of Ratings: {numberOfRatings}\n"
    res += "\nSentiment Scores:\n"
    for key, value in sentiments.items():
        res += f"  - {key}: {value}/100\n"
    return res
def main():
    print(searchHotels(35.6938, 139.7034, 1))
    print(getRating("TELONMFS"))
if __name__ == "__main__":
    main()
