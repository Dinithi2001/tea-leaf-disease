import requests

API_KEY = "7b55a95be33623bf80cf4290d17344b2"

def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    print(data)  # debug — shows exactly what the API returned

    humidity = data["main"]["humidity"]
    temperature = data["main"]["temp"]
    cloud_cover = data["clouds"]["all"]
    sunshine_hours = 8 if cloud_cover < 30 else 2

    return humidity, temperature, sunshine_hours

if __name__ == "__main__":
    lat, lon = 6.9271, 79.8612
    humidity, temperature, sunshine_hours = get_weather(lat, lon)
    print(f"Humidity: {humidity}%")
    print(f"Temperature: {temperature}°C")
    print(f"Estimated sunshine hours: {sunshine_hours}")