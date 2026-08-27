import requests
response = requests.get("https://www.google.com", timeout=10)
print(response.status_code)