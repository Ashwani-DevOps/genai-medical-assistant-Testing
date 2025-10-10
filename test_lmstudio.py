import requests

url = "http://127.0.0.1:1234/v1/completions"

payload = {
    "prompt": "When to take Amlopidine?",
    "temperature": 0.7,
    "max_tokens": 200,
    "model": "mistral-7b-instruct-v0.2"
}


res = requests.post(url, json=payload)
print("Status:", res.status_code)
print("Response:", res.json())