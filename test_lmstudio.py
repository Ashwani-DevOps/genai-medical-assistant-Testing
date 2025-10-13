import requests
#Test local code
payload = {
    "model": "mistral-7b-instruct-v0.2",
    "messages": [
        {"role": "user", "content": "What is ibuprofen?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
}

res = requests.post("https://weariest-lacklustrely-jessia.ngrok-free.dev/v1/chat/completions", json=payload)
print(res.json())