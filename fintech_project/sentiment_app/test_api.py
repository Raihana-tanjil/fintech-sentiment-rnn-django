import requests

url = "http://127.0.0.1:8000/api/predict/"

# Define a list of test payloads
test_cases = [
    {"text": "Company reports record-breaking quarterly net revenue."},
    {"text": "Severe supply chain disruptions lead to massive quarterly losses."},
    {"text": "Stock prices surge following unexpected acquisition announcement."},
    {"text": "Regulatory investigation causes sharp decline in investor confidence."}
]

print("--- Testing Fintech Sentiment API ---")
for i, payload in enumerate(test_cases, 1):
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            data = result.get('data', {})
            print(f"\nTest #{i}")
            print(f"Input: '{data.get('text')}'")
            print(f"Sentiment: {data.get('sentiment')}")
            print(f"Score: {data.get('score')}")
        else:
            print(f"\nTest #{i} failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"\nTest #{i} error: {e}")