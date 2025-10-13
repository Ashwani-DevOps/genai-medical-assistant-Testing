import requests
from dotenv import load_dotenv
import os

load_dotenv()

LM_API = os.getenv("LMSTUDIO_API")
FDA_API = os.getenv("OPENFDA_API")

def get_drug_info(drug_name):
    url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug_name}&limit=1"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        return data["results"][0].get("indications_and_usage", ["No info found"])[0]
    return "Drug info not available"