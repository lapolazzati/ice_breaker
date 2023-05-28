import os
import requests
import json
from dotenv import load_dotenv


load_dotenv()
NUBELA_KEY_LINK = os.getenv('NUBELA_KEY_LINK')


def scrape_linkedin_profile(linkedin_profile_url: str):
    """scrape infomration from LinkedIn profiles,
    Manually scrape the information from the linkedin profile"""
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    header_dic = {"Authorization": f'Bearer {os.environ.get("NUBELA_KEY_LINK")}'}
    response = requests.get(
        api_endpoint, params={"url": linkedin_profile_url}, headers=header_dic
    )

    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")
    
    return data




