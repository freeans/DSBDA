import requests
from bs4 import BeautifulSoup
import pandas as pd

# Target URL
url = "https://www.amazon.in/dp/B08GXYZFNB"

# Request headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Send request
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

# Extract review elements
review_divs = soup.select("div[data-hook='review']")
reviews = []

for div in review_divs:
    rating_tag = div.select_one("i[data-hook='review-star-rating'] span")
    title_tag = div.select_one("a[data-hook='review-title'] span")
    text_tag = div.select_one("span[data-hook='review-body'] span")

    rating = rating_tag.text.strip() if rating_tag else ""
    title = title_tag.text.strip() if title_tag else ""
    text = text_tag.text.strip() if text_tag else ""

    reviews.append({"Rating": rating, "Title": title, "Review": text})

# Save to Excel
df = pd.DataFrame(reviews)
df.to_excel("amazon_reviews.xlsx", index=False)
print("Saved to amazon_reviews.xlsx")
