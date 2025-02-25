import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the stats page
url = "https://stats.espncricinfo.com/ci/engine/stats/index.html?class=2;home_or_away=1;home_or_away=2;home_or_away=3;result=1;result=2;result=3;result=5;spanmin1=05+Jan+2023;spanval1=span;template=results;type=team"

# Headers to mimic a real browser visit
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Fetch the page content
response = requests.get(url, headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the table containing the stats
    table = soup.find("table", {"class": "engineTable"})

    if table:
        # Extract headers
        headers = [th.text.strip() for th in table.find_all("th")]

        # Extract rows
        rows = []
        for tr in table.find_all("tr")[1:]:  # Skip the header row
            cells = tr.find_all("td")
            row = [cell.text.strip() for cell in cells]
            if row:
                rows.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=headers)

        # Save to CSV
        df.to_csv("espn_cricket_stats.csv", index=False)
        print("Data successfully scraped and saved to espn_cricket_stats.csv")

    else:
        print("Stats table not found on the page.")
else:
    print(f"Failed to fetch the page. Status code: {response.status_code}")

