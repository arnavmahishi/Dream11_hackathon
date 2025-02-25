import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# ESPN Statsguru URL (contains all data on one page)
url = "https://stats.espncricinfo.com/ci/engine/stats/index.html?class=2;home_or_away=1;home_or_away=2;home_or_away=3;result=1;result=2;result=3;result=5;spanmax1=11+Feb+2025;spanmin1=11+Feb+2015;spanval1=span;team=40;template=results;type=bowling"
# Configure Selenium WebDriver (Headless Chrome)
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in background
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

# Start WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Open URL and wait for JavaScript to load
driver.get(url)
time.sleep(5)  # Ensure full page loads

# Get page source after JavaScript execution
soup = BeautifulSoup(driver.page_source, "html.parser")

# Close browser
driver.quit()

# Find all tables with class 'engineTable'
tables = soup.find_all("table", {"class": "engineTable"})

if len(tables) > 2:  # Ensure we have enough tables
    stats_table = tables[2]  # "Overall figures" table is the 3rd one

    # Extract table headers
    headers = [th.get_text(strip=True) for th in stats_table.find_all("th")]

    # Extract player rows
    rows = []
    for tr in stats_table.find_all("tr"):
        cols = tr.find_all("td")

        # Ensure the row contains player data (by checking <a class="data-link">)
        if cols and tr.find("a", class_="data-link"):
            row_data = [td.get_text(strip=True) for td in cols]
            rows.append(row_data)

    # Convert to DataFrame if data is present
    if rows:
        df = pd.DataFrame(rows, columns=headers)

        # Display first few rows
        print("\nFinal Extracted Data:")
        print(df.head())

        # Save to CSV
        df.to_csv("espn_bowling_stats_complete.csv", index=False)
        print("✅ Data saved to espn_bowling_stats_complete.csv")
    else:
        print("⚠ No valid player data found.")
else:
    print("⚠ Data table not found.")

