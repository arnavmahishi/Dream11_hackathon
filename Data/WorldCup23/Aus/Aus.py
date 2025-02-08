import pandas as pd
import requests
from bs4 import BeautifulSoup

# URL of the player's stats page
url = 'https://www.espncricinfo.com/records/tournament/averages-batting-bowling-by-team/icc-cricket-world-cup-2023-24-15338?team=2'

# Send a GET request to the URL
r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})

# Parse the HTML content
soup = BeautifulSoup(r.text, 'html.parser')

# Find all tables on the page
tables = soup.find_all('table')

# Ensure at least 2 tables exist
if len(tables) < 2:
    print("Error: Could not find two consecutive tables.")
    exit()

# Function to extract data from a table
def extract_table(table):
    table_head = table.find('thead')
    table_body = table.find('tbody')

    # Extract column names
    column_names = [th.text.strip() for th in table_head.find_all('span', class_='ds-cursor-pointer')]

    # Extract row data
    rows = []
    for tr in table_body.find_all('tr'):
        row = [td.text.strip() for td in tr.find_all('td')]
        rows.append(row)

    return pd.DataFrame(rows, columns=column_names)

# Extract first two tables
df_batting = extract_table(tables[0])  # First table (Batting stats)
df_bowling = extract_table(tables[1])  # Second table (Bowling stats)

# Combine tables (Optional: Add a 'Category' column)
df_batting["Category"] = "Batting"
df_bowling["Category"] = "Bowling"

# Merge the two tables
df_combined = pd.concat([df_batting, df_bowling], ignore_index=True)

# Save to CSV
df_batting.to_csv('bat.csv', index=False)
df_bowling.to_csv('bowl.csv', index=False)
# Print the DataFrame
print(df_batting)
print(df_bowling)
print("Data saved to player_stats_combined.csv")

