import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load the Excel file
file_path = "CricketStats-Dream11Hackathon.xlsx"
xls = pd.ExcelFile(file_path)

# Read relevant sheets
batting_df = pd.read_excel(xls, sheet_name="Afghanisthan(Bat)")
bowling_df = pd.read_excel(xls, sheet_name="Afghanisthan(Bowl)")
points_df = pd.read_excel(xls, sheet_name="Afghanisthan(Total Points)")

# Extract relevant features
batting_features = ["Runs_bat", "SR_bat", "100", "50"]
bowling_features = ["Wkts", "Econ"]

# Ensure selected columns exist
batting_features = [col for col in batting_features if col in batting_df.columns]
bowling_features = [col for col in bowling_features if col in bowling_df.columns]

# Drop NaN values from each dataset separately
batting_clean = batting_df[["Player"] + batting_features].dropna()
bowling_clean = bowling_df[["Player"] + bowling_features].dropna()
points_clean = points_df[["Player", "Total"]].dropna()

# Identify common players across datasets
common_players = set(batting_clean["Player"]) & set(bowling_clean["Player"]) & set(points_clean["Player"])

# Filter each dataset to keep only common players
batting_clean = batting_clean[batting_clean["Player"].isin(common_players)].reset_index(drop=True)
bowling_clean = bowling_clean[bowling_clean["Player"].isin(common_players)].reset_index(drop=True)
points_clean = points_clean[points_clean["Player"].isin(common_players)].reset_index(drop=True)

# Ensure order is the same across datasets
batting_clean.sort_values(by="Player", inplace=True)
bowling_clean.sort_values(by="Player", inplace=True)
points_clean.sort_values(by="Player", inplace=True)

# Combine batting and bowling stats
X = pd.concat([batting_clean[batting_features].reset_index(drop=True), 
               bowling_clean[bowling_features].reset_index(drop=True)], axis=1)
y = points_clean["Total"]

# Convert non-numeric values to NaN and drop them
X = X.apply(pd.to_numeric, errors='coerce').dropna()
y = y.loc[X.index]  # Ensure y matches filtered X

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on test data and calculate MAE
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)

# Predict total points for all players
points_clean = points_clean.loc[X.index]  # Ensure consistent indexing
points_clean["Predicted Points"] = model.predict(scaler.transform(X))

# Get top 11 players by predicted points
top_11_players = points_clean.sort_values(by="Predicted Points", ascending=False).head(11)

# Print results
print("Top 11 Players Based on Predicted Points:")
print(top_11_players[["Player", "Predicted Points"]])
print(f"Mean Absolute Error: {mae:.2f}")

