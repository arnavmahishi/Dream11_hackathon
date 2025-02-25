import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def load_data(file_path):
    """Load the cricket data from an Excel file."""
    return pd.ExcelFile(file_path)

def preprocess_data(excel_data, team_name):
    """Prepare data by merging batting, bowling, and total target data."""
    batting_df = excel_data.parse(f'{team_name}(Bat)')
    bowling_df = excel_data.parse(f'{team_name}(Bowl)')
    total_df = excel_data.parse(f'{team_name}(Total)')
    
    # Drop BBI column if present
    if 'BBI' in bowling_df.columns:
        bowling_df = bowling_df.drop(columns=['BBI'])

    # Merge batting and bowling stats
    merged_data = pd.merge(batting_df, bowling_df, on='Player', how='outer').fillna(0)
    merged_data = pd.merge(merged_data, total_df[['Player','Batting','Bowling','Target']], on='Player', how='left').fillna(0)
    
    
    return merged_data

def train_model(data):
    """Train an AI model to predict total contribution based on all stats."""
    
    # Ensure necessary columns exist
    if 'Batting' not in data.columns or 'Bowling' not in data.columns:
        raise ValueError("Batting or Bowling columns not found in dataset")
    
    # Features and target variable
    X = data[['Batting', 'Bowling']]
    y = data['Target']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict scores
    data['Predicted Score'] = model.predict(X)
    top_players = data[['Player', 'Predicted Score', 'Batting', 'Bowling']].sort_values(by='Predicted Score', ascending=False)
    
    return top_players

if __name__ == "__main__":
    file_path = "./CricketStats-Dream11Hackathon.xlsx"
    team1 = "New Zealand"
    team2 = "Bangladesh"
    
    excel_data = load_data(file_path)
    
    # Process and train for both teams
    data_team1 = preprocess_data(excel_data, team1)
    data_team2 = preprocess_data(excel_data, team2)
    
    top_team1 = train_model(data_team1)
    top_team2 = train_model(data_team2)
    
    # Combine and display top players
    combined_top_players = pd.concat([top_team1, top_team2]).sort_values(by='Predicted Score', ascending=False).head(11)
    print("Top 11 players overall:")
    print(combined_top_players)

