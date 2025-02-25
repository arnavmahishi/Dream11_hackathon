import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

team_mapping = {
    "AFG": "Afghanistan", "AUS": "Australia", "BAN": "Bangladesh", "ENG": "England",
    "IND": "India", "NZ": "New Zealand", "PAK": "Pakistan", "SA": "South Africa"
}

def load_data(file_path):
    """Load the Excel file and return an ExcelFile object."""
    return pd.ExcelFile(file_path)

def load_roles(file_path):
    """Load player roles and team names from the squad sheet."""
    roles_df = pd.read_excel(file_path, sheet_name="Data_20250224")
    
    print("Columns in roles_df:", roles_df.columns.tolist())  # Debugging
    print("Unique values in 'IsPlaying':", roles_df['IsPlaying'].unique())  # Debugging

    if {'Player Name', 'Player Type', 'IsPlaying', 'Team'}.issubset(roles_df.columns):
        roles_df['IsPlaying'] = roles_df['IsPlaying'].astype(str).str.strip().str.lower()
        roles_df['IsPlaying'] = roles_df['IsPlaying'].map({'playing': True, 'not_playing': False})
        
        roles_df['Team'] = roles_df['Team'].map(team_mapping).fillna(roles_df['Team'])
        teams = roles_df['Team'].unique().tolist()
        roles = roles_df.set_index('Player Name')[['Player Type', 'IsPlaying', 'Team']].to_dict(orient='index')
        
        return roles, teams
    else:
        raise KeyError("Expected columns 'Player Name', 'Player Type', 'IsPlaying', and 'Team' not found in roles file.")

def preprocess_data(excel_data, team_name):
    """Merge batting, bowling, and total stats into one DataFrame using mapped team names."""
    # Convert short alias (e.g., BAN) to full team name (e.g., Bangladesh)
    team_full_name = team_mapping.get(team_name, team_name)

    # Check actual sheet names to find the correct match
    available_sheets = excel_data.sheet_names
    def find_matching_sheet(base_name):
        for sheet in available_sheets:
            if base_name in sheet:
                return sheet
        return None  # Return None if no match is found

    # Find correct sheet names dynamically
    batting_sheet = find_matching_sheet(f"{team_full_name}(Bat)")
    bowling_sheet = find_matching_sheet(f"{team_full_name}(Bowl)")
    total_sheet = find_matching_sheet(f"{team_full_name}(Total)")

    # Handle missing sheets gracefully
    if not batting_sheet or not total_sheet:
        raise ValueError(f"Missing required sheets for team {team_full_name}")

    batting_df = excel_data.parse(batting_sheet)
    bowling_df = excel_data.parse(bowling_sheet) if bowling_sheet else pd.DataFrame(columns=['Player'])
    total_df = excel_data.parse(total_sheet)

    if 'BBI' in bowling_df.columns:
        bowling_df = bowling_df.drop(columns=['BBI'])

    merged_data = pd.merge(batting_df, bowling_df, on='Player', how='outer').fillna(0)
    merged_data = pd.merge(merged_data, total_df[['Player', 'Target']], on='Player', how='left').fillna(0)

    return merged_data


def train_model(data):
    """Train a model and predict player contribution."""
    X = data.drop(columns=['Player', 'Target'], errors='ignore')
    X = X.select_dtypes(exclude=['datetime', 'object'])
    X.columns = X.columns.astype(str)
    
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    data['Predicted Score'] = model.predict(X)
    return data[['Player', 'Predicted Score']].sort_values(by='Predicted Score', ascending=False)

def select_top_11(top_team1, top_team2, roles):
    """Selects the best 11 players while ensuring constraints are met."""
    not_playing = {p for p, r in roles.items() if not r["IsPlaying"]}
    print("Total NOT_PLAYING Players:", len(not_playing))  # Debugging
    print("NOT_PLAYING Players List:", not_playing)  # Debugging

    top_team1 = top_team1[~top_team1['Player'].isin(not_playing)].copy()
    top_team2 = top_team2[~top_team2['Player'].isin(not_playing)].copy()
    
    combined_players = pd.concat([top_team1, top_team2]).sort_values(by='Predicted Score', ascending=False)
    
    selected_players = []
    selected_set = set()
    team_counts = {team1: 0, team2: 0}
    role_counts = {'BAT': 0,'BOWL': 0, 'ALL': 0, 'WK': 0}
    min_constraints = {'BAT':5,'BOWL': 2, 'ALL': 3, 'WK': 1}
    
    for _, player in combined_players.iterrows():
        player_name = player['Player']
        if player_name in selected_set:
            continue
        
        role = roles.get(player_name, {}).get("Player Type", "BAT")
        player_team = roles.get(player_name, {}).get("Team")
        if player_team not in team_counts:
           continue  # Skip players with an undefined or incorrect team

        
        if team_counts[player_team] < 6:
            if role in min_constraints and role_counts[role] >= min_constraints[role]:
                continue
            role_counts[role] += 1
            team_counts[player_team] += 1
            
            selected_players.append(player)
            selected_set.add(player_name)
        
        if len(selected_players) == 16:
            break
    
    captain_team = team1 if team_counts[team1] == 6 else team2
    vice_captain_team = team2 if captain_team == team1 else team1
    
    captain = max([p for p in selected_players if roles.get(p['Player'], {}).get("Player Type") == "BAT" and roles.get(p['Player'], {}).get("Team") == captain_team], key=lambda x: x['Predicted Score'])
    vice_captain = max([p for p in selected_players if roles.get(p['Player'], {}).get("Player Type") == "BAT" and roles.get(p['Player'], {}).get("Team") == vice_captain_team], key=lambda x: x['Predicted Score'])
    
    final_df = pd.DataFrame(selected_players[:11])
    final_df['Role'] = "Player"
    final_df.loc[final_df['Player'] == captain['Player'], 'Role'] = "Captain"
    final_df.loc[final_df['Player'] == vice_captain['Player'], 'Role'] = "Vice-Captain"
    
    print("Captain:", captain['Player'])
    print("Vice-Captain:", vice_captain['Player'])
    
    return final_df

if __name__ == "__main__":
    stats_file = "./CricketStats-Dream11Hackathon.xlsx"
    roles_file = "./SquadPlayerNames.xlsx"
    
    roles, teams = load_roles(roles_file)
    team1, team2 = teams[:2]  # Selecting the first two teams found
    
    excel_data = load_data(stats_file)
    
    data_team1 = preprocess_data(excel_data, team1)
    data_team2 = preprocess_data(excel_data, team2)
    
    top_team1 = train_model(data_team1)
    top_team2 = train_model(data_team2)
    
    final_team = select_top_11(top_team1, top_team2, roles)
    
    print("Final Top 11 Players:")
    print(final_team)

