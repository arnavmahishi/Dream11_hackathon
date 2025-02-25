import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Team Mapping Dictionary
team_mapping = {
    "AFG": "Afghanistan", "AUS": "Australia", "BAN": "Bangladesh", "ENG": "England",
    "IND": "India", "NZ": "New Zealand", "PAK": "Pakistan", "SA": "South Africa"
}

def load_data(file_path):
    return pd.ExcelFile(file_path)

def get_teams_from_roles(file_path):
    """Extract unique team names from the roles sheet dynamically."""
    #roles_df = pd.read_excel(file_path, sheet_name="Data_{}".format(datetime.today().strftime('%Y%m%d')))
    roles_df = pd.read_excel(file_path, sheet_name="Data_20250223")
    
    # Get unique team names
    unique_teams = roles_df['Team'].dropna().unique()
    
    if len(unique_teams) < 2:
        raise ValueError("Not enough teams found in the roles sheet.")

    # Convert short names (e.g., IND, AUS) to full team names
    team1, team2 = [team_mapping.get(team, team) for team in unique_teams[:2]]

    return team1, team2

def load_roles(file_path):
    from datetime import datetime

    # Generate today's date in the required format
    today_date = datetime.today().strftime('%Y%m%d')
    sheet_name = f"Data_{today_date}"
    #roles_df = pd.read_excel(file_path, sheet_name=sheet_name)
    roles_df = pd.read_excel(file_path, sheet_name="Data_20250223")
    roles_df['IsPlaying'] = roles_df['IsPlaying'].astype(str).str.strip().str.lower()
    roles_df['IsPlaying'] = roles_df['IsPlaying'].map({'playing': True, 'not_playing': False})

    # Apply team mapping
    roles_df['Team'] = roles_df['Team'].map(team_mapping).fillna(roles_df['Team'])

    roles = roles_df.set_index('Player Name')[['Player Type', 'IsPlaying', 'Team']].to_dict(orient='index')
    return roles

def preprocess_data(excel_data, team_name):
    # Convert short alias (e.g., BAN) to full team name (e.g., Bangladesh)
    team_full_name = team_mapping.get(team_name, team_name)

    # Check available sheets to find correct matches
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
    
    # Ensure all column names are strings
    X.columns = X.columns.astype(str)

    y = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)  # **Fix: Ensuring valid column names**
    
    data['Predicted Score'] = model.predict(X)
    return data[['Player', 'Predicted Score']].sort_values(by='Predicted Score', ascending=False)

def select_top_11(top_team1, top_team2, roles):
    """Selects the best 11 players while ensuring team balance, role constraints, and captain/vice-captain selection."""
    not_playing = {p for p, r in roles.items() if not r["IsPlaying"]}
    
    # Remove not playing players
    top_team1 = top_team1[~top_team1['Player'].isin(not_playing)].copy()
    top_team2 = top_team2[~top_team2['Player'].isin(not_playing)].copy()

    combined_players = pd.concat([top_team1, top_team2]).sort_values(by='Predicted Score', ascending=False)

    selected_players = []
    selected_set = set()
    team_counts = {team1: 0, team2: 0}

    # Ensure exactly 6-5 balance
    max_team1 = 6
    max_team2 = 5

    # **Step 1: First, select the top 6 players from team1**
    for _, player in combined_players.iterrows():
        if team_counts[team1] < max_team1 and player['Player'] in top_team1['Player'].values:
            selected_players.append(player)
            selected_set.add(player['Player'])
            team_counts[team1] += 1
        if len(selected_players) == 6:
            break

    # **Step 2: Then, select the top 5 players from team2**
    for _, player in combined_players.iterrows():
        if team_counts[team2] < max_team2 and player['Player'] in top_team2['Player'].values:
            if player['Player'] not in selected_set:  # Avoid duplicates
                selected_players.append(player)
                selected_set.add(player['Player'])
                team_counts[team2] += 1
        if len(selected_players) == 11:
            break

    # Convert selected players to DataFrame
    final_df = pd.DataFrame(selected_players)

    # Print Final Team Composition
    print("\nFinal Team Composition:")
    print(f"Players from {team1}: {team_counts[team1]}, Players from {team2}: {team_counts[team2]}")

    # Assign Captain and Vice-Captain
    final_df['Role'] = "Player"

    captain = max(final_df.to_dict('records'), key=lambda x: x['Predicted Score'])
    vice_captain = max([p for p in final_df.to_dict('records') if p['Player'] != captain['Player']], key=lambda x: x['Predicted Score'])

    final_df.loc[final_df['Player'] == captain['Player'], 'Role'] = 'Captain (C)'
    final_df.loc[final_df['Player'] == vice_captain['Player'], 'Role'] = 'Vice Captain (VC)'

    return final_df


if __name__ == "__main__":
    stats_file = "../data/CricketStats-Dream11Hackathon.xlsx"
    roles_file = "../data/SquadPlayerNames.xlsx"

    # Load roles and apply team mapping
    roles = load_roles(roles_file)

    # Get dynamic teams from roles sheet
    team1, team2 = get_teams_from_roles(roles_file)
    print(f"Selected Teams: {team1} vs {team2}")

    # Load data and process for both teams
    excel_data = load_data(stats_file)
    
    data_team1 = preprocess_data(excel_data, team1)
    data_team2 = preprocess_data(excel_data, team2)

    top_team1 = train_model(data_team1)
    top_team2 = train_model(data_team2)

    final_team = select_top_11(top_team1, top_team2, roles)

    print("Final Top 11 Players:")
    print(final_team)

