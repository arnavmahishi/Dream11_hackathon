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
    #roles_df = pd.read_excel(file_path, sheet_name="Data_20250212")
    roles_df = pd.read_excel(file_path, sheet_name="Data_{}".format(datetime.today().strftime('%Y%m%d')))    
    unique_teams = roles_df['Team'].dropna().unique()
    
    if len(unique_teams) < 2:
        raise ValueError("Not enough teams found in the roles sheet.")

    team1, team2 = [team_mapping.get(team, team) for team in unique_teams[:2]]
    return team1, team2

def load_roles(file_path):
    from datetime import datetime
    #roles_df = pd.read_excel(file_path, sheet_name="Data_20250212")
    #roles_df = pd.read_excel(file_path, sheet_name="Data_20250223")
    today_date = datetime.today().strftime('%Y%m%d')
    sheet_name = f"Data_{today_date}"
    roles_df = pd.read_excel(file_path, sheet_name=sheet_name)
    roles_df['IsPlaying'] = roles_df['IsPlaying'].astype(str).str.strip().str.lower()
    roles_df['IsPlaying'] = roles_df['IsPlaying'].map({'playing': True, 'not_playing': False})
    roles_df['Team'] = roles_df['Team'].map(team_mapping).fillna(roles_df['Team'])
    roles = roles_df.set_index('Player Name')[['Player Type', 'IsPlaying', 'Team']].to_dict(orient='index')
    return roles

def preprocess_data(excel_data, team_name):
    team_full_name = team_mapping.get(team_name, team_name)
    available_sheets = excel_data.sheet_names
    
    def find_matching_sheet(base_name):
        for sheet in available_sheets:
            if base_name in sheet:
                return sheet
        return None

    batting_sheet = find_matching_sheet(f"{team_full_name}(Bat)")
    bowling_sheet = find_matching_sheet(f"{team_full_name}(Bowl)")
    total_sheet = find_matching_sheet(f"{team_full_name}(Total)")


    
    if not batting_sheet or not total_sheet:
        raise ValueError(f"Missing required sheets for team {team_full_name}")

    batting_df = excel_data.parse(batting_sheet)
    bowling_df = excel_data.parse(bowling_sheet) if bowling_sheet else pd.DataFrame(columns=['Player'])
    total_df = excel_data.parse(total_sheet)
    print(f"\nChecking columns for {team_full_name}:")
    print("Batting Sheet Columns:", batting_df.columns.tolist())
    print("Bowling Sheet Columns:", bowling_df.columns.tolist())
    print("Total Sheet Columns:", total_df.columns.tolist())
    if 'BBI' in bowling_df.columns:
        bowling_df = bowling_df.drop(columns=['BBI'])
    
    merged_data = pd.merge(batting_df, bowling_df, on='Player', how='outer').fillna(0)
    merged_data = pd.merge(merged_data, total_df[['Player', 'Target']], on='Player', how='left').fillna(0)
    return merged_data

def train_model(data):
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
    not_playing = {p for p, r in roles.items() if not r["IsPlaying"]}
    top_team1 = top_team1[~top_team1['Player'].isin(not_playing)].copy()
    top_team2 = top_team2[~top_team2['Player'].isin(not_playing)].copy()
    combined_players = pd.concat([top_team1, top_team2]).sort_values(by='Predicted Score', ascending=False)
    selected_players = []
    selected_set = set()
    
    for _, player in combined_players.iterrows():
        if player['Player'] not in selected_set:
            selected_players.append(player)
            selected_set.add(player['Player'])
        if len(selected_players) == 11:
            break
    
    final_df = pd.DataFrame(selected_players)
    final_df['Team'] = final_df['Player'].apply(lambda x: roles[x]['Team'] if x in roles else "Not Found")
    final_df['C/VC'] = "NA"
    captain = max(final_df.to_dict('records'), key=lambda x: x['Predicted Score'])
    vice_captain = max([p for p in final_df.to_dict('records') if p['Player'] != captain['Player']], key=lambda x: x['Predicted Score'])
    final_df.loc[final_df['Player'] == captain['Player'], 'C/VC'] = 'C'
    final_df.loc[final_df['Player'] == vice_captain['Player'], 'C/VC'] = 'VC'
    final_df.rename(columns={'Player': 'Player Name'}, inplace=True)
    final_df[['Player Name', 'Team', 'C/VC']].to_csv('../output.csv', index=False)
    return final_df

if __name__ == "__main__":
    stats_file = "../data/CricketStats-Dream11Hackathon.xlsx"
    roles_file = "../data/SquadPlayerNames.xlsx"

    # Load roles and apply team mapping
    roles = load_roles(roles_file)
    print("Loaded Roles:", roles)

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

