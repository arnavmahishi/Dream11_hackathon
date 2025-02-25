import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def load_data(file_path):
    """Load the Excel file and return an ExcelFile object."""
    return pd.ExcelFile(file_path)

def load_roles(file_path):
    """Load player roles and team names from the squad sheet."""
    roles_df = pd.read_excel(file_path, sheet_name="Data_20250224")
    
    print("Columns in roles_df:", roles_df.columns.tolist())
    print("Unique values in 'IsPlaying':", roles_df['IsPlaying'].unique())
    
    if {'Player Name', 'Player Type', 'IsPlaying', 'Team'}.issubset(roles_df.columns):
        roles_df['IsPlaying'] = roles_df['IsPlaying'].astype(str).str.strip().str.lower()
        roles_df['IsPlaying'] = roles_df['IsPlaying'].map({'playing': True, 'not_playing': False})
        roles = roles_df.set_index('Player Name')[['Player Type', 'IsPlaying', 'Team']].to_dict(orient='index')
        teams = roles_df['Team'].dropna().unique().tolist()[:2]
        return roles, teams
    else:
        raise KeyError("Expected columns 'Player Name', 'Player Type', 'IsPlaying', and 'Team' not found in roles file.")

def preprocess_data(excel_data, team_alias):
    """Merge batting, bowling, and total stats into one DataFrame."""
    batting_df = excel_data.parse(f'{team_alias}(Bat)')
    bowling_df = excel_data.parse(f'{team_alias}(Bowl)')
    total_df = excel_data.parse(f'{team_alias}(Total)')
    
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
    print("Total NOT_PLAYING Players:", len(not_playing))
    print("NOT_PLAYING Players List:", not_playing)
    
    top_team1 = top_team1[~top_team1['Player'].isin(not_playing)].copy()
    top_team2 = top_team2[~top_team2['Player'].isin(not_playing)].copy()
    
    combined_players = pd.concat([top_team1, top_team2]).sort_values(by='Predicted Score', ascending=False)
    
    selected_players = []
    selected_set = set()
    team_counts = {"team1": 0, "team2": 0}
    
    for _, player in combined_players.iterrows():
        player_name = player['Player']
        if player_name in selected_set:
            continue
        
        player_team = "team1" if player_name in top_team1['Player'].values else "team2"
        if team_counts[player_team] < 6:
            selected_players.append(player)
            selected_set.add(player_name)
            team_counts[player_team] += 1
        
        if len(selected_players) == 11:
            break
    
    # Ensure at least 5 players from each team
    if team_counts['team1'] < 5 or team_counts['team2'] < 5:
        raise ValueError("Final selection does not meet the minimum team constraint (5 players each).")
    
    # Captain and Vice-Captain selection
    batsmen_team1 = [p for p in selected_players if roles[p['Player']]['Player Type'] == 'BAT' and p['Player'] in top_team1['Player'].values]
    batsmen_team2 = [p for p in selected_players if roles[p['Player']]['Player Type'] == 'BAT' and p['Player'] in top_team2['Player'].values]
    
    if team_counts['team1'] == 6:
        captain = max(batsmen_team1, key=lambda x: x['Predicted Score'])
        vice_captain = max(batsmen_team2, key=lambda x: x['Predicted Score'])
    else:
        captain = max(batsmen_team2, key=lambda x: x['Predicted Score'])
        vice_captain = max(batsmen_team1, key=lambda x: x['Predicted Score'])
    
    final_df = pd.DataFrame(selected_players)
    final_df['Role'] = "Player"
    final_df.loc[final_df['Player'] == captain['Player'], 'Role'] = "Captain"
    final_df.loc[final_df['Player'] == vice_captain['Player'], 'Role'] = "Vice-Captain"
    
    return final_df

if __name__ == "__main__":
    stats_file = "./CricketStats-Dream11Hackathon.xlsx"
    roles_file = "./SquadPlayerNames.xlsx"
    
    roles, teams = load_roles(roles_file)
    if len(teams) < 2:
        raise ValueError("Insufficient teams detected in roles file.")
    
    excel_data = load_data(stats_file)
    
    data_team1 = preprocess_data(excel_data, teams[0])
    data_team2 = preprocess_data(excel_data, teams[1])
    
    top_team1 = train_model(data_team1)
    top_team2 = train_model(data_team2)
    
    final_team = select_top_11(top_team1, top_team2, roles)
    
    print("Final Top 11 Players:")
    print(final_team)

