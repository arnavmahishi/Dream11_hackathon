import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def load_data(file_path):
    """Load the Excel file and return an ExcelFile object."""
    return pd.ExcelFile(file_path)

def load_roles(file_path):
    """Load player roles from the squad sheet, ensuring correct role assignment."""
    roles_df = pd.read_excel(file_path, sheet_name="Data_20250224")

    print("Columns in roles_df:", roles_df.columns.tolist())  # Debugging
    print("Unique values in 'IsPlaying':", roles_df['IsPlaying'].unique())  # Debugging

    if {'Player Name', 'Player Type', 'IsPlaying'}.issubset(roles_df.columns):
        # Ensure consistent string conversion before mapping
        roles_df['IsPlaying'] = roles_df['IsPlaying'].astype(str).str.strip().str.lower()
        roles_df['IsPlaying'] = roles_df['IsPlaying'].map({'playing': True, 'not_playing': False})


        # Convert into dictionary for easy lookup
        roles = roles_df.set_index('Player Name')[['Player Type', 'IsPlaying']].to_dict(orient='index')

        return roles
    else:
        raise KeyError("Expected columns 'Player Name', 'Player Type', and 'IsPlaying' not found in roles file.")

def preprocess_data(excel_data, team_name):
    """Merge batting, bowling, and total stats into one DataFrame."""
    batting_df = excel_data.parse(f'{team_name}(Bat)')
    bowling_df = excel_data.parse(f'{team_name}(Bowl)')
    total_df = excel_data.parse(f'{team_name}(Total)')
    
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
    """Selects the best 11 players while ensuring NOT_PLAYING players are excluded and constraints are met."""

    # **Filter out NOT_PLAYING players**
    not_playing = {p for p, r in roles.items() if not r["IsPlaying"]}

    print("Total NOT_PLAYING Players:", len(not_playing))  # Debugging
    print("NOT_PLAYING Players List:", not_playing)  # Debugging

    # Filter out NOT_PLAYING players before selection
    top_team1 = top_team1[~top_team1['Player'].isin(not_playing)].copy()
    top_team2 = top_team2[~top_team2['Player'].isin(not_playing)].copy()

    # Continue with selection logic...


    # Merge and sort by Predicted Score
    combined_players = pd.concat([top_team1, top_team2]).sort_values(by='Predicted Score', ascending=False)

    selected_players = []
    selected_set = set()
    bowlers, all_rounders, wk = [], [], []
    team1_bowlers, team2_bowlers = [], []
    team1_all, team2_all = [], []
    team1_count, team2_count = 0, 0

    for _, player in combined_players.iterrows():
        player_name = player['Player']

        # STRICTLY EXCLUDE NOT_PLAYING PLAYERS
        if player_name in selected_set or player_name in not_playing:
            continue

        role = roles.get(player_name, {}).get("Player Type", "BAT")  # ✅ Correct role lookup
        player_team = "team1" if player_name in top_team1['Player'].values else "team2"

        if role == 'BOWL':
            bowlers.append(player)
            if player_team == "team1":
                team1_bowlers.append(player)
            else:
                team2_bowlers.append(player)

        elif role == 'ALL':
            all_rounders.append(player)
            if player_team == "team1":
                team1_all.append(player)
            else:
                team2_all.append(player)

        elif role == 'WK':
            wk.append(player)

        # Ensure balance of players from both teams
        if (player_team == "team1" and team1_count < 4) or (player_team == "team2" and team2_count < 4):
            selected_players.append(player)
            selected_set.add(player_name)
            if player_team == "team1":
                team1_count += 1
            else:
                team2_count += 1

    # Ensure at least 2 total bowlers and 1 from each team
    while len([p for p in selected_players if roles.get(p['Player'], {}).get("Player Type", "BAT") == 'BOWL']) < 2 and bowlers:
        player = bowlers.pop(0)
        if player['Player'] not in selected_set:
            selected_players.append(player)
            selected_set.add(player['Player'])

    if team1_bowlers and not any(p['Player'] in pd.DataFrame(team1_bowlers)['Player'].values for p in selected_players):
        player = team1_bowlers.pop(0)
        if player['Player'] not in selected_set:
            selected_players.append(player)
            selected_set.add(player['Player'])

    if team2_bowlers and not any(p['Player'] in pd.DataFrame(team2_bowlers)['Player'].values for p in selected_players):
        player = team2_bowlers.pop(0)
        if player['Player'] not in selected_set:
            selected_players.append(player)
            selected_set.add(player['Player'])

    # Ensure at least 3 total all-rounders and 1 from each team
    while len([p for p in selected_players if roles.get(p['Player'], {}).get("Player Type", "BAT") == 'ALL']) < 3 and all_rounders:
        player = all_rounders.pop(0)
        if player['Player'] not in selected_set:
            selected_players.append(player)
            selected_set.add(player['Player'])

    if team1_all and not any(p['Player'] in pd.DataFrame(team1_all)['Player'].values for p in selected_players):
        player = team1_all.pop(0)
        if player['Player'] not in selected_set:
            selected_players.append(player)
            selected_set.add(player['Player'])

    if team2_all and not any(p['Player'] in pd.DataFrame(team2_all)['Player'].values for p in selected_players):
        player = team2_all.pop(0)
        if player['Player'] not in selected_set:
            selected_players.append(player)
            selected_set.add(player['Player'])

    # Ensure at least 1 wicketkeeper
    if wk and not any(roles.get(p['Player'], {}).get("Player Type", "BAT") == 'WK' for p in selected_players):
        player = wk.pop(0)
        if player['Player'] not in selected_set:
            selected_players.append(player)
            selected_set.add(player['Player'])

    # Trim to exactly 11 players
    return pd.DataFrame(selected_players[:11])


if __name__ == "__main__":
    stats_file = "./CricketStats-Dream11Hackathon.xlsx"
    roles_file = "./SquadPlayerNames.xlsx"
    team1, team2 = "New Zealand", "Bangladesh"
    
    excel_data = load_data(stats_file)
    roles = load_roles(roles_file)
    
    data_team1 = preprocess_data(excel_data, team1)
    data_team2 = preprocess_data(excel_data, team2)
    
    top_team1 = train_model(data_team1)
    top_team2 = train_model(data_team2)
    
    final_team = select_top_11(top_team1, top_team2, roles)
    
    print("Final Top 11 Players:")
    print(final_team)

