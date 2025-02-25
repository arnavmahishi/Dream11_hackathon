import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def load_data(file_path):
    return pd.ExcelFile(file_path)

def load_roles(file_path):
    roles_df = pd.read_excel(file_path, sheet_name="Data_20250224")
    
    roles_df['IsPlaying'] = roles_df['IsPlaying'].astype(str).str.strip().str.lower()
    roles_df['IsPlaying'] = roles_df['IsPlaying'].map({'playing': True, 'not_playing': False})
    
    roles = roles_df.set_index('Player Name')[['Player Type', 'IsPlaying']].to_dict(orient='index')
    return roles

def preprocess_data(excel_data, team_name):
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
    team1_count, team2_count = 0, 0

    # Role constraints
    role_counts = {'BOWL': 0, 'ALL': 0, 'WK': 0}
    min_constraints = {'BOWL': 2, 'ALL': 3, 'WK': 1}
    team_role_counts = {'team1': {'BOWL': 0, 'ALL': 0}, 'team2': {'BOWL': 0, 'ALL': 0}}

    for _, player in combined_players.iterrows():
        player_name = player['Player']
        if player_name in selected_set:
            continue

        role = roles.get(player_name, {}).get("Player Type", "BAT")
        player_team = "team1" if player_name in top_team1['Player'].values else "team2"

        # Enforce team balance (at least 5 players from each team)
        if (team1_count < 5 and player_team == "team1") or (team2_count < 5 and player_team == "team2") :
            if role in min_constraints:
                if role_counts[role] >= min_constraints[role]:
                    continue  # Skip if the role quota is already met
                role_counts[role] += 1
                if role in team_role_counts[player_team]:
                    team_role_counts[player_team][role] += 1

            selected_players.append(player)
            selected_set.add(player_name)
            if player_team == "team1":
                team1_count += 1
            else:
                team2_count += 1

        if len(selected_players) == 11:
            break

    # Ensure exactly 11 players are selected
    while len(selected_players) < 11:
        remaining_players = combined_players[~combined_players['Player'].isin(selected_set)]
        if not remaining_players.empty:
            selected_players.append(remaining_players.iloc[0])
            selected_set.add(remaining_players.iloc[0]['Player'])

    # Convert selected players to DataFrame
    final_df = pd.DataFrame(selected_players[:11])

    # Determine which team has 6 players and which has 5
    team1_final_count = sum(final_df['Player'].isin(top_team1['Player']))
    team2_final_count = sum(final_df['Player'].isin(top_team2['Player']))

    team_with_6 = "team1" if team1_final_count == 6 else "team2"
    team_with_5 = "team2" if team_with_6 == "team1" else "team1"

    print("\nFinal Team Composition:")
    print(f"Players from Team1: {team1_final_count}, Players from Team2: {team2_final_count}")

    # Select Captain and Vice-Captain
    final_df['Role'] = "Player"

    def assign_captain_vc(team_label, role):
        """Finds the highest scoring BAT from the given team."""
        return final_df[(final_df['Player'].isin(top_team1['Player'] if team_label == "team1" else top_team2['Player']))
                        & (final_df['Player'].map(lambda x: roles.get(x, {}).get("Player Type", "")) == "BAT")]\
                        .sort_values(by='Predicted Score', ascending=False)

    captain_candidates = assign_captain_vc(team_with_6, "BAT")
    vice_captain_candidates = assign_captain_vc(team_with_5, "BAT")

    if not captain_candidates.empty:
        final_df.loc[final_df['Player'] == captain_candidates.iloc[0]['Player'], 'Role'] = 'Captain (C)'

    if not vice_captain_candidates.empty:
        final_df.loc[final_df['Player'] == vice_captain_candidates.iloc[0]['Player'], 'Role'] = 'Vice Captain (VC)'

    

    return final_df



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

