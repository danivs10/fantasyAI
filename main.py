import asyncio
import json
import aiohttp
from understat import Understat
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

async def playerMatchesStats(id):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        player_matches = await understat.get_player_matches(id)
        #print(json.dumps(player_matches))
        return player_matches

async def getPlayerStats():
    async with aiohttp.ClientSession(id) as session:
        understat = Understat(session)
        grouped_stats = await understat.get_player_grouped_stats(id)
        #print(json.dumps(grouped_stats))
        return grouped_stats

async def getLeaguePlayers():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        players = await understat.get_league_players(
            "La liga",
            2023
        )
        return players

def calculate_weights(num_games, decay_factor):
    weights = np.exp(-decay_factor * np.arange(num_games))
    print(weights)
    return weights / np.sum(weights)

def calculate_avg_weights(num_games, decay_factor):
    weights = [decay_factor ** i for i in range(num_games)]
    print(weights)
    return weights 


def make_prediction(stats):
    # Split the data into training and testing sets
    X = list(zip(
        stats["goals"],
        stats["shots"],
        stats["xG"],
        stats["time"],
        stats["xA"],
        stats["assists"],
        stats["key_passes"],
        stats["npg"],
        stats["npxG"],
        stats["xGChain"],
        stats["xGBuildup"],
        stats["h_team"],
        stats["a_team"],
        stats["is_home"]
    ))
    y = {
        "goals": stats["goals"],
        "shots": stats["shots"],
        "xG": stats["xG"],
        "time": stats["time"],
        "xA": stats["xA"],
        "assists": stats["assists"],
        "key_passes": stats["key_passes"],
        "npg": stats["npg"],
        "npxG": stats["npxG"],
        "xGChain": stats["xGChain"],
        "xGBuildup": stats["xGBuildup"]
    }
    y_df = pd.DataFrame(y)


    # Create a random forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y_df, test_size=0.2, shuffle=False)

    # Calculate the weights based on the time difference between each game and the most recent game
    num_games = len(X_train)
    weights = calculate_weights(num_games, 0.7)

    # Train the model with the weighted data
    model.fit(X_train, y_train, sample_weight=weights)

    predictions = model.predict(X_test)

    # Calculate the absolute errors
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")

    avg_weights = calculate_avg_weights(len(X), 0.7)
    # Calculate the weighted average stats

    avg_stats = {
        "goals": np.average(stats["goals"], weights=avg_weights),
        "shots": np.average(stats["shots"], weights=avg_weights),
        "xG": np.average(stats["xG"], weights=avg_weights),
        "time": np.average(stats["time"], weights=avg_weights),
        "xA": np.average(stats["xA"], weights=avg_weights),
        "assists": np.average(stats["assists"], weights=avg_weights),
        "key_passes": np.average(stats["key_passes"], weights=avg_weights),
        "npg": np.average(stats["npg"], weights=avg_weights),
        "npxG": np.average(stats["npxG"], weights=avg_weights),
        "xGChain": np.average(stats["xGChain"], weights=avg_weights),
        "xGBuildup": np.average(stats["xGBuildup"], weights=avg_weights)
    }

    # Convert your stats to the same format as your training data
    stats_input = [avg_stats["goals"], avg_stats["shots"], avg_stats["xG"], avg_stats["time"], avg_stats["xA"], avg_stats["assists"], avg_stats["key_passes"], avg_stats["npg"], avg_stats["npxG"], avg_stats["xGChain"], avg_stats["xGBuildup"], 31, 52, 1]

    # Make a prediction for the next match
    next_match_prediction = model.predict([stats_input])

    # Define the factors for each parameter
    factors = {
        "goals": 3,
        "shots": 1,
        "xG": 1,
        "time": 0.01,
        "xA": 1,
        "assists": 2,
        "key_passes": 1.5,
        "npg": 0,
        "npxG": 0,
        "xGChain": 1,
        "xGBuildup": 1
    }

    # Calculate the overall score based on the average stats
    next_match_prediction = next_match_prediction[0]
    score = 0
    next_match_prediction_dict = {
        "goals": next_match_prediction[0],
        "shots": next_match_prediction[1],
        "xG": next_match_prediction[2],
        "time": next_match_prediction[3],
        "xA": next_match_prediction[4],
        "assists": next_match_prediction[5],
        "key_passes": next_match_prediction[6],
        "npg": next_match_prediction[7],
        "npxG": next_match_prediction[8],
        "xGChain": next_match_prediction[9],
        "xGBuildup": next_match_prediction[10]
    }
    for key, value in next_match_prediction_dict.items():
        score += value * factors[key]

    print(f"Player score: {score}")
    # add the score also to the next_match_prediction_dict
    next_match_prediction_dict["score"] = score
    return next_match_prediction

async def get_player(name):
    players = await getLeaguePlayers()
    filtered_players = [player for player in players if name in player["player_name"].lower()]
    return filtered_players[0]

async def main():

    players = await getLeaguePlayers()
    filtered_players = [player for player in players if "kubo" in player["player_name"].lower()]
     
    player = filtered_players[0]
    matchesStats = await playerMatchesStats(player['id'])

    match_stats = {
        "goals": [],
        "shots": [],
        "xG": [],
        "time": [],
        "xA": [],
        "assists": [],
        "key_passes": [],
        "npg": [],
        "npxG": [],
        "xGChain": [],
        "xGBuildup": [],
        "h_team": [],
        "a_team": [],
    }

    for match in matchesStats:
        match_stats["goals"].append(int(match["goals"]))
        match_stats["shots"].append(int(match["shots"]))
        match_stats["xG"].append(float(match["xG"]))
        match_stats["time"].append(int(match["time"]))
        match_stats["xA"].append(float(match["xA"]))
        match_stats["assists"].append(int(match["assists"]))
        match_stats["key_passes"].append(int(match["key_passes"]))
        match_stats["npg"].append(int(match["npg"]))
        match_stats["npxG"].append(float(match["npxG"]))
        match_stats["xGChain"].append(float(match["xGChain"]))
        match_stats["xGBuildup"].append(float(match["xGBuildup"]))
        match_stats["h_team"].append(match["h_team"])
        match_stats["a_team"].append(match["a_team"])

    # Convert match_stats dictionary to a DataFrame
    df = pd.DataFrame(match_stats)

    # Get unique team names from h_team and a_team columns
    teams = df["h_team"].unique().tolist() + df["a_team"].unique().tolist()

    # Create a mapping dictionary to assign a number to each team name
    team_mapping = {team: i for i, team in enumerate(teams)}

    # Replace team names with their corresponding numbers
    df["h_team"] = df["h_team"].map(team_mapping)
    df["a_team"] = df["a_team"].map(team_mapping)

    filtered_players[0]['team_title']

    # Get the team number of the player's team
    player_team = team_mapping[filtered_players[0]['team_title']]

    # Add a column that indicates whether the team playing home is the player's team
    df["is_home"] = df["h_team"] == player_team

    # Convert back to dictionary
    match_stats = df.to_dict(orient="list")

    # Make predictions for each category based on the last match stats
    prediction = make_prediction(match_stats)
    np.set_printoptions(suppress=True)
    print(prediction)

asyncio.run(main())
