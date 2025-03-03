from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("match_predictor.pkl")

team_stats = pd.read_csv("premier_league_2023_24.csv")
head_to_head = pd.read_csv("head_to_head_stats.csv")

team_stats["Team"] = team_stats["Team"].str.strip()
head_to_head["team"] = head_to_head["team"].str.strip()
head_to_head["opponent"] = head_to_head["opponent"].str.strip()

def get_team_stats(team_name):
    row = team_stats[team_stats["Team"] == team_name]
    return row.iloc[0] if not row.empty else None

def get_head_to_head(home_team, away_team):
    row = head_to_head[(head_to_head["team"] == home_team) & (head_to_head["opponent"] == away_team)]
    return row.iloc[0] if not row.empty else {"head_to_head": 0, "head_to_head_goals": 0}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        data = request.json
        home_team = data.get("home_team")
        away_team = data.get("away_team")

        if not home_team or not away_team:
            return jsonify({"error": "Missing team names in request"}), 400


        home_stats = get_team_stats(home_team)
        away_stats = get_team_stats(away_team)
        h2h_stats = get_head_to_head(home_team, away_team)

        if home_stats is None or away_stats is None:
            return jsonify({"error": "Invalid team name"}), 400

        team_mapping = {team: idx + 1 for idx, team in enumerate(team_stats["Team"].unique())}
        opponent_encoded = team_mapping.get(away_team, 0) 
        
        features = [
            float(home_stats["Recent Form"]), 
            float(home_stats["Avg Goals/Game"]), 
            float(home_stats["Avg Goals Against/Game"] * 3.0),
            float(home_stats["Avg Shots/Game"]), 
            float(home_stats["Avg Possession (%)"]),
            float(h2h_stats["head_to_head"] * 2.0), 
            float(h2h_stats["head_to_head_goals"] * 2.0), 
            opponent_encoded, 
            0.2, 
        ]
        print("MODEL INPUT FEATURES:", features)

        prediction = model.predict([features])[0]

        
        result_mapping = {1: f"{home_team} wins", 0: "Draw", -1: f"{home_team} loses"}
        
        if home_team == away_team:
            prediction_text = "Choose two different teams"
        else:
            prediction_text = result_mapping.get(prediction, "Unknown")

        return jsonify({"home_team": home_team, "away_team": away_team, "prediction": prediction_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)



