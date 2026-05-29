import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
lr_model = pickle.load(open("lr_model.pkl", "rb"))
scalar = pickle.load(open("Min_Max_scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_api", methods=["POST"])
def predict_api():

    data = request.json["data"]

    # Encoding gender
    if data["gender"] == "Male":
        data["gender"] = 0
    else:
        data["gender"] = 1

    # Encoding platform
    platform_mapping = {
        "Instagram": 0,
        "TikTok": 1,
        "Both": 2
    }

    data["platform"] = platform_mapping[data["platform"]]

    # Encoding interaction level
    interaction_mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    data["interaction"] = interaction_mapping[data["interaction"]]

    print(data)

    input_data = np.array(list(data.values())).reshape(1, -1)

    new_data = scalar.transform(input_data)

    output = lr_model.predict(new_data)

    return jsonify(int(output[0]))


@app.route("/predict", methods=["POST"])
def predict():

    # Get form data
    gender = request.form["gender"]
    platform_usage= request.form["platform_usage"]
    social_interaction_level = request.form["social_interaction_level"]

    # Encoding gender
    gender = 0 if gender == "Male" else 1

    # Encoding platform
    platform_mapping = {
        "Instagram": 0,
        "TikTok": 1,
        "Both": 2
    }

    platform_usage = platform_mapping[platform_usage]

    # Encoding interaction level
    interaction_mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    social_interaction_level = interaction_mapping[social_interaction_level]

    # Get remaining numerical inputs
    age = float(request.form["age"])
   
    daily_social_media_hours = float(request.form["daily_social_media_hours"])
    sleep_hours = float(request.form["sleep_hours"])
    screen_time_before_sleep = float(request.form["screen_time_before_sleep"])
    academic_performance = float(request.form["academic_performance"])
    physical_activity = float(request.form["physical_activity"])
    stress_level = float(request.form["stress_level"])
    anxiety_level = float(request.form["anxiety_level"])
    addiction_level = float(request.form["addiction_level"])

    # Final input order
    data = [
        age,
        gender,
        daily_social_media_hours,
        platform_usage,
        sleep_hours,
        screen_time_before_sleep,
        academic_performance,
        physical_activity,
        social_interaction_level,
        stress_level,
        anxiety_level,
        addiction_level
    ]

    #
    final_input = scalar.transform(
    np.array(data, dtype=float).reshape(1, -1)
)

    output = lr_model.predict(final_input)[0]

    return render_template(
        "home.html",
        prediction_text="Prediction: {}".format(output)
    )


if __name__ == "__main__":
    app.run(debug=True)