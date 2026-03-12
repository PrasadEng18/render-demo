from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("sales_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Get values from form
    tv = float(request.form["tv"])
    radio = float(request.form["radio"])
    newspaper = float(request.form["newspaper"])

    # Convert into numpy array
    features = np.array([[tv, radio, newspaper]])

    # Prediction
    prediction = model.predict(features)

    # Convert prediction to scalar
    result = prediction.item()

    return render_template(
        "index.html",
        prediction_text=f"Predicted Sales = {round(result,2)}"
    )

if __name__ == "__main__":
    app.run(debug=True)

