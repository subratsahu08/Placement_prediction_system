from flask import Flask, render_template, request
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

# Create Flask app
app = Flask(__name__)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    cgpa = float(request.form["cgpa"])
    internships = int(request.form["internships"])
    projects = int(request.form["projects"])
    workshops = int(request.form["workshops"])
    aptitude = float(request.form["aptitude"])
    softskills = float(request.form["softskills"])
    extracurricular = int(request.form["extracurricular"])
    training = int(request.form["training"])
    ssc = float(request.form["ssc"])
    hsc = float(request.form["hsc"])

    # Arrange in same order as training
    data = np.array([[
        cgpa,
        internships,
        projects,
        workshops,
        aptitude,
        softskills,
        extracurricular,
        training,
        ssc,
        hsc
    ]])

    # Predict
    prediction = model.predict(data)[0]

    # Probability
    probability = model.predict_proba(data)[0]

    if prediction == 1:
        result = "🎉 Student is Likely to be Placed"
        confidence = probability[1] * 100
    else:
        result = "❌ Student is Not Likely to be Placed"
        confidence = probability[0] * 100

    return render_template(
        "result.html",
        prediction=result,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)