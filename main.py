import os
from typing import Union

from fastapi import FastAPI, Form, Request
from starlette.responses import HTMLResponse
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained pipeline ONCE at app startup
pipeline = joblib.load("titanic_pipeline.joblib")
print("✅ Titanic model loaded successfully!")

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the index.html form."""
    with open("index.html") as f:
        return HTMLResponse(f.read())

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    pclass: int = Form(...),
    sex: str = Form(...),
    age: float = Form(...),
    fare: float = Form(...),
    embarked: str = Form(...),
    deck: str = Form(...),
    alone: int = Form(...),
    sibsp: int = Form(...),
    parch: int = Form(...),
):
    """Handle POST from the form and return prediction."""

    # Input validation (basic)
    allowed_decks = {"A", "B", "C", "D", "E", "F", "G"}
    allowed_embarked = {"C", "Q", "S"}

    if deck not in allowed_decks or embarked not in allowed_embarked:
        return HTMLResponse(f"""
        <h1>Invalid input!</h1>
        <p>Deck or Port of Embarkation value is not allowed.</p>
        <p><a href="/">Try again</a></p>
        """, status_code=400)

    # Prepare input data as DataFrame row
    input_data = pd.DataFrame([{
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "fare": fare,
        "embarked": embarked,
        "deck": deck,
        "alone": alone,
        "sibsp": sibsp,
        "parch": parch
    }])

    # Log input
    print("Input data received:")
    print(input_data)

    try:
        # Predict hard class (1 = survived, 0 = not survived)
        y_pred = pipeline.predict(input_data)[0]

        # Predict probability
        y_proba = pipeline.predict_proba(input_data)[0]
        survival_prob = y_proba[1]  # Probability of class 1 (survived)

        # Log output
        print(f"Prediction: {y_pred}, Survival probability: {survival_prob:.2%}")

        # Build result text
        result_text = "🎉 You would have survived!" if y_pred == 1 else "💀 You would NOT have survived."

        # Convert inputs to HTML table for display
        input_summary = input_data.to_html(index=False, classes="input-summary", border=1)

        # Return result page with input recap
        return HTMLResponse(f"""
        <h2>Prediction Result</h2>
        <p>{result_text}</p>
        <p>Estimated probability of survival: {survival_prob:.2%}</p>
        <h2>Your inputs:</h2>
        {input_summary}
        <hr>
        <p><a href="/">🔄 Try again</a></p>
        """, headers={"Cache-Control": "no-cache"})

    except Exception as e:
        # Log error
        print(f"⚠️ Error during prediction: {e}")

        # Return an error message to user
        return HTMLResponse(f"""
        <h3>Error during prediction</h3>
        <p>{str(e)}</p>
        <p><a href="/">🔄 Try again</a></p>
        """, status_code=500)
