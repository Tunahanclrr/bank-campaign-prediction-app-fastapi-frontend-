from fastapi import FastAPI, Request
import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

##templates
templates = Jinja2Templates(directory="templates")

# ✅ Modeli uygulama başlarken yükle
saved_data = joblib.load("models/model.pkl")
model = saved_data["model"]


# ✅ Test endpoint
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class ModelFeature(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str


@app.post("/predict")
async def predict(features: ModelFeature):

    # 🔥 JSON → DataFrame
    input_data = pd.DataFrame([features.model_dump()])
    print("Input data:")
    print(input_data)

    # 🔥 Prediction
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    # 🔥 Olasılıklar
    prob_0 = float(probabilities[0][0])
    prob_1 = float(probabilities[0][1])

    print("Prediction result:", int(prediction[0]))
    print("Probability class 0:", prob_0)
    print("Probability class 1:", prob_1)
    print("-" * 40)

    return {
        "prediction": int(prediction[0]),
        "probability_0": prob_0,
        "probability_1": prob_1
    }
