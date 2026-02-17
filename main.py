import joblib
import pandas as pd

def main():
    # ✅ Doğru yolu ver
    saved_data = joblib.load("models/model.pkl")

    model = saved_data["model"]

    # ✅ CSV oku
    df = pd.read_csv("test_scaled.csv")

    # ✅ İlk 5 satırı al

    # ✅ Predict yap
    predictions = model.predict(df[:5])

    print("First 5 predictions:")
    print(predictions)

if __name__ == "__main__":
    main()
