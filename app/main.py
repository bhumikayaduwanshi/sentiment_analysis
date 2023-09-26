from transformers import pipeline
from fastapi import FastAPI

app = FastAPI()


@app.post("/sentiment-analysis/")
async def sentiment_analysis(text: str):
    try:
        # Load a pre-trained sentiment analysis model
        sentiment_classifier = pipeline("sentiment-analysis")

        # Perform sentiment analysis
        sentiment = sentiment_classifier(text)
        return {"result": sentiment}

    except ValueError:
        return {"error": "Invalid input. Please provide a valid string."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
