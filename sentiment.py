from transformers import pipeline 

pipe = pipeline("sentiment-analysis",  model="j-hartmann/emotion-english-distilroberta-base")

def analyze_sentiment(input):
    result = pipe(input)
    #confident threshold
    if result[0]['score'] < 0.6 :
        return{"label":"neutral","score":result[0]['score']}
    return result[0]

test_texts = [
    "I'm so excited about this new opportunity!",
    "This is absolutely terrifying.",
    "I can't believe they did that to me.",
    "The weather today is quite pleasant."
]

for text in test_texts:
    print(f"Text: {text}")
    print(f"Emotion: {analyze_sentiment(text)}\n")
  

