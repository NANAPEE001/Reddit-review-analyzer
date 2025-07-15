import praw
import pandas as pd
from datetime import datetime
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

#load Environment Variables 
load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")
REDDIT_USER_AGENT = "ReviewAnalyzerBot/1.0"

AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

#Fetch Reddit Comments
def fetch_reddit_comments(post_id):
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        username=REDDIT_USERNAME,
        password=REDDIT_PASSWORD,
        user_agent=REDDIT_USER_AGENT
    )

    submission = reddit.submission(id=post_id)
    submission.comments.replace_more(limit=0)

    data = []
    for comment in submission.comments.list():
        data.append({
            "author": str(comment.author),
            "text": comment.body,
            "score": comment.score,
            "created_utc": datetime.utcfromtimestamp(comment.created_utc).isoformat()
        })

    return pd.DataFrame(data)

# Azure NLP in Batches
def azure_nlp_analysis(df):
    credential = AzureKeyCredential(AZURE_LANGUAGE_KEY)
    client = TextAnalyticsClient(endpoint=AZURE_LANGUAGE_ENDPOINT, credential=credential)

    texts = df["text"].tolist()
    batch_size = 10
    sentiment_results_all = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        sentiment_batch = client.analyze_sentiment(documents=batch)
        key_phrase_batch = client.extract_key_phrases(documents=batch)

        for j in range(len(batch)):
            sentiment = sentiment_batch[j]
            key_phrases = key_phrase_batch[j]

            sentiment_results_all.append({
                "sentiment": sentiment.sentiment,
                "positive_score": sentiment.confidence_scores.positive,
                "neutral_score": sentiment.confidence_scores.neutral,
                "negative_score": sentiment.confidence_scores.negative,
                "key_phrases": ", ".join(key_phrases.key_phrases if not key_phrases.is_error else [])
            })

    return pd.DataFrame(sentiment_results_all)

# GPT-4.1 AI Response Generator 
def generate_responses(texts):
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version="2023-05-15",
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

    responses = []
    for text in texts:
        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a polite customer support agent."},
                    {"role": "user", "content": f"Generate a professional reply to this Reddit comment: '{text}'"}
                ],
                temperature=0.6
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"Error: {e}"
        responses.append(reply)

    return responses

# Pipeline Runner 
def run_pipeline(reddit_post_url):
    post_id = reddit_post_url.split("/comments/")[1].split("/")[0]
    print(f"Fetching comments from post ID: {post_id}")

    df = fetch_reddit_comments(post_id)
    print(f"Fetched {len(df)} comments")

    print("Analyzing sentiment and extracting key phrases...")
    nlp_df = azure_nlp_analysis(df)

    print("Generating GPT-4.1 responses...")
    df["ai_response"] = generate_responses(df["text"].tolist())

    print("Combining results...")
    result = pd.concat([df, nlp_df], axis=1)

    filename = f"reddit_analysis_{post_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result.to_csv(filename, index=False)
    print(f" Done! Results saved to: {filename}")


if __name__ == "__main__":
    reddit_post = "https://www.reddit.com/r/cocacola/comments/1j5g5ib/opinions_on_this_one/"
    run_pipeline(reddit_post)

