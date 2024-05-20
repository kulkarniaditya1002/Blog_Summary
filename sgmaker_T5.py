import streamlit as st
from newspaper import Article
from transformers import T5Tokenizer
from rouge import Rouge
import boto3
import json

# AWS settings
sagemaker_runtime = boto3.client('sagemaker-runtime')
endpoint_name = 'your-sagemaker-endpoint-name'  # Replace with your SageMaker endpoint name

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def fetch_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def summarize_text_t5_sagemaker(text, max_length, min_length, length_penalty, num_beams, no_repeat_ngram_size):
    payload = {
        "text": text,
        "max_length": max_length,
        "min_length": min_length,
        "length_penalty": length_penalty,
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size
    }
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    summary = json.loads(response['Body'].read().decode())['summary_text']
    return summary

def evaluate_summary(rouge, reference, candidate):
    return rouge.get_scores(candidate, reference)

def main():
    st.title("Blog Summarizer")

    url = st.text_input("Enter the URL of the blog you want to summarize:")
    if url:
        with st.spinner('Fetching and summarizing the blog...'):
            article_text = fetch_article(url)
            max_length = st.slider("Max Length", 100, 1000, 500)
            min_length = st.slider("Min Length", 10, 500, 100)
            summary = summarize_text_t5_sagemaker(article_text, max_length, min_length, 5.0, 4, 3)
            
            st.subheader("Generated Summary")
            st.write(summary)

            # Evaluation (ROUGE Scores)
            rouge = Rouge()
            if summary:
                scores = evaluate_summary(rouge, article_text, summary)
                st.subheader("ROUGE Scores Evaluation")
                st.json(scores)

if __name__ == "__main__":
    main()
