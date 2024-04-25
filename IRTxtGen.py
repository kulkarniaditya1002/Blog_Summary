import streamlit as st
import spacy
from newspaper import Article
from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import Counter
from heapq import nlargest
from rouge import Rouge

# Define a function to load different models based on user selection
def load_model(model_name):
    if "t5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        nlp = None  # Not used for T5 models
    else:  # Assume spaCy model
        tokenizer = None
        model = None
        nlp = spacy.load(model_name)
    return tokenizer, model, nlp

def fetch_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def summarize_text_t5(tokenizer, model, text, max_length, min_length, length_penalty, num_beams, no_repeat_ngram_size):
    preprocess_text = "summarize: " + text.strip().replace("\n", " ").replace("\\n", " ")
    tokenized_text = tokenizer.encode(preprocess_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        tokenized_text,
        max_length=max_length,
        min_length=min_length,
        length_penalty=length_penalty,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_text_extractive(nlp, text, num_sentences):
    doc = nlp(text)
    word_freq = Counter(token.text.lower() for token in doc if not token.is_stop and not token.is_punct)
    sentence_strength = {sent: sum(word_freq[word.text.lower()] for word in sent) for sent in doc.sents}
    return ' '.join(sent.text for sent in nlargest(num_sentences, sentence_strength, key=sentence_strength.get))

def evaluate_summary(rouge, reference, candidate):
    return rouge.get_scores(candidate, reference)

def main():
    st.title("Blog Summarizer")

    model_options = ["t5-small", "t5-base", "en_core_web_sm"]
    model_choice = st.selectbox("Select the model for summarization:", model_options)

    tokenizer, model, nlp = load_model(model_choice)

    url = st.text_input("Enter the URL of the blog you want to summarize:")
    if url:
        with st.spinner('Fetching and summarizing the blog...'):
            article_text = fetch_article(url)
            if "t5" in model_choice:
                max_length = st.slider("Max Length", 100, 1000, 500)
                min_length = st.slider("Min Length", 10, 500, 100)
                summary = summarize_text_t5(tokenizer, model, article_text, max_length, min_length, 5.0, 4, 3)
            else:
                num_sentences = st.slider("Number of Sentences for Extractive Summary", 1, 10, 3)
                summary = summarize_text_extractive(nlp, article_text, num_sentences)
            
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
