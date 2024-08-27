
# Text Summarization Tool

## Overview
The Technical Content Summarization Tool is designed to aid professionals, students, and researchers in the fields of data science and software engineering. Given the rapid advancements and the high volume of complex information in these areas, this tool leverages state-of-the-art Natural Language Processing (NLP) techniques to generate concise summaries of lengthy technical articles. This enables users to stay informed without the need to invest significant amounts of time in reading.

## Features
- **Abstractive and Extractive Summarization:** Combines the strengths of Google's T5 model for abstractive summarization and spaCy's en_core_web_sm model for extractive summarization.
- **ROUGE Evaluation:** Implements the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric to assess the quality of generated summaries.
- **Fine-tuned for Technical Content:** Specifically tuned for summarizing content related to data science and software engineering.

## Project Motivation
Professionals and students in fast-evolving fields like data science and software engineering face significant challenges due to the rapid pace of advancements and the sheer volume of detailed technical content. This tool was developed to address these challenges by providing a way to distill essential information from large volumes of content, making it easier to stay updated with minimal time investment.

## Methodology
### Models Used:
- **T5 (Text-to-Text Transformer):** A versatile model that approaches all NLP tasks as a text-to-text problem. Fine-tuned for technical content to generate concise yet informative summaries.
- **spaCy en_core_web_sm Model:** A lightweight, efficient NLP model used for extractive summarization, helping in selecting key sentences and phrases directly from the text.

### Data Sources
- **High-traffic blogs and websites:** Medium, Towards Data Science, and Amazon Blog posts were used as primary data sources.
- **Data Collection:** Articles were extracted using the `newspaper3k` Python library.

### Evaluation
The tool’s performance was evaluated using the ROUGE metric, which provides a comprehensive analysis of text summarization quality by measuring n-gram overlap, sentence structure, and sequence of words between the generated summary and reference texts.

## Results
- **T5 Generated Summary ROUGE Scores:**
  - Rouge-1: R: 0.27, P: 0.90, F: 0.41
  - Rouge-2: R: 0.14, P: 0.81, F: 0.24
  - Rouge-L: R: 0.27, P: 0.90, F: 0.41
- **spaCy Generated Summary ROUGE Scores:**
  - Rouge-1: R: 0.37, P: 1.00, F: 0.54
  - Rouge-2: R: 0.25, P: 0.98, F: 0.40
  - Rouge-L: R: 0.37, P: 1.00, F: 0.54

## Future Directions
- **Model Upgrades:** Explore more advanced models like T-11B or GPT-3 for deeper content understanding.
- **User Interface Enhancements:** Plans include multilingual support and personalized summary preferences to enhance user engagement and broaden applicability.

## Conclusion
This tool represents a significant step towards making technical content more accessible and easier to digest. While effective in reducing reading time and aiding in information retention, future upgrades and enhancements will focus on improving model depth, performance, and user experience.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/kulkarniaditya1002/Technical-Content-Summarization-Tool.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Blog_Summary
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the summarization tool:
   ```bash
   python IRTxtGen.py
   ```

