# 34

# **Natural Language Processing (NLP)**

Natural Language Processing (NLP) is a branch of **Artificial Intelligence (AI)** that enables computers to understand, interpret, process, and generate human language.

It combines:

* Computer Science  
* Linguistics  
* Machine Learning  
* Deep Learning

The goal is to make machines communicate with humans in natural language.

---

# **Intuition Behind NLP**

Humans understand:

"I am feeling happy today."

instantly.

For a computer, this is just:

Characters  
Words  
Tokens

NLP converts human language into a format machines can understand and process.

---

# **Real-Life Examples of NLP**

### **ChatGPT**

Question → Understand → Generate Response

---

### **Google Translate**

English → Hindi

---

### **Siri / Alexa**

Speech → Text → Meaning → Action

---

### **Gmail Spam Filter**

Email → Spam / Not Spam

---

# **NLP Pipeline**

Typically NLP involves:

Text  
 ↓  
Preprocessing  
 ↓  
Feature Extraction  
 ↓  
Model Training  
 ↓  
Prediction

---

# **Types of NLP Tasks**

---

## **1\. Text Classification**

Assign text to categories.

Examples:

Spam Detection  
Sentiment Analysis  
News Classification

Input:

"I love this movie."

Output:

Positive

---

## **2\. Sentiment Analysis**

Determines emotional tone.

Example:

"This phone is amazing."

Output:

Positive

---

## **3\. Machine Translation**

Converts one language into another.

Example:

English → French

---

## **4\. Named Entity Recognition (NER)**

Identifies important entities.

Example:

Elon Musk founded SpaceX.

Entities:

Elon Musk → Person  
SpaceX → Organization

---

## **5\. Question Answering**

Example:

Q: Capital of India?  
A: New Delhi

---

## **6\. Text Summarization**

Converts long documents into short summaries.

---

## **7\. Speech Recognition**

Example:

Voice → Text

Used in:

* Siri  
* Alexa  
* Google Assistant

---

## **8\. Text Generation**

Example:

Prompt → Story

Used by:

* ChatGPT  
* Gemini  
* Claude

---

# **NLP Preprocessing**

Before training models, text must be cleaned.

---

## **Tokenization**

Split text into words.

Example:

"I love NLP"

becomes

\["I","love","NLP"\]

---

## **Stop Word Removal**

Remove common words.

Example:

the  
is  
are  
was

---

## **Stemming**

Reduces words to their root form.

Example:

| Word | Stem |
| ----- | ----- |
| Playing | Play |
| Played | Play |
| Plays | Play |

---

### **Example**

running  
runner  
runs

becomes

run

Popular stemmer:

Porter Stemmer

---

## **Lemmatization**

More advanced than stemming.

Example:

better

becomes

good

Produces meaningful root words.

---

# **Bag of Words (BoW)**

One of the most important classical NLP techniques.

---

# **Intuition of Bag of Words**

Suppose we have sentences:

I love AI  
AI is awesome

Create vocabulary:

\[I, Love, AI, Is, Awesome\]

Now count word occurrences.

---

## **Sentence 1**

I love AI

Vector:

\[1,1,1,0,0\]

---

## **Sentence 2**

AI is awesome

Vector:

\[0,0,1,1,1\]

---

# **Advantages of BoW**

* Simple  
* Fast  
* Easy implementation

---

# **Limitations of BoW**

Ignores:

Word order  
Grammar  
Context  
Meaning

Example:

Dog bites man  
Man bites dog

Same BoW representation.

---

# **Classical NLP Models**

Traditional NLP approaches before Deep Learning.

---

## **Common Feature Extraction Methods**

### **Bag of Words (BoW)**

Counts words.

---

### **TF-IDF**

Measures importance of words.

---

### **N-Grams**

Captures neighboring words.

Example:

Machine Learning

Bigram:

Machine Learning

---

# **Classical Machine Learning Models**

Used with BoW or TF-IDF.

Examples:

### **Naive Bayes**

Spam Detection.

---

### **Logistic Regression**

Sentiment Analysis.

---

### **SVM**

Text Classification.

---

### **Random Forest**

Document Classification.

---

# **Deep Learning in NLP**

Deep Learning automatically learns text representations without manual feature engineering.

Instead of:

BoW  
TF-IDF

it learns:

Embeddings  
Semantic Meaning  
Context

---

# **Why Deep Learning?**

Classical NLP struggles with:

Context  
Polysemy  
Long Dependencies  
Meaning

Example:

Apple released a new phone.

vs

I ate an apple.

Deep Learning understands the difference.

---

# **Word Embeddings**

Words represented as dense vectors.

Example:

King  
Queen  
Man  
Woman

similar words appear close together.

Popular embeddings:

* Word2Vec  
* GloVe  
* FastText

---

# **Deep Learning Models for NLP**

---

## **1\. RNN (Recurrent Neural Networks)**

Processes sequence one word at a time.

Example:

I love Machine Learning

Reads words sequentially.

---

## **Problem**

RNN suffers from:

Vanishing Gradient  
Long-Term Dependency Issues

---

## **2\. LSTM**

Long Short-Term Memory Network.

Designed to remember long-term information.

Used in:

* Translation  
* Speech Recognition

---

## **3\. GRU**

Simplified version of LSTM.

Faster training.

---

## **4\. Transformers**

Modern NLP architecture.

Examples:

* BERT  
* GPT  
* Gemini  
* Claude

Most state-of-the-art NLP systems use Transformers.

---

# **Sequence-to-Sequence (Seq2Seq) Models**

Seq2Seq models convert one sequence into another sequence.

---

# **Intuition**

Input:

English Sentence

Output:

French Sentence

Example:

Hello World

↓

Bonjour le Monde

---

# **Architecture**

### **Encoder**

Reads input sequence.

---

### **Decoder**

Generates output sequence.

---

Diagram:

Input Sentence  
      ↓  
   Encoder  
      ↓  
Context Vector  
      ↓  
   Decoder  
      ↓  
Output Sentence

---

# **Applications of Seq2Seq**

### **Machine Translation**

English → French

---

### **Text Summarization**

Long Article → Summary

---

### **Chatbots**

Question → Response

---

### **Speech Recognition**

Audio → Text

---

# **Classical NLP vs Deep Learning NLP**

| Feature | Classical NLP | Deep Learning NLP |
| ----- | ----- | ----- |
| Feature Engineering | Manual | Automatic |
| Context Understanding | Poor | Excellent |
| Data Requirement | Small | Large |
| Training Time | Fast | Slow |
| Accuracy | Lower | Higher |
| Interpretability | Higher | Lower |
| Examples | BoW, TF-IDF, SVM | LSTM, BERT, GPT |

---

# **Use Cases of NLP**

## **Healthcare**

* Medical report analysis  
* Disease prediction

---

## **Finance**

* Fraud detection  
* Sentiment analysis

---

## **E-Commerce**

* Product recommendations  
* Review analysis

---

## **Customer Support**

* Chatbots  
* Virtual assistants

---

## **Search Engines**

* Query understanding  
* Ranking results

---

## **Social Media**

* Toxic comment detection  
* Sentiment analysis

---

# **Interview Summary**

**Natural Language Processing (NLP)** is the field of AI that enables machines to understand and generate human language. Traditional NLP relies on preprocessing techniques such as **Tokenization, Stop Word Removal, Stemming, Lemmatization**, and feature extraction methods like **Bag of Words (BoW), TF-IDF, and N-Grams**, combined with machine learning algorithms such as **Naive Bayes, Logistic Regression, and SVM**. Modern NLP uses **Deep Learning**, including **RNNs, LSTMs, GRUs, and Transformers**, which automatically learn contextual representations of language. **Sequence-to-Sequence (Seq2Seq)** models use an Encoder-Decoder architecture for tasks like machine translation, summarization, and chatbots. Today, Transformer-based models such as **BERT** and **GPT** dominate NLP due to their superior ability to capture context and meaning.

