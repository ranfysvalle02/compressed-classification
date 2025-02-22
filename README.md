# compressed-classification

https://maxhalford.github.io/blog/text-classification-by-compression/

---

**Demystifying Text Classification: From Compression Techniques to Cosine Similarity**  
   
*Exploring Innovative Approaches in Natural Language Processing*  
   
---  
   
**Introduction**  
   
Text classification is a core task in Natural Language Processing (NLP), involving the assignment of natural language texts to predefined categories. It's widely used in areas like spam detection, sentiment analysis, and topic labeling.  
   
While traditional machine learning models are often employed for this task, there are unconventional methods that can offer fresh perspectives and insights. In this post, we'll explore two distinct approaches to text classification:  
   
- **Compression-Based Classification using Normalized Compression Distance (NCD)**  
- **Cosine Similarity with TF-IDF Embeddings**  
   
We'll implement both methods using the **20 Newsgroups** dataset, analyze their performance, and discuss the strengths and limitations of each approach.  
   
---  
   
**Understanding Compression-Based Classification**  
   
*What is Normalized Compression Distance (NCD)?*  
   
Normalized Compression Distance is a way to measure the similarity between two pieces of text based on compression. The basic idea is that if two texts share a lot of information, then compressing them together will not increase the size by much compared to compressing them separately.  
   
*Intuition Behind NCD*  
   
- **Shared Information**: Similar texts contain redundant information.  
- **Compression Exploits Redundancy**: Compression algorithms reduce the size of data by eliminating this redundancy.  
- **Measuring Similarity**: NCD quantifies how much extra information is needed when two texts are combined, revealing their similarity.  
   
*Implementing the Compression-Based Classifier*  
   
**Dataset Preparation**  
   
We'll use the **20 Newsgroups** dataset, focusing on these categories:  
   
- Computer Graphics  
- Recreational Sports Hockey  
- Science Space  
- Talk Politics Mideast  
   
**Preprocessing Steps**  
   
- Convert all text to lowercase.  
- Remove non-alphanumeric characters to eliminate punctuation and symbols.  
- Remove common English stopwords (like "the", "and", "is") to focus on meaningful words.  
   
**Classifier Logic**  
   
1. **Combine Training Texts per Category**: For each category, concatenate all the training documents into one large text block.  
2. **Compress Category Texts**: Use a compression algorithm (like zlib) to compress these combined texts.  
3. **Classify Test Documents**:  
   - Preprocess each test document in the same way.  
   - Calculate the NCD between the test document and each category text.  
   - Assign the document to the category with the lowest NCD value, indicating the highest similarity.  
   
**Results and Analysis**  
   
*Compression-Based Classifier Report*  
   
```  
Compression-Based Classifier Report:  
                               precision    recall  f1-score   support  
  
                comp.graphics       0.79      0.37      0.51       195  
             rec.sport.hockey       0.65      0.81      0.72       200  
                    sci.space       0.41      0.87      0.56       197  
        talk.politics.mideast       0.96      0.12      0.22       188  
  
                     accuracy                           0.55       780  
                    macro avg       0.70      0.54      0.50       780  
                 weighted avg       0.70      0.55      0.51       780  
```  
   
*Analysis*  
   
- **Overall Accuracy**: The classifier achieved about 55% accuracy.  
- **Variability Across Categories**:  
  - High precision but low recall for 'talk.politics.mideast' suggests that when the classifier predicts this category, it's often correct, but it misses many documents that belong to this category.  
  - 'sci.space' has a high recall, meaning it correctly identifies most documents in that category, but lower precision indicates it also includes documents from other categories.  
- **F1-Score**: The balance between precision and recall varies, indicating inconsistent performance across categories.  
   
*Observations*  
   
- The compression-based approach performs better on some categories than others, possibly due to differences in vocabulary and the amount of shared information.  
- It may struggle with categories that have less overlap in terminology with other categories.  
   
---  
   
**Cosine Similarity with TF-IDF Embeddings**  
   
*What are TF-IDF and Cosine Similarity?*  
   
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: A statistic that reflects how important a word is to a document in a collection. It increases with the number of times a word appears in a document but is offset by how common the word is across all documents.  
- **Cosine Similarity**: A measure of similarity between two non-zero vectors. In text analysis, it calculates the cosine of the angle between two document vectors, indicating how similar they are.  
   
*Implementing the Cosine Similarity Classifier*  
   
**Preprocessing Steps**  
   
- Same as before: lowercase conversion, removal of non-alphanumeric characters, and stopword removal.  
   
**Classifier Logic**  
   
1. **Vectorize Texts Using TF-IDF**:  
   - Convert the training and test documents into numerical vectors where each dimension represents a word's TF-IDF score.  
2. **Compute Cosine Similarity**:  
   - For each test document, calculate the cosine similarity with all training documents.  
   - Identify the training document with the highest similarity score.  
3. **Assign Category**:  
   - Assign the test document to the same category as its most similar training document.  
   
**Results and Analysis**  
   
*Cosine Similarity Classifier Report*  
   
```  
Cosine Similarity Classifier Report:  
                               precision    recall  f1-score   support  
  
                comp.graphics       0.87      0.90      0.88       195  
             rec.sport.hockey       0.83      0.93      0.88       200  
                    sci.space       0.88      0.78      0.83       197  
        talk.politics.mideast       0.88      0.85      0.87       188  
  
                     accuracy                           0.86       780  
                    macro avg       0.87      0.86      0.86       780  
                 weighted avg       0.87      0.86      0.86       780  
```  
   
*Analysis*  
   
- **Overall Accuracy**: Achieved approximately 86% accuracy, significantly higher than the compression-based method.  
- **Balanced Performance**: High precision and recall across all categories indicate consistent and reliable classification.  
- **Strong F1-Scores**: Reflects a good balance between precision (correctness of positive predictions) and recall (ability to find all positive instances).  
   
*Observations*  
   
- The use of TF-IDF embeddings with cosine similarity effectively captures the importance of words and the context within the documents.  
- This method outperforms the compression-based approach, likely due to its ability to model the semantic relationships between words.  
   
---  
   
**Comparative Analysis**  
   
*Performance Comparison*  
   
- The compression-based classifier achieved around 55% accuracy.  
- The cosine similarity classifier reached around 86% accuracy.  
   
*Strengths of Compression-Based Classification*  
   
- **Unsupervised**: Does not require labeled data for training.  
- **Language Agnostic**: Can be applied to any textual data.  
- **Conceptually Simple**: Based on fundamental principles of information theory.  
   
*Limitations of Compression-Based Classification*  
   
- **Lower Accuracy**: Underperforms compared to more sophisticated models.  
- **Computational Intensity**: Compression operations can be time-consuming, especially with large datasets.  
- **Limited Semantic Understanding**: Does not effectively capture the meanings of words and their relationships.  
   
*Strengths of Cosine Similarity Classification*  
   
- **High Accuracy**: Demonstrates strong performance in classifying documents correctly.  
- **Semantic Awareness**: TF-IDF weights enhance the importance of meaningful words, and cosine similarity measures contextual similarity.  
- **Efficiency**: Optimized mathematical operations allow for faster computations on larger datasets.  
   
*Limitations of Cosine Similarity Classification*  
   
- **Requires Preprocessing**: Needs thorough text cleaning to be effective.  
- **Dependent on Vocabulary**: May not handle new or unseen words well without proper handling.  
   
---  
   
**Conclusion**  
   
This exploration highlights how different approaches to text classification can yield varying results. The compression-based method offers an innovative, unsupervised technique that can be useful in certain contexts. However, it may not be as effective for tasks requiring high precision and recall.  
   
The cosine similarity classifier, utilizing TF-IDF embeddings, provides stronger performance and better captures the nuances of the text. It's more suited for applications where accuracy is critical.  
   
*Key Takeaways*  
   
- **Method Selection**: The choice of classification method should align with the specific needs and constraints of the task at hand.  
- **Understanding Trade-offs**: It's important to consider the trade-offs between simplicity, computational resources, and accuracy.  
- **Value of Exploration**: Experimenting with different techniques can lead to valuable insights and potential improvements in NLP tasks.  
   
---  

```
# demo.py  
  
import os  
import re  
import zlib  
import nltk  
import numpy as np  
from collections import defaultdict  
from sklearn.datasets import fetch_20newsgroups  
from sklearn.metrics import classification_report  
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  
  
# Download NLTK resources if not already downloaded  
nltk.download('stopwords')  
  
# Set of English stopwords  
stop_words = set(nltk.corpus.stopwords.words('english'))  
  
def preprocess_text(text):  
    """  
    Preprocess the input text by:  
    - Lowercasing  
    - Removing non-alphanumeric characters  
    - Removing extra whitespaces  
    - Removing stopwords  
    """  
    # Lowercase the text  
    text = text.lower()  
    # Remove non-alphanumeric characters  
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  
    # Remove extra whitespaces  
    text = re.sub(r'\s+', ' ', text).strip()  
    # Remove stopwords  
    words = text.split()  
    words_filtered = [word for word in words if word not in stop_words]  
    return ' '.join(words_filtered)  
  
def compressed_size(s):  
    """  
    Calculate the compressed size of a string using zlib compression.  
    """  
    return len(zlib.compress(s.encode('utf-8')))  
  
def normalized_compression_distance(s1, s2):  
    """  
    Calculate the Normalized Compression Distance (NCD) between two strings.  
    """  
    c_s1 = compressed_size(s1)  
    c_s2 = compressed_size(s2)  
    c_s1s2 = compressed_size(s1 + s2)  
    ncd = (c_s1s2 - min(c_s1, c_s2)) / max(c_s1, c_s2)  
    return ncd  
  
def classify_document_ncd(doc, category_texts):  
    """  
    Classify a document based on NCD similarity to category texts.  
    """  
    min_ncd = float('inf')  
    best_category = None  
    for category, cat_text in category_texts.items():  
        ncd = normalized_compression_distance(doc, cat_text)  
        if ncd < min_ncd:  
            min_ncd = ncd  
            best_category = category  
    return best_category  
  
def main():  
    # Fetch the dataset  
    categories = ['comp.graphics', 'rec.sport.hockey', 'sci.space', 'talk.politics.mideast']  
  
    print("Fetching the 20 Newsgroups dataset...")  
    data = fetch_20newsgroups(  
        subset='all',  
        categories=categories,  
        remove=('headers', 'footers', 'quotes')  
    )  
  
    texts = data.data  
    labels = data.target  
    label_names = data.target_names  
  
    # Split the data into training and test sets  
    print("Splitting data into training and test sets...")  
    train_texts, test_texts, train_labels, test_labels = train_test_split(  
        texts,  
        labels,  
        test_size=0.2,  
        random_state=42,  
        stratify=labels  
    )  
  
    # Preprocess all training texts and build category texts  
    print("Preprocessing training texts...")  
    category_texts = defaultdict(str)  
    for text, label in zip(train_texts, train_labels):  
        label_name = label_names[label]  
        processed_text = preprocess_text(text)  
        category_texts[label_name] += ' ' + processed_text  
  
    # Compression-Based Classification  
    print("\nStarting Compression-Based Classification...")  
    predictions_ncd = []  
    for text in test_texts:  
        processed_text = preprocess_text(text)  
        predicted_label = classify_document_ncd(processed_text, category_texts)  
        predictions_ncd.append(predicted_label)  
    true_labels_text = [label_names[label] for label in test_labels]  
  
    print("\nCompression-Based Classifier Report:")  
    print(classification_report(true_labels_text, predictions_ncd, target_names=label_names))  
  
    # Cosine Similarity with TF-IDF Embeddings  
    print("\nStarting Cosine Similarity Classification with TF-IDF embeddings...")  
    vectorizer = TfidfVectorizer()  
  
    # Preprocess texts for TF-IDF  
    train_texts_processed = [preprocess_text(text) for text in train_texts]  
    test_texts_processed = [preprocess_text(text) for text in test_texts]  
  
    # Fit the vectorizer on the training data  
    tfidf_train = vectorizer.fit_transform(train_texts_processed)  
    tfidf_test = vectorizer.transform(test_texts_processed)  
  
    # Compute cosine similarity between test documents and all training documents  
    similarity_matrix = cosine_similarity(tfidf_test, tfidf_train)  
  
    # Predict labels based on the most similar training document  
    predictions_cosine = []  
    for idx in range(similarity_matrix.shape[0]):  
        most_similar_idx = similarity_matrix[idx].argmax()  
        predicted_label = label_names[train_labels[most_similar_idx]]  
        predictions_cosine.append(predicted_label)  
  
    print("\nCosine Similarity Classifier Report:")  
    print(classification_report(true_labels_text, predictions_cosine, target_names=label_names))  
  
if __name__ == '__main__':  
    main()  
```
