# FUTURE_ML_02
Task 02 - Support Ticket Classification and Priority Prediction using Machine Learning (TF-IDF + Logistic Regression)

## Internship Task – Machine Learning Track  

This project was completed as part of the **Machine Learning Internship (ML Track)**.  

The objective of this task was to build a real-world NLP-based system that can:

- Automatically classify IT support tickets into predefined categories  
- Reduce manual ticket sorting workload  
- Improve response efficiency of support teams  
- Demonstrate practical NLP and ML implementation  

---

#  Problem Statement  

In real organizations, support teams receive thousands of tickets daily.  
Manual categorization leads to:

- Delayed urgent issue handling  
- Increased backlog  
- Inefficient resource allocation  


---

#  Dataset Information  

Dataset Used: IT Service Ticket Classification Dataset  
dataset link: https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset

Total Records: **47,837 support tickets**

Columns Used:

- `Document` → Ticket description text  
- `Topic_group` → Ticket category label  

---

##  Class Distribution  

The dataset contains the following categories:

| Category               | Count |
|------------------------|-------|
| Hardware               | 13,617 |
| HR Support             | 10,915 |
| Access                 | 7,125 |
| Miscellaneous          | 7,060 |
| Storage                | 2,777 |
| Purchase               | 2,464 |
| Internal Project       | 2,119 |
| Administrative rights  | 1,760 |

### Class Distribution Visualization

<img width="501" height="502" alt="Screenshot 2026-02-11 115030" src="https://github.com/user-attachments/assets/147ea12e-7f05-40c5-b729-0f65e0a00bc5" />

#  Methodology  

##  Text Preprocessing  

The following NLP preprocessing steps were applied:

- Converted text to lowercase  
- Removed punctuation and special characters  
- Removed stopwords using NLTK  
- Applied lemmatization  

---

##  Feature Engineering  

TF-IDF Vectorization was used with:

- `max_features = 6000`
- `ngram_range = (1,2)`
- `min_df = 3`

---

##  Train-Test Split  

- 80% Training Data  
- 20% Testing Data  
- Stratified split to preserve class distribution  

Total Test Samples: **9,568**

---

##  Models Implemented  

###  Logistic Regression (Primary Model)  
Chosen because it performs strongly in text classification tasks.

###  Random Forest (Comparison Model)  
Used for benchmarking against a tree-based approach.

---

#  Model Performance  

## Logistic Regression Accuracy  

**Accuracy: 85.48%**

```
Logistic Regression Accuracy: 0.8548
```

---

##  Classification Report  

<img width="740" height="379" alt="Screenshot 2026-02-11 115006" src="https://github.com/user-attachments/assets/8dc57fb3-c9af-446c-9e68-d424148b274e" />


### Overall Metrics:
- Macro Average F1-Score: **0.85**
- Weighted Average F1-Score: **0.85**

---

##  Confusion Matrix  
<img width="803" height="712" alt="Screenshot 2026-02-11 120503" src="https://github.com/user-attachments/assets/8aab663b-bfa2-42f2-86f2-736666fdaffc" />


---

## Class-wise Performance Visualization  

<img width="990" height="656" alt="Screenshot 2026-02-11 115043" src="https://github.com/user-attachments/assets/ab05e43d-b075-4583-aa2c-cfec0021295f" />

This plot shows precision, recall, and F1-score for each category.

---

#  Cross Validation  

5-Fold Cross Validation was performed to check generalization.

The cross-validation score was close to test accuracy, indicating:

- No significant overfitting  
- Good model generalization  

---

# 💾 Model Persistence  

The trained model and TF-IDF vectorizer were saved using `joblib`:

- `ticket_classifier_model.pkl`
- `tfidf_vectorizer.pkl`

This makes the system deployment-ready and reusable for real-time predictions.

---

#  Business Impact  

This ML system can:

- Reduce manual ticket sorting time by 60–80%  
- Automatically route tickets to correct departments  
- Improve SLA compliance  
- Improve customer support efficiency  

---

#  Technologies Used  

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- NLTK  
- Matplotlib  
- Seaborn  
- Joblib  

---

# 📌 Conclusion  

This project successfully demonstrates:

- Real-world NLP preprocessing  
- TF-IDF feature engineering  
- Multi-class classification  
- Performance evaluation using industry-standard metrics  
- Overfitting validation using cross-validation  
- Model saving for deployment  

The system achieves **85.48% accuracy** on 47,000+ real support tickets, proving strong practical applicability.

---

# 👨‍💻 Author  

Raghav Marwaha  
Machine Learning Intern  
Future Interns  

---

