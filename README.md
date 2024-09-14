# Sentiment Analysis on Review Data

## Overview

This project performs sentiment analysis on review data using text classification techniques. The primary goal is to classify reviews into 'good' or 'bad' categories based on their ratings. The project involves data preprocessing, model training, evaluation, and visualization. The final model is tested on a cleaned dataset to ensure its effectiveness.

## Dataset

- **File:** `usercourses.csv`
- **Description:** Contains reviews with ratings and comments.
- **Columns:**
  - `review_comment`: The text of the review.
  - `review_rating`: The rating given in the review.

## Steps and Methodology

### 1. Data Cleaning and Preprocessing

- **Handle Missing Values:** Removed rows with missing values in `review_comment` or `review_rating`.
- **Convert Ratings:** Converted `review_rating` to numeric, handling any conversion issues.
- **Text Cleaning:** Applied a function to clean and normalize text data by removing punctuation and converting to lowercase.

### 2. Binary Classification

- **Convert Ratings to Binary:** Transformed ratings into binary labels ('good' for ratings >= 4 and 'bad' otherwise).
- **Class Imbalance Handling:** Oversampled the minority class to balance the dataset.
- **Feature Extraction:** Used TF-IDF vectorization to convert text into numerical features.
- **Model Training:** Trained a Naive Bayes classifier using the upsampled dataset.

### 3. Evaluation

- **Confusion Matrix:** Visualized the performance using a confusion matrix plotted as a heatmap.
- **Classification Report:** Generated a detailed classification report showing precision, recall, and F1-score for each class.
- **Test Set Evaluation:** Applied the trained model to a cleaned test set to assess its performance on unseen data.

## Code Explanation

### Data Cleaning

```python
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load and clean data
test_set = pd.read_csv('usercourses.csv', error_bad_lines=False, warn_bad_lines=True)
test_set_clean = test_set.dropna(subset=['review_comment', 'review_rating'])
test_set_clean['review_rating'] = pd.to_numeric(test_set_clean['review_rating'], errors='coerce')
test_set_clean = test_set_clean.dropna(subset=['review_rating'])
test_set_clean['review_comment'] = test_set_clean['review_comment'].apply(clean_text)
test_set_clean['review_rating'] = test_set_clean['review_rating'].apply(binary_rating)
```

### Model Training

```python
# Initialize a TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
model = MultinomialNB()
text_clf_binary = make_pipeline(vectorizer, model)

# Split data and train
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)
text_clf_binary.fit(X_train_binary, y_train_binary)
```

### Evaluation and Visualization

```python
# Predict and evaluate
y_pred_binary = text_clf_binary.predict(X_test_binary)
report_binary_df = pd.DataFrame(report_binary).transpose()

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_binary, annot=True, fmt='d', cmap='Blues', xticklabels=['bad', 'good'], yticklabels=['bad', 'good'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
```

## Results

The model demonstrated improved performance with balanced class distribution and accurate sentiment classification. The evaluation metrics and visualizations provide insights into the model's effectiveness and areas for potential improvement.

## Requirements

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `re`

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Rishav123raj/MachineLearnnigCA2.git
   ```
2. Navigate to the project directory:
   ```bash
   cd MachineLearnnigCA2
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebook or Python scripts to perform sentiment analysis.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to [ChatGPT](https://openai.com) for assistance with code snippets and explanations.
