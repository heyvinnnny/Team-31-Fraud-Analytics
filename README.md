# Team-31-Fraud-Analytics
This repository contains the work of Team 31's Fraud Analytic Project, focusing on the detection and analysis of fraudulent job postings. The project involves various stages of data handling, from scraping and cleaning data to applying machine learning models to identify potential fraud.

## Repository Structure

### Data Preprocessing
This folder contains the initial data processing notebooks:
- `Data Preprocessing.ipynb`: Jupyter notebook for initial data cleaning and preparation.
- `Exploratory Data Analysis.ipynb`: Notebook for statistical analysis and visualization of data features.

### Models
This section of the repository contains the machine learning models that were developed and evaluated for the fraud detection task:

- **Modeling Notebooks**:
  - `0 Modeling.ipynb`: Modeling notebook
  - `1 Modeling - AdaBoost.ipynb`: Implementation and training of the AdaBoost classifier.
  - `1 Modeling - CatBoost.ipynb`: Notebook for the CatBoost algorithm, a gradient boosting classifier.
  - `1 Modeling - DecisionTree.ipynb`: Decision Tree classifier application and analysis.
  - `1 Modeling - KNN.ipynb`: K-Nearest Neighbors algorithm for fraud detection.
  - `1 Modeling - LogisticRegression.ipynb`: Logistic Regression model for binary classification tasks.
  - `1 Modeling - MLP.ipynb`: Multi-layer Perceptron classifier, a type of neural network.
  - `1 Modeling - NaiveBayes.ipynb`: Naive Bayes classifier for probabilistic prediction.
  - `1 Modeling - RandomForest.ipynb`: RandomForest classifier for handling complex data structures.
  - `1 Modeling - SVM.ipynb`: Support Vector Machine model for margin-based classification.
  - `1 Modeling - XGBoost.ipynb`: eXtreme Gradient Boosting model implementation.
  - `1 Modeling Evaluation - BiLSTM_with_numerical.ipynb`: Evaluation of a BiLSTM model that includes numerical data integration.
  - `1 Modeling Evaluation - BiLSTM.ipynb`: Evaluation of a Bi-directional Long Short-Term Memory network.
  - `1 Modeling Evaluation - NN.ipynb`: Evaluation of a simple Neural Network classifier.

- **Model Evaluation Notebooks**:
  - `2 Modeling - Evaluation.ipynb`: Combined evaluation notebook for comparing the performance of various models.

- **Serialized Models**:
  - `adaboost.joblib`: Serialized AdaBoost model for quick loading and prediction.
  - `catboost.joblib`: Serialized CatBoost model.
  - `decision_tree.joblib`: Serialized Decision Tree model.
  - `LogisticRegression.joblib`: Serialized Logistic Regression model.
  - `MLP.joblib`: Serialized Multi-layer Perceptron model.
  - `NaiveBayes.joblib`: Serialized Naive Bayes model.
  - `random_forest.joblib`: Serialized Random Forest model.
  - `SVM.joblib`: Serialized Support Vector Machine model.
  - `xgboost.joblib`: Serialized XGBoost model.

### Processed Data Files
Cleaned and processed datasets after pre-processing:
- `out.csv`: Output file for the text-embedding feature
- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`: Split datasets for training and testing models.

### Raw Data Files
Original unprocessed datasets:
- `fake_job_postings.csv`: Dataset containing synthetic fraudulent job postings.

Processed datasets:
- `final_job_posting.csv`: Consolidated dataset post-processing for final analysis.

### Scraping
Scripts and notebooks used to scrape data from job listing sites:
- `CareersFuture`:
  - `scrapedata`:
    - `all jobs.csv`: This folder contains each individual scraped data search term which is referenced from the cf-search.csv.
  - `cf-search.csv`: CSV file that contains all the search terms for the scrapping process from the CareersFuture website.
  - `cleaned_data_nonfraud.csv`: CSV file all the scraped data (after first stage of initial cleaning)
  - `merged_data.csv`: CSV file to merge all the job postings from the scrapedata folder.
  - `Cleaning of Scraped Data.ipynb`: Notebook detailing the cleaning process of scraped data.
  - `Job Scraping from CareersFuture SG.ipynb`: Notebook used for scraping job postings from CareersFuture SG.
  - `Fake Data`:
    - `cf_fakejobs.csv`: Generated synthetic dataset of fake job postings using ChatGPT.
    - `Fake Data Generation.ipynb`: Notebook used for generating synthetic job listing data, code generated using ChatGPT.

