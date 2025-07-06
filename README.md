# hw3_srinivas.ipynb: Credit Scoring Prediction

This notebook demonstrates a machine learning workflow for credit risk assessment, predicting the likelihood of a customer defaulting on their loan within two years.

## Overview

The `hw3_srinivas.ipynb` notebook implements a credit scoring model using the Gradient Boosting Classifier. It covers the end-to-end process from data loading and handling missing values through median imputation, to splitting data into training and testing sets, training a classification model, evaluating its performance using various metrics, and finally, demonstrating prediction on new, unseen data.

## Key Features

*   **Data Loading & Preprocessing**: Loads a credit scoring dataset and handles missing numerical values efficiently using median imputation.
*   **Data Splitting**: Divides the dataset into training and testing sets to prepare for model development and evaluation.
*   **Gradient Boosting Model**: Utilizes the powerful Gradient Boosting Classifier for building the predictive model.
*   **Comprehensive Model Evaluation**: Assesses model performance using standard metrics such as Accuracy Score, Classification Report (precision, recall, f1-score), and a visual Confusion Matrix heatmap.
*   **New Data Prediction**: Demonstrates how to apply the trained model to predict the default risk for new, hypothetical customer profiles.

## Technologies and Libraries Used

The notebook utilizes the following Python libraries:

*   **pandas**: For robust data manipulation and analysis.
*   **numpy**: For fundamental numerical operations and array processing.
*   **matplotlib.pyplot**: For creating static, interactive, and animated visualizations.
*   **seaborn**: A high-level data visualization library, built on matplotlib, for creating informative and attractive statistical graphics.
*   **sklearn (scikit-learn)**: A comprehensive machine learning library, specifically leveraging:
    *   `GradientBoostingClassifier`: The core ensemble model for classification.
    *   `train_test_split`: For dividing datasets into training and testing subsets.
    *   `accuracy_score`, `confusion_matrix`, `classification_report`: For evaluating classifier performance.
    *   `SimpleImputer`: For handling missing data by filling `NaN` values.
*   **warnings**: To manage warning messages during execution.

## Main Sections and Steps

The notebook follows a logical and structured approach to building and evaluating the credit scoring model:

1.  **Setup and Library Imports**:
    *   All necessary Python libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, and various `sklearn` modules) are imported.
    *   A `SimpleImputer` is initialized with a `median` strategy, ready to fill any missing numerical values.
2.  **Data Loading and Imputation**:
    *   The `credit_scoring_sample (1).csv` dataset is loaded into a pandas DataFrame.
    *   Missing values in the DataFrame are imputed using the pre-configured `SimpleImputer`, creating `imputed_df`.
3.  **Data Preparation for Modeling**:
    *   The dataset is split into features (`X`) and the target variable (`y`), which is `'SeriousDlqin2yrs'`.
    *   The data is then divided into training and testing sets using `train_test_split`.
        *   **Note**: The split is configured such that 20% of the data is used for training (`X_train`, `y_train`) and 80% for testing (`X_test`, `y_test`), which is an unconventional split proportion where the test set is significantly larger than the training set.
4.  **Model Training**:
    *   A `GradientBoostingClassifier` is initialized with default parameters (100 estimators, learning rate of 0.1, max depth of 3, and a fixed `random_state`).
    *   The model is trained (`fit`) using the 20% training data (`X_train`, `y_train`).
5.  **Model Evaluation**:
    *   Predictions (`y_pred`) are generated on the 80% test set (`X_test`).
    *   The `accuracy_score` is calculated and displayed.
    *   A comprehensive `classification_report` (detailing precision, recall, and f1-score for each class) is printed.
    *   The `confusion_matrix` is computed and displayed, providing a breakdown of correct and incorrect predictions.
    *   A heatmap visualization of the `confusion_matrix` is generated using `seaborn` for clear interpretation.
6.  **Prediction on New Data**:
    *   A sample DataFrame `new_data` is created, representing a hypothetical customer with specific attributes.
    *   The trained `gbmodel` is used to predict the default status for this new data point.
    *   The prediction result (0 for no default, 1 for default) is printed.

## Key Insights and Results

*   The notebook successfully trains a Gradient Boosting Classifier for credit risk prediction.
*   The model's performance on the test set is evaluated through standard metrics (accuracy, precision, recall, F1-score) and visually represented by a confusion matrix heatmap.
*   A practical demonstration of the model's utility is provided by predicting the default risk for a new customer profile. For the given sample data, the model predicts `0`, indicating that the customer is "not going to default in next 2 years."

## How to Use/Run the Notebook

To execute this notebook and replicate the analysis:

1.  **Open in Google Colab**: Navigate to the `hw3_srinivas.ipynb` file in your repository or local machine and select "Open with Google Colaboratory".
2.  **Upload Data**: Ensure the dataset named `credit_scoring_sample (1).csv` is available in your Google Colab environment. You can upload it directly to the `/content/` directory (where Cell 3 expects it) using the "Files" tab (folder icon) in the left sidebar of the Colab interface.
3.  **Run All Cells**: Once the notebook is open and the data is uploaded, go to `Runtime` -> `Run all` in the Google Colab menu.
4.  **Review Output**: Observe the execution flow and examine the output of each cell, including the data head, model performance metrics, and the final prediction for the new data.