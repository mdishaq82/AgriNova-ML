# Methodology

1. Load agricultural prototype data.
2. Validate duplicates, missing values, and target labels.
3. Explore distributions and correlations.
4. Split data into stratified training and test sets.
5. Train Random Forest, SVM, and XGBoost.
6. Compare weighted precision, recall, and F1.
7. Save the best model.
8. Expose prediction through Flask.
9. Connect a simple web form to the API.

**Data limitation:** the bundled dataset is synthetic and therefore results are demonstrations of the pipeline, not evidence of field performance.
