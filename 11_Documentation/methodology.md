# Methodology

1. Validate the supplied crop dataset.
2. Remove duplicates and rows with missing required values.
3. Split data into 80% training and 20% stratified testing using random_state 42.
4. Train Random Forest, XGBoost and SVM.
5. Evaluate accuracy, weighted precision, weighted recall and weighted F1.
6. Run 5-fold stratified cross-validation on the selected model family.
7. Save each model and copy the selected model to `06_Final_Model/best_model.joblib`.
8. Expose prediction through Flask.
9. Display the crop recommendation and held-out test accuracy on the website.
