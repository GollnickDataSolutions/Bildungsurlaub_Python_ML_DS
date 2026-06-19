# Model Comparison Script

This script demonstrates how to compare multiple machine learning models on a classification task. Here's what the script does:

## Key Components

1. **Data Loading and Preparation**:
   - Loads the diabetes dataset
   - Splits data into train/test sets (stratified)
   - Scales features using StandardScaler

2. **Model Definition**:
   - Defines 6 different classifiers:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Gradient Boosting
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)

3. **Hyperparameter Tuning**:
   - Uses GridSearchCV with StratifiedKFold cross-validation
   - Optimizes for ROC-AUC score
   - Each model has its own parameter grid

4. **Evaluation**:
   - Tests all models on the test set
   - Calculates multiple metrics: accuracy, precision, recall, F1-score, ROC-AUC
   - Compares performance across all models

5. **Results Summary**:
   - Displays a comprehensive table of results
   - Identifies the best performing model based on ROC-AUC

## Key Concepts Demonstrated

- **Cross-validation**: Using StratifiedKFold to ensure balanced folds
- **Hyperparameter optimization**: GridSearchCV for finding optimal parameters
- **Model comparison**: Evaluating multiple models using consistent metrics
- **Performance metrics**: Understanding different ways to assess model performance

This approach is widely used in machine learning projects to select the best model for a given task.