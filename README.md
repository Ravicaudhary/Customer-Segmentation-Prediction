# Customer Segmentation Prediction using KNN

Practical supervised machine learning project using K-Nearest Neighbors (KNN) to predict customer segments.

## Features
- Age
- Annual Income
- Spending Score
- Visits Per Month

## Target
- High Value
- Regular
- Low Value

## Workflow
1. Load customer data
2. Train/test split with stratification
3. Standardize features with StandardScaler
4. Test K values from 1 to 11 using weighted Recall
5. Select the best K
6. Train final KNN model
7. Evaluate Accuracy, Precision, Recall and F1
8. Generate classification report and confusion matrix
9. Predict a new customer's segment

## Run

```bash
pip install -r requirements.txt
python customer_segmentation_knn.py
```

KNN is used here for supervised segment prediction because segment labels are provided. For discovering unknown customer groups without labels, K-Means clustering is typically used.
