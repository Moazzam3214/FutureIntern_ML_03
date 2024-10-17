# Credit Card Fraud Detection

This repository contains a machine learning project aimed at detecting fraudulent credit card transactions. Given the highly imbalanced nature of the dataset, the focus is on handling this imbalance while training models to effectively classify fraudulent and non-fraudulent transactions.

## Project Overview

The objective of this project is to build a classification model that can identify fraudulent transactions from a large dataset of credit card transactions. Due to the nature of the problem, the dataset is highly imbalanced, with only a small fraction of transactions being fraudulent. This project explores various techniques to address this imbalance and improve model performance.

### Dataset

The dataset used in this project includes credit card transaction data with the following key features:
- **Time**: Time of the transaction
- **V1-V28**: Principal components obtained through PCA for privacy reasons
- **Amount**: Transaction amount
- **Class**: Target variable (1 for fraud, 0 for non-fraud)


## Technologies Used

- Python
- Jupyter Notebook
- Pandas and NumPy for data manipulation
- Matplotlib and Seaborn for data visualization
- Scikit-Learn for machine learning models

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/moazzam3214/credit-card-fraud-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd credit-card-fraud-detection
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Credit_Card_Fraud_Detection.ipynb
   ```
2. Run the notebook cells to load the data, train models, and evaluate results.

## Models and Techniques

The project includes the following steps:
1. **Data Preprocessing**: 
   - Handling missing values (if any)
   - Scaling numerical features using StandardScaler or MinMaxScaler
2. **Imbalanced Data Handling**: 
   - Techniques such as **SMOTE** (Synthetic Minority Over-sampling Technique) and **Random Undersampling** are used to balance the dataset.
3. **Model Training**:
   - **Logistic Regression**
   - **Random Forest Classifier**
   - **XGBoost**
4. **Evaluation Metrics**:
   - **Confusion Matrix**
   - **Precision, Recall, F1-Score**
   - **ROC-AUC Score**

## Results

The models were evaluated based on their performance on the imbalanced dataset, with a particular focus on precision, recall, and F1-score due to the higher cost of misclassifying fraudulent transactions.


## Future Improvements

- Explore deep learning approaches like **Neural Networks** for better results.
- Perform hyperparameter tuning using **GridSearchCV** or **RandomizedSearchCV** for model optimization.
- Investigate further techniques for handling data imbalance, such as **ADASYN** or **NearMiss**.

## Contributing

Feel free to fork this repository, create a new branch, and submit a pull request with your changes. Contributions are always welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
