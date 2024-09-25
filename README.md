# Bharti Airtel Stock Price Prediction

This project performs stock price prediction for Bharti Airtel using four different machine learning models: Long Short-Term Memory (LSTM), Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Extreme Gradient Boosting (XGBoost). The workflow includes Exploratory Data Analysis (EDA), data preprocessing, training, evaluation, and prediction for each model.

## Project Structure

```
├── data/                    # Folder containing the stock dataset
│   └── BHARTIARTL.csv       # Dataset of Bharti Airtel stock prices
├── models/                  # Folder containing saved models (optional)
├── notebooks/               # Jupyter notebooks for model experimentation
├── src/                     # Source code for different models
│   ├── lstm_model.py        # LSTM model training and prediction
│   ├── svm_model.py         # SVM model training and prediction
│   ├── knn_model.py         # KNN model training and prediction
│   └── xgb_model.py         # XGBoost model training and prediction
├── README.md                # Project documentation
└── requirements.txt         # List of Python dependencies
```


## Requirements

- Python 3.x
- TensorFlow 2.x
- XGBoost
- scikit-learn
- pandas
- matplotlib
- seaborn
- numpy

To install all dependencies, use the following command:

```
pip install -r requirements.txt
```

## Dataset

The dataset used for this project is Bharti Airtel's historical stock prices. It includes the following columns:

- **Date**: The date of the stock price.

- **Close**: The closing price of Bharti Airtel stock.

The dataset is preprocessed and normalized for model training.

## Models

## 1. Long Short-Term Memory (LSTM)

The LSTM model is designed to handle time series data by learning temporal dependencies. It uses a sequence length of 60 days to predict the stock price for the next day.

**Steps**:

- Data normalization

- Sequence creation

- LSTM model building and training

- Evaluation using Mean Squared Error (MSE)

## 2. Support Vector Machine (SVM)

SVM is a regression model that finds the optimal hyperplane in a high-dimensional space. For stock price prediction, the RBF kernel is used.

**Steps**:

- Data flattening

- SVM model training

- Evaluation using Mean Squared Error (MSE)

## 3. K-Nearest Neighbors (KNN)

The KNN algorithm is used to predict stock prices by comparing the input with the k-nearest data points in the training set.

**Steps**:

- Data flattening

- KNN model training

- Evaluation using Mean Squared Error (MSE)

## 4. Extreme Gradient Boosting (XGBoost)

XGBoost is a powerful decision-tree-based model used for regression tasks.

**Steps**:

- Data flattening

- XGBoost model training

- Evaluation using Mean Squared Error (MSE)

## Results

The Mean Squared Error (MSE) for each model is printed during the evaluation step. Predictions are plotted against actual stock prices for comparison.

## How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/bharti-airtel-stock-prediction.git
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Run the scripts for each model:

```
python src/lstm_model.py
python src/svm_model.py
python src/knn_model.py
python src/xgb_model.py
```

4. View the results and stock price prediction charts.

