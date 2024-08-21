# forex-and-stock-price-prediction-using-radom-forest-classifier
#Forex Price Prediction with Random Forest Classifier
Project Overview
This project uses a RandomForestClassifier to predict future movements in forex prices. The model is trained on historical data, using rolling minimum and maximum values of closing prices to derive features that inform the predictions. The primary goal is to determine whether the price will significantly move upward or downward within the next 10 days.

Dependencies
To run this project, you will need the following Python libraries:
pandas
numpy
scipy
pandas_ta
scikit-learn

You can install these dependencies using pip:
pip install pandas numpy scipy pandas_ta scikit-learn

#Data
The dataset (featured_data.csv) should contain forex price data with a time column as the index and a close column representing the closing prices of the forex pair.

Features
The following features are calculated:

Rolling Min Close (rolling_min_close): The minimum closing price over the last 10 days.
Rolling Max Close (rolling_max_close): The maximum closing price over the last 10 days.
Percent Difference from Min (percent_diff_from_min): The percentage difference between the current closing price and the rolling minimum.
Percent Difference from Max (percent_diff_from_max): The percentage difference between the current closing price and the rolling maximum.
The target variable is a binary indicator (target) that identifies whether the percent difference from the maximum is twice as large as the percent difference from the minimum. This serves as the basis for predicting significant upward price movements.

Model
A RandomForestClassifier is used for prediction due to its ability to capture non-linear relationships in the data. The model is configured with:

n_estimators=200: The number of trees in the forest.
min_samples_split=50: The minimum number of samples required to split an internal node.
random_state=1: Ensures reproducibility of results.
Backtesting
The backtest function is used to simulate the trading strategy on historical data. It incrementally trains the model on an expanding window of data and then makes predictions on the subsequent data points.

Usage
Load Data:

df = pd.read_csv('featured_data.csv', index_col=['time'])
Calculate Features:
The code automatically calculates rolling features and target labels.

Backtest:

predictions = backtest(df, model)
Evaluate Performance:
The precision of the model's predictions is measured, and the distribution of predictions is analyzed:

precision_score(predictions["target"], predictions["Predictions"])
predictions["target"].value_counts() / predictions.shape[0]
predictions['target'].value_counts()
predictions["Predictions"].value_counts()

Conclusion
This project provides a framework for predicting forex price movements using machine learning. By calculating rolling statistics and backtesting predictions, it offers insights into the effectiveness of the model in a trading scenario.
