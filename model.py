import pandas as pd
from sklearn.metrics import precision_score

from sklearn.ensemble import RandomForestClassifier
#df = pd.read_csv('/content/gdrive/MyDrive/featured_data.csv',index_col=['time'])
df = pd.read_csv('featured_data.csv',index_col=['time'])

#2 upto 10 candles
days = 10

df['rolling_min_close']  = df['close'].rolling(window=days).min().shift(-days)
df['rolling_max_close']  = df['close'].rolling(window=days).max().shift(-days)

# Calculate the percent difference between 'close' and its rolling minimum and maximum
df['percent_diff_from_min'] = abs(((df['close'] - df['rolling_min_close']) / df['rolling_min_close']) * 100)
df['percent_diff_from_max'] = abs(((df['close'] - df['rolling_max_close']) / df['rolling_max_close']) * 100)

# Check if the close to max difference percent is twice as big as the close to min difference percent
df['target'] = (df['percent_diff_from_max'] > 2 * df['percent_diff_from_min']).astype(int)

#keep in mind that in forex, relationship of target and high/low/open prices are non linear/indirect,
#meaning if open is high it doesnt mean that the next candles is also gonna be going higher
#so for non linear randomforestclassifier picks them well.
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)




def backtest(data,model, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):

        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train,test, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)



df =df.dropna()



def predict(train, test, model):
    X_train = train.drop(columns=['rolling_min_close','rolling_max_close','percent_diff_from_min','percent_diff_from_max','target'])
    X_target = train.target
    model.fit(X_train,X_target)


    y_test = test.drop(columns=['rolling_min_close','rolling_max_close','percent_diff_from_min','percent_diff_from_max','target'])
    y_target = test.target
    preds = model.predict_proba(y_test)[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["target"], preds], axis=1)
    return combined

predictions = backtest(df, model)


precision_score(predictions["target"], predictions["Predictions"])
predictions["target"].value_counts() / predictions.shape[0]
predictions['target'].value_counts()
predictions["Predictions"].value_counts()