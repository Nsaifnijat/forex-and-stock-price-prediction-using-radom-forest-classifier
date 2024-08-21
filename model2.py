import pandas as pd
from sklearn.metrics import precision_score

from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('featured_data.csv',index_col=['time'])
#2 upto 50 candles
df['tomorrow'] = df['close'].shift(-5)
#check if price is going up or down in 2rows or next 2 candles,astype changes true or false to 1 or 0
df['target'] = (df['tomorrow'] > df['close']).astype(int)



#keep in mind that in forex, relationship of target and high/low/open prices are non linear/indirect, 
#meaning if open is high it doesnt mean that the next candles is also gonna be going higher
#so for non linear randomforestclassifier picks them well.
model = RandomForestClassifier(n_estimators=300,min_samples_split=100,random_state=1)


def predict(train,test, model):
    X_train = train.drop(columns=['tomorrow','target'])
    X_target = train.target
    model.fit(X_train,X_target)
    
    y_test = test.drop(columns=['tomorrow','target'])
    y_target = test.target
    prediction = model.predict(y_test)
    prediction = pd.Series(prediction,index=test.index, name="Predictions")
    combined = pd.concat([y_target, prediction],axis=1)
    
    return combined

def backtest(data,model, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
    
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train,test, model)
        all_predictions.append(predictions)
        
    return pd.concat(all_predictions)
    

predictions = backtest(df, model)
print(predictions)
#0==down, 1==up
up_down_counts = predictions['Predictions'].value_counts()
print(up_down_counts)

precision = precision_score(predictions['target'], predictions['Prediction'])
percentage = predictions['target'].value_counts()/predictions.shape[0]
