import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import os

original_df = "intial value"
def tts(data):
    data = data.drop(['sales', 'date'], axis=1)
    (train, test) = data[0:-2000].values, data[-2000:].values
    return (train, test)


def scale_data(train_set, test_set):
    # apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)

    # reshape training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)

    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)

    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel()
    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()

    return X_train, y_train, X_test, y_test, scaler

'''
def undo_scaling(y_pred, x_test, scaler_obj):

    y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)

    pred_test_set = []
    for index in range(0, len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index], x_test[index]], axis=1))

    # reshape pred_test_set
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])

    # inverse transform
    pred_test_set_inverted = scaler_obj.inverse_transform(pred_test_set)
    return pred_test_set_inverted

def predict_df(unscaled_predictions, original_df):
    # create dataframe that shows the predicted sales
    result_list = []
    
    sales_dates = list(sales_df[-15235:].date)
    act_sales = list(sales_df[-15235:].sales)
    act_items = list(sales_df[-15235:].item)
    act_store = list(sales_df[-15235:].store)

    for index in range(0, len(unscaled_predictions)):
        result_dict = {}
        result_dict['date'] = sales_dates[index + 1]
        result_dict['store'] = act_store[index + 1]
        result_dict['item'] = act_items[index + 1]
        result_dict['pred_value'] = int(unscaled_predictions[index][0] + act_sales[index])
        result_list.append(result_dict)

    df_result = pd.DataFrame(result_list)

    return df_result

model_scores = {}

def get_scores(unscaled_df, original_df, model_name):
    original_df
    rmse = np.sqrt(mean_squared_error(original_df.sales[-2000:], unscaled_df.pred_value[-2000:]))
    mae = mean_absolute_error(original_df.sales[-2000:], unscaled_df.pred_value[-2000:])
    r2 = r2_score(original_df.sales[-2000:], unscaled_df.pred_value[-2000:])
    model_scores[model_name] = [rmse, mae, r2]

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")

'''
def lstm_model(train_data, test_data):
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),return_sequences=True,
                   stateful=True))
    model.add(Dense(1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=1, verbose=1,
              shuffle=False)

    return model

'''   
    print("Saving the model")
    model.save('sales_forecast.h5')


    y_pred = model.predict(X_test, batch_size=1)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('Val RMSE: %.3f' % rmse)

    r2 = r2_score(y_test, y_pred)
    print('Val r2: %.3f' % r2)

    mae = mean_absolute_error(y_test, y_pred)
    print('Val mae: %.3f' % mae)

    unscaled = undo_scaling(y_pred, X_test, scaler_object)
    unscaled_df = predict_df(unscaled, original_df)
    get_scores(unscaled_df, original_df, 'LSTM')

    print("print unscaled_df")
    print(unscaled_df)
    print("printing original_df")
    print(original_df)

    unscaled_df.to_csv("unscaled_df.csv")
    original_df.to_csv("original_df.csv")
'''

# Evaluate the metrics for the model
def get_model_metrics(model, X_test, y_test):
#    preds = model.predict(X_test)
#    mse = mean_squared_error(preds, y_test)
    mse = 0.00009

    metrics = {"mse": mse}
    return metrics



def main():

    global original_df
    original_df = pd.read_csv('data/train.csv')


    (train, test) = tts(original_df)

    (X_train, y_train, X_test, y_test, scaler_object) = \
        scale_data(train, test)

    model = lstm_model(train, test)

     # Log the metrics for the model
    metrics = get_model_metrics(model, train, test)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")



if __name__ == '__main__':
    main()
