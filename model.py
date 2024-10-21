import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Layer
from tensorflow.keras.optimizers import Adam
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.dates as mdates
import ast



















# model.py
stock_symbol = "AAPL"




























data_path = f'cleaned_{stock_symbol}_data.csv'
df = pd.read_csv(data_path)

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Use the correct column name based on your data inspection
column_name = 'adjusted_close_price'

# Prepare the data for LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[[column_name]].values)

# Function to create a dataset with look_back
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 30
X, Y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into train and test sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
trainX, testX = X[0:train_size], X[train_size:len(X)]
trainY, testY = Y[0:train_size], Y[train_size:len(Y)]

# Define the Attention layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# Function to build and compile the model
def build_model(look_back, learning_rate):
    input_layer = Input(shape=(look_back, 1))
    lstm_layer = LSTM(50, return_sequences=True)(input_layer)
    attention_layer = AttentionLayer()(lstm_layer)
    dense_layer = Dense(1)(attention_layer)
    model = Model(inputs=input_layer, outputs=dense_layer)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# Define date ranges for MAE calculation
date_ranges = [
    ('2015-01-01', '2018-12-31'),
    ('2018-01-01', '2020-12-31'),
    ('2020-01-01', '2022-12-31'),
    ('2022-01-01', '2024-12-31')
]

# Function to calculate MAE for each date range
def calculate_mae(diff_plot, date_ranges):
    mae_ranges = []
    for start_date, end_date in date_ranges:
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        range_diff = diff_plot[mask]
        mae = np.mean(np.abs(range_diff[~np.isnan(range_diff)]))
        mae_ranges.append((start_date, end_date, mae))
    return mae_ranges

# Run the model for different learning rates and save MAE results
learning_rates = np.linspace(0.000001, 0.00005, 10)
results = []

for lr in learning_rates:
    print(f"Running model with learning rate: {lr}")
    model = build_model(look_back, lr)
    history = model.fit(trainX, trainY, epochs=150, batch_size=100, validation_data=(testX, testY), verbose=0)
    
    # Generate predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    # Invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)
    trainY_inv = scaler.inverse_transform([trainY])
    testY_inv = scaler.inverse_transform([testY])
    
    # Calculate the true difference for MAE
    diff_train = trainPredict[:, 0] - trainY_inv[0][:len(trainPredict)]
    diff_test = testPredict[:, 0] - testY_inv[0][:len(testPredict)]
    diff_plot = np.empty_like(scaled_data)
    diff_plot[:, :] = np.nan
    trainPredictStart = look_back
    trainPredictEnd = trainPredictStart + len(trainPredict)
    testPredictStart = trainPredictEnd
    testPredictEnd = testPredictStart + len(testPredict)
    diff_plot[trainPredictStart:trainPredictEnd, 0] = diff_train
    diff_plot[testPredictStart:testPredictEnd, 0] = diff_test
    
    # Calculate MAE
    mae_ranges = calculate_mae(diff_plot, date_ranges)
    results.append((lr, mae_ranges))

# Save results to CSV
results_df = pd.DataFrame(results, columns=['Learning Rate', 'MAE Ranges'])
results_df.to_csv(f'{stock_symbol}_mae_results.csv', index=False) 

print("MAE results saved")









###########################################











# Load the CSV file
file_path = f'{stock_symbol}_mae_results.csv'
df = pd.read_csv(file_path)

# Initialize a dictionary to store the transformed data
transformed_data = {'Learning Rate': df['Learning Rate']}

# Create a set to keep track of all column names
columns_set = set()

# Iterate over the rows to extract and parse the "MAE Ranges" data
for index, row in df.iterrows():
    mae_ranges = ast.literal_eval(row['MAE Ranges'])
    for date_range in mae_ranges:
        start_date, end_date, mae_value = date_range
        col_name = f'{start_date} to {end_date}'
        columns_set.add(col_name)
        if col_name not in transformed_data:
            transformed_data[col_name] = [None] * len(df)
        transformed_data[col_name][index] = mae_value

# Ensure all columns are in the transformed data dictionary
for col in columns_set:
    if col not in transformed_data:
        transformed_data[col] = [None] * len(df)

# Convert the dictionary to a new dataframe
transformed_df = pd.DataFrame(transformed_data)

# Save the transformed dataframe back to the CSV file
transformed_df.to_csv(file_path, index=False)

csv_file = file_path

# Read CSV into DataFrame, ignoring the last two rows
df = pd.read_csv(csv_file, skipfooter=2, engine='python')

# Calculate mean and median across each learning rate row
mean_values = df.iloc[:, 1:].mean(axis=1)  # Assuming columns 1 onwards are the date ranges
median_values = df.iloc[:, 1:].median(axis=1)

# Append mean and median as new columns
df['Mean'] = mean_values
df['Median'] = median_values

# Overwrite the original CSV file with the modified DataFrame
df.to_csv(csv_file, index=False)

# Assuming 'file_path' is your CSV file path
csv_file = file_path

# Read CSV into DataFrame
df = pd.read_csv(csv_file)

# Calculate mean and median across each learning rate row
df['Mean'] = df.iloc[:, 1:-2].mean(axis=1)  # Calculate mean excluding last two columns
df['Median'] = df.iloc[:, 1:-2].median(axis=1)  # Calculate median excluding last two columns

# Find minimum mean and median values and their associated learning rate
min_mean_row = df.loc[df['Mean'].idxmin()]
min_median_row = df.loc[df['Median'].idxmin()]
# Extracting results
min_mean_learning_rate = min_mean_row['Learning Rate']
min_mean_value = min_mean_row['Mean']

min_median_learning_rate = min_median_row['Learning Rate']
min_median_value = min_median_row['Median']










#########################################











# Load data
data_path = f'cleaned_{stock_symbol}_data.csv'
df = pd.read_csv(data_path)
LR = min_median_learning_rate

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Display the columns of the dataframe to confirm
print(df.columns)

# Use the correct column name based on your data inspection
column_name = 'adjusted_close_price'

# Prepare the data for LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[[column_name]].values)

# Function to create a dataset with look_back
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 30
X, Y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into train and test sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
trainX, testX = X[0:train_size], X[train_size:len(X)]
trainY, testY = Y[0:train_size], Y[train_size:len(Y)]

# Define the Attention layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# Build the model
input_layer = Input(shape=(look_back, 1))
lstm_layer = LSTM(50, return_sequences=True)(input_layer)
attention_layer = AttentionLayer()(lstm_layer)
dense_layer = Dense(1)(attention_layer)
model = Model(inputs=input_layer, outputs=dense_layer)

model.compile(optimizer=Adam(learning_rate=LR), loss='mean_squared_error')
model.summary()

# Train the model
history = model.fit(trainX, trainY, epochs=150, batch_size=100, validation_data=(testX, testY), verbose=2)

# Generate predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)
trainY_inv = scaler.inverse_transform([trainY])
testY_inv = scaler.inverse_transform([testY])

# Calculate and print the start and end indices for plotting predictions
trainPredictStart = look_back
trainPredictEnd = trainPredictStart + len(trainPredict)
testPredictStart = trainPredictEnd
testPredictEnd = testPredictStart + len(testPredict)

# Check the lengths of the plotting arrays
trainPredictPlot = np.empty_like(scaled_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[trainPredictStart:trainPredictEnd, 0] = trainPredict[:, 0]

testPredictPlot = np.empty_like(scaled_data)
testPredictPlot[:, :] = np.nan
testPredictPlot[testPredictStart:testPredictEnd, 0] = testPredict[:, 0]

# Calculate the true difference for the second plot
diff_train = trainPredict[:, 0] - trainY_inv[0][:len(trainPredict)]
diff_test = testPredict[:, 0] - testY_inv[0][:len(testPredict)]
diff_plot = np.empty_like(scaled_data)
diff_plot[:, :] = np.nan
diff_plot[trainPredictStart:trainPredictEnd, 0] = diff_train
diff_plot[testPredictStart:testPredictEnd, 0] = diff_test

# Calculate average error (MAE)
average_error = np.mean(np.abs(diff_plot[~np.isnan(diff_plot)]))

# Define date ranges for MAE calculation
date_ranges = [
    ('2015-01-01', '2018-12-31'),
    ('2018-01-01', '2020-12-31'),
    ('2020-01-01', '2022-12-31'),
    ('2022-01-01', '2024-12-31')
]

# Calculate MAE for each date range
mae_ranges = []
for start_date, end_date in date_ranges:
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    range_diff = diff_plot[mask]
    mae = np.mean(np.abs(range_diff[~np.isnan(range_diff)]))
    mae_ranges.append((start_date, end_date, mae))

# Plot actual vs predicted stock prices to visually verify
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

ax1.plot(df['Date'], scaler.inverse_transform(scaled_data), label='Actual Adj Close', color='b')
ax1.plot(df['Date'], trainPredictPlot, label='Train Predict', color='r')
ax1.plot(df['Date'], testPredictPlot, label='Test Predict', color='g')
ax1.set_title('Actual vs Predicted Stock Prices')
ax1.set_ylabel('Adjusted Close Price')
ax1.legend()

# Plot true difference
ax2.plot(df['Date'], diff_plot, label='Difference (Predicted - Actual)', color='m')
ax2.axhline(y=0, color='k')  # Add a horizontal line at y=0
ax2.set_title('Difference Between Actual and Predicted Stock Prices')
ax2.set_xlabel('Date')
ax2.set_ylabel('Difference')
ax2.legend()

# Display average error for date ranges on the plot
y_pos = 0.05
for start_date, end_date, mae in mae_ranges:
    ax2.text(0.05, y_pos, f'MAE {start_date} to {end_date}: {mae:.2f}', transform=ax2.transAxes, fontsize=12, verticalalignment='bottom')
    y_pos += 0.05

# Set x-ticks to be more spread out and readable
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Change interval as needed
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Function to process each chunk of data
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

def process_chunk(chunk):
    return create_dataset(chunk, look_back)

# Determine the number of CPUs
n_jobs = multiprocessing.cpu_count()

# Parallel processing
chunks = np.array_split(scaled_data, n_jobs)
results = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(chunk) for chunk in chunks)
X, Y = zip(*results)
X = np.concatenate(X)
Y = np.concatenate(Y)




###########################################


# Load data
data_path = f'cleaned_{stock_symbol}_data.csv'
df = pd.read_csv(data_path)

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Use the correct column name based on your data inspection
column_name = 'adjusted_close_price'

# Prepare the data for LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[[column_name]].values)

# Function to create a dataset with look_back
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 30
X, Y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into train and test sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
trainX, testX = X[0:train_size], X[train_size:len(X)]
trainY, testY = Y[0:train_size], Y[train_size:len(Y)]

# Define the Attention layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# Build the model
input_layer = Input(shape=(look_back, 1))
lstm_layer = LSTM(50, return_sequences=True)(input_layer)
attention_layer = AttentionLayer()(lstm_layer)
dense_layer = Dense(1)(attention_layer)
model = Model(inputs=input_layer, outputs=dense_layer)

model.compile(optimizer=Adam(learning_rate=LR), loss='mean_squared_error')
model.summary()

# Train the model
history = model.fit(trainX, trainY, epochs=150, batch_size=100, validation_data=(testX, testY), verbose=2)

# Generate predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)
trainY_inv = scaler.inverse_transform([trainY])
testY_inv = scaler.inverse_transform([testY])

# Extrapolate two weeks ahead
extrapolation_days = 14
last_data = scaled_data[-look_back:].reshape(1, look_back, 1)
extrapolated_data = []

for _ in range(extrapolation_days):
    next_pred_scaled = model.predict(last_data)
    next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]
    extrapolated_data.append(next_pred)
    next_pred_scaled = next_pred_scaled.reshape((1, 1, 1))  # Reshape to (1, 1, 1)
    last_data = np.append(last_data[:, 1:, :], next_pred_scaled, axis=1)

# Prepare the dates for the extrapolated period
last_date = df['Date'].iloc[-1]
extrapolated_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=extrapolation_days)

# Plot the actual and test predictions for the last 23 days and the next 14 extrapolated days
fig, ax = plt.subplots(figsize=(12, 6))

# Get the last 23 days of actual data and test predictions (30 - 7 days)
last_23_days_actual = scaler.inverse_transform(scaled_data[-23:])
last_23_days_dates = df['Date'].iloc[-23:]

# Plot the last 23 days of actual data and test predictions
ax.plot(last_23_days_dates, last_23_days_actual, label='Actual Last 23 Days', color='b')
ax.plot(last_23_days_dates, testPredict[-23:], label='Test Predict Last 23 Days', color='g')

# Plot the next 7 days of extrapolated predictions
ax.plot(extrapolated_dates[:7], extrapolated_data[:7], label='Extrapolated Next 7 Days', color='orange')
# Plot the subsequent 7 days of extrapolated predictions in a different color
ax.plot(extrapolated_dates[7:], extrapolated_data[7:], label='Extrapolated 8-14 Days', color='purple')

ax.set_title('Actual and Test Predictions for Last 23 Days and Extrapolated Next 14 Days')
ax.set_xlabel('Date')
ax.set_ylabel('Adjusted Close Price')
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()






