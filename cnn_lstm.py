import numpy as np  # Linear algebra operations
import pandas as pd  # Data processing and CSV file I/O
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns  # Advanced visualization
from tensorflow.keras.layers import Dense, RepeatVector, LSTM, Dropout
from tensorflow.keras.layers import Flatten, Conv1D, MaxPooling1D, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys
import logging

# Configure logging to capture all outputs and errors to a log file
logging.basicConfig(
    filename="script_output.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Redirect stdout and stderr to log file
sys.stdout = open("script_output.log", "a")
sys.stderr = sys.stdout

def main():
    # Load the dataset
    df = pd.read_csv("testset.csv")
    logging.info("Dataset loaded successfully.")

    # Display the first few rows of the dataset
    print("First 5 rows of the dataset:")
    print(df.head())

    # Analyze weather conditions
    print("Weather condition frequency:")
    print(df[' _conds'].value_counts())

    # Plot the top 15 weather conditions
    plt.figure(figsize=(15, 10))
    df[' _conds'].value_counts().head(15).plot(kind='bar')
    plt.title('Top 15 Most Common Weather Conditions in Delhi')
    plt.show()

    # Analyze common wind directions
    plt.figure(figsize=(15, 10))
    plt.title("Common Wind Directions in Delhi")
    df[' _wdire'].value_counts().plot(kind="bar")
    plt.show()

    # Analyze temperature distribution
    plt.figure(figsize=(15, 10))
    sns.histplot(df[' _tempm'], bins=range(0, 61, 5), kde=False, color='blue')
    plt.title("Temperature Distribution in Delhi")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

    # Convert datetime column to pandas datetime
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    logging.info("Converted datetime column to pandas datetime format.")

    # Handle missing temperature values by filling them with the mean
    df[' _tempm'].fillna(df[' _tempm'].mean(), inplace=True)
    print("Missing temperature values filled with mean.")

    # Extract year and month from datetime for further analysis
    df['year'] = df['datetime_utc'].dt.year.astype(str)
    df['month'] = df['datetime_utc'].dt.month.astype(str).str.zfill(2)

    # Create a heatmap of average monthly temperatures over the years
    temp_year = pd.crosstab(df['year'], df['month'], values=df[' _tempm'], aggfunc='mean')
    plt.figure(figsize=(15, 10))
    sns.heatmap(temp_year, cmap='coolwarm', annot=True, fmt=".1f")
    plt.title("Average Monthly Temperatures in Delhi (1996-2017)")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.show()

    # Handle missing humidity values by filling them with the mean
    df[' _hum'].fillna(df[' _hum'].mean(), inplace=True)
    print("Missing humidity values filled with mean.")

    # Create a heatmap of average monthly humidity over the years
    humidity_year = pd.crosstab(df['year'], df['month'], values=df[' _hum'], aggfunc='mean')
    plt.figure(figsize=(15, 10))
    sns.heatmap(humidity_year, cmap='coolwarm', annot=True, fmt=".1f")
    plt.title("Average Monthly Humidity in Delhi (1996-2017)")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.show()

    # Prepare data for time series forecasting
    data = pd.DataFrame(df[' _tempm'].values, index=df['datetime_utc'], columns=['temp'])
    data = data.resample('D').mean()  # Resample daily

    # Handle missing values in the resampled data
    data.fillna(data['temp'].mean(), inplace=True)

    print(f"Final dataset shape after resampling: {data.shape}")
    print(data.head())

    # Plot the time series
    plt.figure(figsize=(25, 7))
    plt.plot(data, linewidth=0.5, color='blue')
    plt.title("Time Series of Temperature in Delhi")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.grid()
    plt.show()

    # Scale the data to a range (-1, 1)
    scalar = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scalar.fit_transform(data)

    # Create input-output pairs for training and testing
    steps = 30  # Time steps
    inp, out = [], []

    for i in range(len(data_scaled) - steps):
        inp.append(data_scaled[i:i+steps])
        out.append(data_scaled[i+steps])

    inp = np.array(inp)
    out = np.array(out)

    # Split data into training and testing sets
    x_train, x_test = inp[:7300], inp[7300:]
    y_train, y_test = out[:7300], out[7300:]

    print(f"Training set shape: {x_train.shape}")
    print(f"Testing set shape: {x_test.shape}")

    # Define the CNN-LSTM model
    early_stop = EarlyStopping(monitor="loss", mode="min", patience=7)

    model = Sequential([
        Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(30, 1)),
        Conv1D(filters=128, kernel_size=2, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        RepeatVector(30),
        LSTM(units=100, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(units=100, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(units=100, return_sequences=True, activation='relu'),
        LSTM(units=100, return_sequences=True, activation='relu'),
        Bidirectional(LSTM(128, activation='relu')),
        Dense(100, activation='relu'),
        Dense(1)
    ])

    model.compile(loss='mse', optimizer='adam')

    # Print the model structure
    print("Model Summary:")
    model.summary()

    # Train the model
    print("Training the model...")
    history = model.fit(x_train, y_train, epochs=2, verbose=1, callbacks=[early_stop])

    # Save the trained model
    model.save("./regressor.hdf5")
    logging.info("Model training complete and saved as regressor.hdf5.")

    # Make predictions
    predict = model.predict(x_test)
    predict = scalar.inverse_transform(predict)
    Ytesting = scalar.inverse_transform(y_test)

    # Plot actual vs predicted temperatures
    plt.figure(figsize=(20, 9))
    plt.plot(Ytesting, label='Actual Temperatures', color='blue', linewidth=2)
    plt.plot(predict, label='Predicted Temperatures', color='red', linewidth=2)
    plt.title("Actual vs Predicted Temperatures")
    plt.legend()
    plt.show()

    # Evaluate model performance
    mse = mean_squared_error(Ytesting, predict)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    main()

