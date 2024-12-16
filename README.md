# CNN-LSTM Temperature Forecasting

## Overview
This project demonstrates the use of a Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) layers to forecast daily temperatures in Delhi based on historical weather data. By leveraging the strengths of CNNs and LSTMs, this project aims to model temporal patterns in temperature data effectively and generate accurate forecasts.

## How It Works

### 1. Data Loading and Preprocessing
- **Dataset**: The model uses a CSV file (`testset.csv`) containing weather data such as temperature, humidity, wind direction, and weather conditions.
- **Steps**:
  1. Load the dataset and handle missing values by imputing the mean for temperature and humidity.
  2. Extract relevant time-based features such as `year` and `month` from the `datetime` column.
  3. Resample the temperature data to a daily frequency and scale it to a range of `(-1, 1)` for training.

### 2. Data Visualization
- Generate exploratory visualizations, including:
  - Frequency distribution of weather conditions.
  - Common wind directions in Delhi.
  - Temperature and humidity heatmaps over time.
  - Line plots to show temperature trends.

### 3. Model Architecture
The model is built using TensorFlow/Keras and comprises:
- **Convolutional Layers**: Extract spatial features from sequences.
- **LSTM Layers**: Model temporal dependencies in the time series data.
- **Bidirectional LSTMs**: Capture forward and backward patterns in the sequence.
- **Dense Layers**: Produce the final temperature prediction.

The structure of the model is printed in the output and logged for reference.

### 4. Training
- Data is split into training and testing sets.
- Training is performed over 2 epochs (configurable), with early stopping to prevent overfitting.
- The model is saved as `regressor.hdf5` after training.

### 5. Evaluation
- The model's performance is evaluated using the Mean Squared Error (MSE) metric.
- Predicted temperatures are compared visually with actual values on a plot.

### 6. Logging
- All outputs, including data summaries, model structure, and evaluation metrics, are logged into a file named `script_output.log`.

## Features
- **Scalable Framework**: Easily modify the number of layers, neurons, or training epochs.
- **Visual Insights**: Detailed plots for exploratory data analysis and result validation.
- **Reproducibility**: Outputs are logged for consistent tracking and debugging.

## Requirements
To run this project, you need:

- **Python**: Version 3.8 or higher
- **Dependencies**:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - tensorflow
  - scikit-learn

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. Place the dataset (`testset.csv`) in the project directory.
2. Run the script:
```bash
python script_name.py
```
3. Outputs will be saved in `script_output.log` and visualizations will be displayed during execution.

## Goals Achieved
- **Exploratory Analysis**: Gain insights into Delhi’s historical weather patterns.
- **Accurate Forecasting**: Predict daily temperatures using advanced deep learning techniques.
- **Reproducibility**: Provide a comprehensive and reusable framework for time series forecasting.


