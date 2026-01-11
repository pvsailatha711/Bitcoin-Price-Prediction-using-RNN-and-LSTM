# Bitcoin Price Prediction using RNN and LSTM

## Overview

This project leverages deep learning techniques to predict Bitcoin prices using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks. The model analyzes historical price data spanning nearly seven years to forecast future price movements. Additionally, the project incorporates sentiment analysis from Twitter data to explore the relationship between social media sentiment and cryptocurrency market trends.

## Dataset

### Price Data
The primary dataset consists of historical Bitcoin price information:

- **Source**: `BTC-USD.csv` (Bitcoin to USD exchange rate)
- **Time Period**: September 17, 2014 to July 9, 2021
- **Total Entries**: 2,548 daily records
- **Features**:
  - `Date`: Trading date
  - `Open`: Opening price
  - `High`: Highest price of the day
  - `Low`: Lowest price of the day
  - `Close`: Closing price (primary target variable)
  - `Adj Close`: Adjusted closing price
  - `Volume`: Trading volume

### Sentiment Data
A supplementary dataset for sentiment analysis:

- **Source**: Bitcoin-related tweets from 2016 to 2019
- **Purpose**: Correlate social media sentiment with price movements
- **Classification**: Tweets categorized as "Increase," "Decrease," or "Neutral" based on daily price changes
- **Distribution**: Approximately 52% Decrease sentiment and 48% Increase sentiment

## Methodology

### Data Preprocessing

1. **Missing Value Handling**: Null values filled with column means to maintain data integrity
2. **Data Split**: 80-20 train-test split (2,038 training samples, 510 test samples)
3. **Feature Scaling**: MinMaxScaler applied to normalize closing prices between 0 and 1
4. **Sequence Creation**: 60-day lookback window used to predict the next day's price

### Model Architecture

The project implements a sophisticated Stacked LSTM architecture:

- **Model Type**: Sequential
- **LSTM Layers**: Four stacked LSTM layers, each with 50 units
- **Activation**: Default tanh activation for LSTM cells
- **Regularization**: Dropout of 0.2 (20%) after each LSTM layer to prevent overfitting
- **Output Layer**: Dense layer with 1 unit for continuous price prediction

### Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Evaluation Metrics**: MSE and Mean Absolute Error (MAE)
- **Epochs**: 50
- **Batch Size**: 50

## Results

The LSTM model demonstrates strong predictive performance on Bitcoin price forecasting:

### Training Performance
- **Final Training Loss (MSE)**: 0.0008
- **Final Training MAE**: 0.0187

### Test Performance
- **Test Loss (MSE)**: 0.0013
- **Test MAE**: 0.0252

### Key Findings

The model successfully captures Bitcoin's price trends and patterns, as evidenced by the close alignment between predicted and actual prices in the visualization. The low MSE and MAE values indicate that the model's predictions are highly accurate, with minimal deviation from actual market prices.

The comparative plot of "Real Stock Price" vs. "Predicted Stock Price" shows that the LSTM model effectively follows actual market trends, demonstrating its ability to learn complex temporal dependencies in cryptocurrency price movements.

## Sentiment Analysis Insights

The sentiment analysis component reveals interesting patterns:

- Social media sentiment shows a relatively balanced distribution between positive and negative market expectations
- The correlation between tweet sentiment and price movements provides additional context beyond numerical historical data
- This dual approach (price prediction + sentiment analysis) offers a more comprehensive view of market dynamics

## Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for building and training LSTM networks
- **Scikit-learn**: Data preprocessing and scaling utilities
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization for price trends and predictions

## Model Advantages

1. **LSTM Architecture**: Effectively handles long-term dependencies in sequential data, making it ideal for time series forecasting
2. **Stacked Layers**: Multiple LSTM layers enable the model to learn hierarchical temporal features
3. **Dropout Regularization**: Prevents overfitting and improves generalization to unseen data
4. **Sentiment Integration**: Combines quantitative price data with qualitative social sentiment for richer insights

## Future Improvements

Potential enhancements to further improve model performance:

1. **Hyperparameter Tuning**: Optimize number of LSTM units, layers, dropout rates, and learning rate
2. **Additional Features**: Incorporate trading volume, market capitalization, and other technical indicators
3. **Advanced Architectures**: Experiment with Bidirectional LSTM, GRU, or Transformer models
4. **Multi-Step Prediction**: Extend the model to predict prices multiple days ahead
5. **Ensemble Methods**: Combine multiple models for more robust predictions
6. **Real-Time Sentiment**: Integrate live Twitter sentiment analysis for dynamic predictions
7. **External Factors**: Include macroeconomic indicators, regulatory news, and market events

## Acknowledgments

Dataset sourced from Yahoo Finance (BTC-USD historical data) and Twitter sentiment data for Bitcoin-related discussions.
