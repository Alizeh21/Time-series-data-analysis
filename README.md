# Time-series-data-analysis

# Weather-forecasting-Timeseries-data
Our project focuses on leveraging time series data tracking carbon emissions in parts per million (ppm) over the years. We've undertaken the task of forecasting future trends in carbon emissions using various predictive models. The primary objective is to develop a comprehensive forecasting system capable of implementing and comparing different time series models such as ARIMA, Artificial Neural Networks (ANN), and a Hybrid ARIMA-ANN approach across multiple sectors. Additionally, we aim to provide a user-friendly frontend interface for visualizing both historical data and forecasted trends, enhancing accessibility and usability for users.

## Table of Contents
- [Technologies used](#Technoloies)
- [Usage](#usage)
- [Features](#features)
- [Credits](#credits)

## Technologies Used
- Backend: Python with Flask for server-side logic, handling API requests.
-  Frontend: Node JS for building a dynamic and responsive user interface, HTML/CSS for layout and
styling.
- Data Science: Python with Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, TensorFlow/Keras.

## Usage
Please note that this project is still under development and is continually being improved. Our ultimate goal is to enhance its capabilities to the extent that it can accurately predict weather patterns in real-time. As such, while you can currently explore and utilize the existing features, expect further enhancements and refinements as we progress towards this ambitious objective. Stay tuned for updates!

## Features

### Preprocessing Steps:
• Cleaning: Identify and impute or remove missing values.
• Normalization/Standardization: Scale the data to a uniform range.
• Stationarization: Apply differencing and logarithmic transformations as necessary to achieve
stationarity.

### Model Development

#### ARIMA Configuration and Tuning:
- **Purpose:** ARIMA (Autoregressive Integrated Moving Average) is utilized to model and forecast time series data that exhibit levels of non-stationarity or seasonal patterns.
- **Process:** Identify the order of differencing (d), the number of autoregressive terms (p), and the number of lagged forecast errors in the prediction equation (q) using statistical tests like the ADF (Augmented Dickey-Fuller) test for stationarity and ACF/PACF plots for parameter estimation.

#### ANN Design and Training:
- **Purpose:** Artificial Neural Networks (ANN) are employed for their capability to model complex nonlinear relationships and interactions in data, which might be missed by ARIMA.
- **Process:** Design neural networks with varying architectures, including different numbers of layers and neurons, to find the best fit for the dataset. Implement backpropagation to train the models using historical data.

#### SARIMA (Seasonal ARIMA):
- **Purpose:** SARIMA extends ARIMA to specifically model and forecast seasonal time series data, accounting for both non-stationary and seasonal changes within a dataset.
- **Process:** Determine the seasonal order of differencing (D), the number of seasonal autoregressive terms (P), and the number of seasonal lagged forecast errors (Q) along with the non-seasonal parameters. Utilize the ADF test for overall stationarity and use seasonal ACF and PACF plots to estimate the seasonal parameters.

#### Exponential Smoothing (ETS):
- **Purpose:** Exponential Smoothing models are used to forecast time series data by applying weighted averages of past observations, with the weights decaying exponentially over time. The model is particularly effective for data with trends and seasonalities.
- **Process:** Select appropriate models based on data characteristics—simple, double, or triple exponential smoothing. Use error, trend, and seasonal components (ETS) to configure the model. Parameters are typically selected based on model fit criteria such as AIC or BIC.

#### Prophet:
- **Purpose:** Prophet is designed to handle time series with strong seasonal effects and historical holidays, making it ideal for daily data with multiple seasonality patterns and irregularities.
- **Process:** Define the model with potentially yearly, weekly, and daily seasonality components and holiday effects. Adjust the model’s flexibility by tuning the seasonality mode and adding custom seasonality if needed. The model parameters are optimized automatically using a scalable fitting procedure.

#### Support Vector Regression (SVR):
- **Purpose:** SVR applies the principles of support vector machines to regression problems, modeling nonlinear relationships in the data through the use of kernel functions.
- **Process:** Choose an appropriate kernel (linear, polynomial, radial basis function) and tune parameters such as C (regularization parameter) and gamma (kernel coefficient). Use cross-validation to ensure the model generalizes well to unseen data.

#### Long Short-Term Memory (LSTM):
- **Purpose:** LSTM networks are a type of recurrent neural network (RNN) suitable for sequence prediction problems. They are capable of learning order dependence in sequence prediction problems.
- **Process:** Design the network architecture with one or more LSTM layers, define the number of neurons in each layer, and select a backpropagation algorithm for training (typically using Adam or SGD). The model learns from sequences of historical data points, with the length of input sequences being a tunable parameter.

#### Hybrid Models Integration:
- **Purpose:** Combining ARIMA and ANN models leverages ARIMA's proficiency in capturing linear relationships and ANN's ability to model complex patterns. This integration aims to enhance overall forecast accuracy by handling residuals effectively.
- **Process:** Use the forecast results from the ARIMA model as input features to the ANN, which then models the residuals. This step is crucial as it allows the ANN to correct and improve the predictions based on the errors generated by the ARIMA model.


#### Credits
Extending special acknowledgments to @potetoepotatoe for their invaluable partnership in this project.

## COLLABORATORS

> [Alizeh21](https://github.com/Alizeh21) (i211775@nu.edu.pk)

> [potetoepotatoe](https://github.com/PotetoePotatoe) (i211367@nu.edu.pk Maryam Salman)

 
