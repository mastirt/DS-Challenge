# Fruit Quality Prediction App

This repository contains a Streamlit-based web application for predicting the quality of fruits based on various features. The model and scaler used in this application were pre-trained and saved using joblib.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## Overview
The Fruit Quality Prediction App takes in features such as size, weight, sweetness, acidity, softness, harvest time, ripeness, color, and blemishes of a fruit to predict its quality on a scale.

## Features
- User-friendly interface for inputting fruit features.
- Real-time prediction of fruit quality.
- Uses a pre-trained Random Forest model for prediction.

## Requirements
- Python 3.7 or higher
- Streamlit
- Pandas
- Numpy
- Scikit-learn
- Joblib

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/mastirt/DS-Challenge.git
    cd DS-Challenge
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Ensure that `model_rf.pkl` and `feature_scalar.pkl` are in the project directory.
2. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```
3. Open your web browser and go to `http://localhost:8501`.

## Model Details
The prediction model is a Random Forest model trained on a dataset of fruit features. The model predicts the quality of the fruit on a scale. The features used for prediction include:
- Size (cm)
- Weight (g)
- Brix (Sweetness)
- pH (Acidity)
- Softness (1-5)
- Harvest Time (days)
- Ripeness (1-5)
- Color
- Blemishes (Y/N)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
