# Taxi Demand Predictor

## Overview

The Taxi Demand Predictor is a machine learning application designed to predict taxi demand in a given area using historical data. This project leverages various data science and machine learning techniques to provide accurate demand forecasts, which can be beneficial for taxi companies and ride-sharing services.

## Features

- **Data Ingestion**: Fetches and processes historical taxi demand data.
- **Machine Learning Models**: Utilizes models such as LightGBM and XGBoost for demand prediction.
- **Interactive Visualization**: Provides a user-friendly interface using Streamlit for visualizing predictions and trends.
- **Geospatial Analysis**: Integrates geospatial data to enhance prediction accuracy based on location.
- **Feature Store Integration**: Uses Hopsworks for managing and serving features for model training and inference.

## Technologies Used

- **Python**: The primary programming language for data processing and model development.
- **Streamlit**: A framework for building interactive web applications.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For implementing machine learning algorithms.
- **LightGBM**: For gradient boosting framework that uses tree-based learning algorithms.
- **XGBoost**: An optimized distributed gradient boosting library.
- **Geopandas**: For geospatial data processing.
- **Hopsworks**: For managing feature stores and serving features to models.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/taxi_demand_predictor.git
   cd taxi_demand_predictor
   ```

2. Install Poetry if you haven't already:

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Install the project dependencies:

   ```bash
   poetry install
   ```

4. Create a `.env` file in the project root and add your HOPSWORKS_API_KEY:

   ```plaintext
   HOPSWORKS_API_KEY="your_api_key_here"
   ```

## Usage

To run the application, use the following command:

```bash
poetry run streamlit run src/frontend.py
```

Open your web browser and navigate to `http://localhost:8501` to access the application.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hopsworks](https://hopsworks.ai/) for providing the feature store.
- [Streamlit](https://streamlit.io/) for enabling easy web app development.
- The open-source community for their contributions and support.
