# Mine Safety Hazard Prediction using LSTMs

This repository contains a Jupyter Notebook that builds and evaluates deep learning models to predict hazard levels in a mine based on time-series sensor data.

## Project Overview

The goal of this project is to predict a continuous "Hazard" index based on 30 preceding time steps of sensor readings. This is treated as a time-series regression problem. The notebook explores, preprocesses, and models the data using Keras and TensorFlow.

The core of the analysis involves comparing three different recurrent neural network (RNN) architectures:
1.  A baseline single-layer LSTM.
2.  A deeper, Stacked LSTM model.
3.  A hybrid CNN-LSTM model.

The final model (Stacked LSTM) achieves a high degree of accuracy in predicting the hazard index.

## Dataset

The analysis uses the `mine_hazard_data.csv` file, which contains the following columns:

* **Time Stamp**: The timestamp for each reading.
* **Sensor Features (12):**
    * `Hydrogen Sulfide (H₂S)`
    * `Carbon Monoxide (CO)`
    * `Nitrogen Oxides (NOₓ)`
    * `Sulfur Dioxide (SO₂)`
    * `Methane (CH₄)`
    * `Temperature (°C)`
    * `Humidity (%)`
    * `Total Gas Concentration`
    * `Heat Index`
    * `Dew Point`
    * `Toxic Gas Index`
    * `Temperature-Humidity Index`
* **Target Variable:**
    * `Hazard`: A continuous numerical index representing the hazard level (ranging from 1.25 to 8.125).

## Methodology

1.  **Data Loading & Exploration:** The `mine_hazard_data.csv` is loaded into a `pandas` DataFrame. Initial statistics are reviewed using `df.describe()` and `df.head()`.

2.  **Data Preprocessing:**
    * **Feature/Target Separation:** The `Hazard` column is selected as the target (Y), and the 12 sensor readings are used as features (X).
    * **Scaling:** Features (X) and the target (Y) are independently scaled using `MinMaxScaler`. The target is scaled to a range of [0, 10].
    * **Sequence Generation:** The data is transformed into sequences. Each sample consists of 30 time steps of all 12 features, with the target being the `Hazard` value at the 30th time step.
    * **Train/Test Split:** The sequenced data is split into training (80%) and testing (20%) sets.

3.  **Model Development & Comparison:**
    Three Keras models are defined, trained (for 25 epochs), and evaluated:
    * **`LSTM_Baseline`**: A single LSTM layer (50 units) followed by a Dense output layer.
    * **`Stacked_LSTM`**: A model with two stacked LSTM layers (64 and 32 units) with Dropout, followed by two Dense layers.
    * **`CNN_LSTM`**: A hybrid model starting with a 1D Convolutional layer and MaxPooling, followed by an LSTM layer and two Dense layers.

## Results

The models were evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).

| Model | MSE | MAE | R² Score |
| :--- | :--- | :--- | :--- |
| `LSTM_Baseline` | 33.376 | 5.656 | -23.02 |
| **`Stacked_LSTM`** | **0.207** | **0.363** | **0.851** |
| `CNN_LSTM` | 4.189 | 1.948 | -2.02 |

The **Stacked LSTM** model was the clear winner, achieving an **R² score of 0.851**. The other two models failed to learn the task effectively.

### Visualizations

The notebook includes several plots to analyze the model's performance:

* **Validation Loss:** Shows the Stacked LSTM model converging while the others fail.
* **Actual vs. Predicted Scatter Plot:** Demonstrates a strong linear correlation between the actual and predicted hazard values from the best model.
* **Time-Series Prediction:** A line plot comparing the actual vs. predicted hazard index for the first 100 test samples, showing the model's ability to track changes.
* **Confusion Matrix:** The continuous hazard predictions are discretized into five categories ("Very Low" to "Very High") to show classification accuracy. The model performs very well, with most predictions falling on the main diagonal.
* **Mine Hazard Heatmap:** A *simulated* example heatmap showing how these hazard predictions could be visualized over a 2D mine grid.

## How to Run

1.  Ensure you have the required libraries installed:
    ```bash
    pip install pandas numpy matplotlib scikit-learn tensorflow seaborn jupyter
    ```
2.  Place the `mine_hazard_data.csv` file in the same directory as the notebook.
3.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook mine-safety-msm.ipynb
    ```
4.  Run all cells in the notebook.

## Dependencies

* Python 3
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* tensorflow (keras)
