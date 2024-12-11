# Deep Learning for Portfolio Optimization

## Overview
This project leverages deep learning techniques to optimize financial portfolios and dynamically rebalancing allocations. By integrating historical market data and advanced neural network architectures, the project aims to maximize portfolio performance while managing risk effectively.

## Features
- **Data Processing**: Preprocessing historical market data for training and validation using scripts in the `data_clean/` directory.
- **Model Architecture**: Custom deep learning models defined in `model.py` for financial time series data.
  - Includes layers like CNN and LSTM for capturing temporal and spatial dependencies.
- **Portfolio Optimization**: Core portfolio optimization logic implemented in `portfolio_optimizer.py`.
- **Backtesting**: Comprehensive backtesting framework available in `backtest.py` to evaluate strategies against historical data.
- **Research Papers**: Supporting theoretical insights and references stored in the `paper/` directory.
- **Jupyter Notebook**: Interactive workflow and examples in `main.ipynb`.

## Project Structure
- `main.ipynb`: Jupyter notebook containing the main code for the project, including data preprocessing, model training, and evaluation.
- `data_clean/`: Directory for data cleaning and preprocessing scripts.
- `paper/`: Directory containing research papers and references.
- `backtest.py`: Python script for backtesting portfolio strategies.
- `model.py`: Python script defining deep learning model architectures.
- `portfolio_optimizer.py`: Python script for implementing portfolio optimization algorithms.

## Requirements
- Python 3.8+
- Key Python libraries:
  - TensorFlow/PyTorch
  - NumPy
  - Pandas
  - Matplotlib/Seaborn
  - Scikit-learn

Install required dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**:
   - Ensure the data files are placed in the `data_clean/` directory.
   - Update paths in `main.ipynb` and relevant scripts as needed.

2. **Model Training**:
   - Use `model.py` to define and train deep learning models.
   - Run the cells in `main.ipynb` to preprocess data and train the model.

3. **Portfolio Optimization**:
   - Use `portfolio_optimizer.py` for portfolio allocation.

4. **Backtesting**:
   - Evaluate the model and strategy using `backtest.py`.
   - Generate performance metrics and visualizations.

## Results
The project outputs include:
- Performance plots showing cumulative returns over time.
- Metrics such as Sharpe ratio and portfolio volatility.

