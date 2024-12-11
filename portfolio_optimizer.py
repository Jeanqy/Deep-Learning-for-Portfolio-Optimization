import pandas as pd
import numpy as np
from scipy.optimize import minimize
from model import Model_LSTM


class portfolio_optimizer:
    def __init__(self, price_long):
        self.price_long = price_long
        self.price_wide = price_long.pivot(index='Date', columns='Ticker', values='Price')
        self.return_wide = self.price_wide.pct_change().iloc[1:,:] #drop 1st row
        self.weight_long = None
    

    def __optimize_1_run_non_ML(self, returns:pd.DataFrame, period:int, loss_func:str, cov_estimation:str):
        # basic feature
        tickers = returns.columns
        date = returns.index[-1]
        num_assets = returns.shape[1]
        mean_returns = np.mean(returns, axis=0)
        annualized_vol = returns.std() * np.sqrt(252)

        def constraint_basic(weights):
            return np.sum(weights) - 1
        
        def constraint_long_only(weights):
            return weights    # make sure weights are positive
        
        def objective_variance(weights):
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_volatility
        
        def objective_sharpe(weights):
            portfolio_return = np.dot(mean_returns, weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return - portfolio_return / portfolio_volatility
        
        def objective_markowitz3(weights):
            a = 1000 # hyperparameter tuning through grid search
            b = 1
            portfolio_return = np.dot(mean_returns, weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = portfolio_return / portfolio_volatility
            entropy = np.exp(-np.sum(weights * np.log(weights)))
            return -(a * sharpe + b * entropy) / (a + b)
        
        def objective_divers_ratio(weights):
            portfolio_variance = 0
            weighted_sum = 0
            for i in range(cov_matrix.shape[0]):
                portfolio_variance += weights[i]**2*cov_matrix[i][i]
                weighted_sum += weights[i]**2*cov_matrix[i][i]
                for j in range(cov_matrix.shape[0]):
                    if i>j:
                        portfolio_variance+= 2*weights[i]*weights[j]*cov_matrix[i][j]
            DR = np.sqrt(weighted_sum/portfolio_variance)
            return - DR

        def objective_marginal_risk_contribution(weights):
            portfolio_TR = 0
            portfolio_variance = 0
            for i in range(cov_matrix.shape[0]):
                portfolio_variance += weights[i]**2*cov_matrix[i][i]
                RC_i = 2*weights[i]**2*cov_matrix[i][i]
                for j in range(cov_matrix.shape[0]):
                    if j!=i:
                        portfolio_variance+= weights[i]*weights[j]*cov_matrix[i][j]
                        RC_i += weights[i]*weights[j]*cov_matrix[i][j]
                portfolio_TR+=RC_i
            return portfolio_TR/np.sqrt(portfolio_variance)
        
        # Covariance Matrix:
        if cov_estimation == 'historical':
            cov_matrix = np.cov(returns.T, ddof=1)
        if cov_estimation == 'regularized':
            c = 0.00005
            cov_matrix = np.cov(returns.T, ddof=1) 
            cov_matrix += c * np.eye(cov_matrix.shape[0])
        if cov_estimation == 'DCC_GARCH':
            vol = mgarch.mgarch()
            vol.fit(returns)
            ndays = 1
            cov_matrix = vol.predict(ndays)['cov']
        
        # calculate weight by minimizing objective, subject to constraint
        constraints = [{'type': 'eq', 'fun': constraint_basic}, {'type': 'ineq', 'fun': constraint_long_only}]
        init_weights = np.array([1 / num_assets] * num_assets)
        if loss_func == 'equal_weight':
            return pd.DataFrame({'Date':date, 
                                 'Ticker':tickers, 
                                 'Weight':init_weights, 
                                 'Annualized_vol':annualized_vol
                                })
        objective = locals()['objective_' + loss_func] # call the objective function associated with input
        weights = minimize(objective, init_weights, method='SLSQP', constraints=constraints).x
        
        return pd.DataFrame({'Date':date, 
                             'Ticker':tickers, 
                             'Weight':weights, 
                             'Annualized_vol':annualized_vol
                            })


    def __optimize_1_run_LSTM(self, price_wide:pd.DataFrame, period:int, loss:str, reg:bool, structure_change):
        tickers = price_wide.columns
        date = price_wide.index[-1]
        return_wide = price_wide.pct_change().dropna()
        num_assets = price_wide.shape[1]
        annualized_vol = return_wide.std() * np.sqrt(252)
        
        model = Model_LSTM(loss = loss, reg = reg, structure_change = structure_change)              
        weights = model.get_allocations(price_wide)
        
        return pd.DataFrame({'Date':date, 
                             'Ticker':tickers, 
                             'Weight':weights, 
                             'Annualized_vol':annualized_vol
                            })


    def optimize(self, 
                 method:str, 
                 cov_estimation:str='historical', 
                 period:int=252, 
                 loss = "paper", 
                 reg = False,
                 structure_change = False,
                ):
        n_rebalance = self.price_wide.shape[0] // period
        weights = pd.DataFrame(columns=['Date', 'Ticker', 'Weight', 'Annualized_vol'])
        if method == "LSTM":
            chunks = [self.price_wide.iloc[i:i+period, :] for i in range(0, period * n_rebalance, period)]
            for chunk in chunks:
                na_threshold = 5
                prices = chunk.dropna(thresh = period - na_threshold, axis=1)
                prices = prices.fillna(method = 'ffill')
                prices = prices.iloc[-50:, :] # paper says only use that last 50 days info, make sure period > 50
                chunk_weights = self.__optimize_1_run_LSTM(prices, period, loss, reg, structure_change)
                weights = pd.concat([weights, chunk_weights], axis=0, ignore_index=True)
        else:
            chunks = [self.return_wide.iloc[i:i+period, :] for i in range(0, period * n_rebalance, period)]
            for chunk in chunks:
                na_threshold = 5
                returns = chunk.dropna(thresh = period-na_threshold, axis=1)
                returns = returns.fillna(0)
                chunk_weights = self.__optimize_1_run_non_ML(returns, period, method, cov_estimation)
                weights = pd.concat([weights, chunk_weights], axis=0, ignore_index=True)

        weights = pd.merge(weights, self.price_long[['Date', 'Ticker', 'Price']], on=['Date', 'Ticker'], how='right')
        weights.sort_values('Date',inplace=True)
        weights["Weight"] = weights.groupby("Ticker")["Weight"].fillna(method='ffill',axis=0)
        weights["Annualized_vol"] = weights.groupby("Ticker")["Annualized_vol"].fillna(method='ffill',axis=0)

        self.weight_long = weights.reset_index(drop=True).dropna()
        print("successfully optimized portfolio weights")
        return self.weight_long