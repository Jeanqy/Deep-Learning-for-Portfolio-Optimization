import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, jarque_bera
import matplotlib.pyplot as plt
from typing import List


# Auxuliary Functions
def maximum_drawdown(pnl: pd.Series):
    nav = (1 + pnl).cumprod()
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_drawdown = drawdown.min()
    max_drawdown_date = drawdown.idxmin()
    
    if max_drawdown == 0:
        return [drawdown, 0, np.NaN, np.NaN]
    
    def find_nearest_zeros(series):
        min_index = series.idxmin()
        left_zero_index = series[series <= 0].loc[:min_index][::-1].idxmax()
        righ_zero_index = series[series <= 0].loc[min_index:].idxmax()
        return left_zero_index, righ_zero_index
    
    drawdown_start_date, drawdown_end_date = find_nearest_zeros(drawdown)
    recovery_days = (drawdown_end_date - drawdown_start_date).total_seconds() / 86400.0
    
    return [drawdown, max_drawdown, max_drawdown_date, recovery_days]



def backtest(weight_long):
    data = weight_long.copy()
    data["ret"] = data.groupby("Ticker")["Price"].transform(lambda x: x.shift(-1)/x -1)
    data["pnl"] = data["ret"] * data["Weight"]
    port_pnl = data.groupby("Date").apply(lambda x: np.sum(x["pnl"]))
    port_pnl.index = pd.to_datetime(port_pnl.index)
    port_nav = port_pnl.cumsum() + 1
    port_vol = weight_long.groupby('Date').apply(lambda x: np.sqrt(np.sum((x['Weight'] * x['Annualized_vol'])**2)))
    
    def calc_var(port_pnl, confidence_level = 0.05):
        sorted_pnl = port_pnl.sort_values()
        index = int(confidence_level * len(sorted_pnl))
        return sorted_pnl.iloc[index]
    
    def calc_cvar(port_pnl, confidence_level = 0.05):
        var = calc_var(port_pnl, confidence_level)
        pnl_below_var = port_pnl[port_pnl <= var]
        return pnl_below_var.mean()
    
    avg_annual_ret = port_pnl.mean() * 252
    avg_annual_std = port_pnl.std() * np.sqrt(252)
    s = skew(port_pnl)
    k = kurtosis(port_pnl)
    sharpe_ratio = avg_annual_ret / avg_annual_std
    adj_sharpe_ratio = sharpe_ratio * (1 + s/6*sharpe_ratio - k/24*sharpe_ratio**2)
    drawdown_results = maximum_drawdown(port_pnl)
    
    shannon_entropy = data.groupby("Date")["Weight"].apply(lambda x: np.exp(-np.sum(x * np.log(x))))
    shannon_entropy.index = port_pnl.index
    weighted_vol = weight_long.groupby('Date').apply(lambda x: np.sum(x['Weight'] * x['Annualized_vol']))
    diversification_ratio = weighted_vol / port_vol
    
    return {'avg annualized ret': avg_annual_ret,
            'avg annualized std': avg_annual_std,
            'sharpe ratio': sharpe_ratio, 
            'adjusted sharpe ratio': adj_sharpe_ratio,
            'skewness': s,
            'excess kurtosis': k,
            'maximum drawdown': drawdown_results[1],
            'maximum drawdown length (days)': drawdown_results[3],
            'VaR (95%)': calc_var(port_pnl),
            'CVaR (95%)': calc_cvar(port_pnl),
            # 'shannon entropy mean': shannon_entropy.mean(),
            # 'shannon entropy std': shannon_entropy.std(),
            # 'diversification ratio mean': diversification_ratio.mean(),
            # 'diversification ratio std': diversification_ratio.std(),
            'effective number of uncorrelated bets': np.square(diversification_ratio).mean(),
            # time series data below
            'pnl': port_pnl,
            'nav': port_nav,
            # 'annualized_realized_vol': port_vol,
            'drawdown': drawdown_results[0],
            # 'shannon_entropy': shannon_entropy,
            # 'diversification_ratio': diversification_ratio,
           }


def comparison_table(result_list):
    # this compiles all numerical results (not a timeseries)
    comparison = pd.DataFrame()
    for result in result_list:
        float_values = {key: value for key, value in result.items() if isinstance(value, float)}
        tmp_df = pd.DataFrame(float_values.values(), index=float_values.keys())
        comparison = pd.concat([comparison, tmp_df], axis = 1)
    return comparison


def display_backtest_results(list_of_weight_long, list_of_names = []):
    list_of_results = []
    for weight_long in list_of_weight_long:
        list_of_results.append(backtest(weight_long))

    table = comparison_table(list_of_results)
    if list_of_names != []:
        table.columns = list_of_names
    display(table)

    plot_lists = ['nav', 
                  'pnl', 
                  # 'annualized_realized_vol', 
                  'drawdown',
                  # 'shannon_entropy', 
                  # 'diversification_ratio'
                 ]

    fig, axs = plt.subplots(len(plot_lists), figsize = (10, 5*len(plot_lists)))
    for i in range(len(plot_lists)):
        for j in range(len(list_of_results)):
            if list_of_names != []:
                axs[i].plot(list_of_results[j][plot_lists[i]], label = f"{list_of_names[j]}", linewidth = 1)
            else:
                axs[i].plot(list_of_results[j][plot_lists[i]], label = f"Strategy {j + 1}", linewidth = 1)
        axs[i].legend()
        axs[i].set_title(f"{plot_lists[i]}");