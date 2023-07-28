
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
from scipy.stats import norm
import requests
from io import StringIO
import seaborn as sns; sns.set()
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10,6)

def getDailyData(symbol):
    parameters = {'function': 'TIME_SERIES_DAILY_ADJUSTED',
                  'symbol': symbol,
                  'outputsize': 'full',
                  'datatype': 'csv',
                  'apikey': 'UWALAP8D7WX2JEGO'}

    response = requests.get('https://www.alphavantage.co/query',
                            params=parameters)

    csvText = StringIO(response.text)
    data = pd.read_csv(csvText, index_col='timestamp')
    return data

def VaR_parametric(initial_investment, conf_level):
    alpha = norm.ppf(1 - conf_level, stocks_returns_mean, port_std)
    for i, j in zip(stocks.columns, range(len(stocks.columns))):
        VaR_param = (initial_investment - initial_investment *
                     (1 + alpha))[j]
        print("Parametric VaR result for {} is {} "
              .format(i, VaR_param))
    VaR_param = (initial_investment - initial_investment * (1 + alpha))
    print('--' * 25)
    return VaR_param


def VaR_historical(initial_investment, conf_level):
    Hist_percentile95 = []
    for i, j in zip(stocks_returns.columns,
                    range(len(stocks_returns.columns))):
        Hist_percentile95.append(np.percentile(stocks_returns.loc[:, i],
                                               5))
        print("Based on historical values 95% of {}'s return is {:.4f}"
              .format(i, Hist_percentile95[j]))
        VaR_historical = (initial_investment - initial_investment *
                          (1 + Hist_percentile95[j]))
        print("Historical VaR result for {} is {:.2f} "
              .format(i, VaR_historical))
        print('--' * 35)


def MC_VaR(initial_investment, conf_level):
    MC_percentile95 = []
    for i, j in zip(sim_data.columns, range(len(sim_data.columns))):
        MC_percentile95.append(np.percentile(sim_data.loc[:, i], 5))
        print("Based on simulation 95% of {}'s return is {:.4f}"
              .format(i, MC_percentile95[j]))
        VaR_MC = (initial_investment - initial_investment *
                  (1 + MC_percentile95[j]))
        print("Simulation VaR result for {} is {:.2f} "
              .format(i, VaR_MC))
        print('--' * 35)


def VaR_parametric_denoised(initial_investment, conf_level):
    port_std = np.sqrt(weights.T.dot(cov_matrix_denoised)
                       .dot(weights))
    4
    alpha = norm.ppf(1 - conf_level, stocks_returns_mean, port_std)
    for i, j in zip(stocks.columns, range(len(stocks.columns))):
        print("Parametric VaR result for {} is {} ".format(i, VaR_param))
    VaR_params = (initial_investment - initial_investment * (1 + alpha))
    print('--' * 25)
    return VaR_params

def ES_parametric(initial_investment, conf_level):
    alpha = - norm.ppf(1 - conf_level, stocks_returns_mean, port_std)
    for i, j in zip(stocks.columns, range(len(stocks.columns))):
        VaR_param = (initial_investment * alpha)[j]
        ES_param = (1 / (1 - conf_level)) \
                   * initial_investment \
                   * norm.expect(lambda x: x,
                                 lb=norm.ppf(conf_level,
                                             stocks_returns_mean[j],
                                             port_std),
                                 loc=stocks_returns_mean[j],
                                 scale=port_std)
        print(f"Parametric ES result for {i} is {ES_param}")


def ES_historical(initial_investment, conf_level):
    for i, j in zip(stocks_returns.columns,
                    range(len(stocks_returns.columns))):
        ES_hist_percentile95 = np.percentile(stocks_returns.loc[:, i],
                                             5)
        ES_historical = stocks_returns[str(i)][stocks_returns[str(i)] <=
                                               ES_hist_percentile95] \
            .mean()
        print("Historical ES result for {} is {:.4f} "
              .format(i, initial_investment * ES_historical))

if __name__ == "__main__":
    symbols = ["IBM", "MSFT", "INTC"]
    stock3 = []
    for symbol in symbols:
        stock3.append(getDailyData(symbol)[::-1]['close']
                      ['2020-01-01': '2020-12-31'])
    stocks = pd.DataFrame(stock3).T
    stocks.columns = symbols
    stocks.head()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
spread_meas_scaled = scaler.fit_transform(np.abs(spread_meas))
pca = PCA(n_components=5)
prin_comp = pca.fit_transform(spread_meas_scaled)

var_expl = np.round(pca.explained_variance_ratio_, decimals=4)

cum_var = np.cumsum(np.round(pca.explained_variance_ratio_,
                             decimals=4))

print('Individually Explained Variances are:\n{}'.format(var_expl))
print('==' * 30)
print('Cumulative Explained Variances are: {}'.format(cum_var))


plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Number of Components')
plt.ylabel('Variance Explained')
plt.title('Scree Plot')
plt.show()
pca = PCA(n_components=2)
pca.fit(np.abs(spread_meas_scaled))
prin_comp = pca.transform(np.abs(spread_meas_scaled))
prin_comp = pd.DataFrame(np.abs(prin_comp), columns=['Component 1',
                                                     'Component 2'])
print(pca.explained_variance_ratio_ * 100)


def myplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex * 4, ys * scaley * 4, s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r',
                  alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0], coeff[i, 1], "Var" + str(i),
                     color='black')
        else:
            plt.text(coeff[i, 0], coeff[i, 1], labels[i],
                     color='black')

    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()


spread_measures_scaled_df = pd.DataFrame(spread_meas_scaled,
                                                 columns=spread_meas.columns)

myplot(np.array(spread_measures_scaled_df)[:, 0:2],
               np.transpose(pca.components_[0:2, :]),
               list(spread_measures_scaled_df.columns))
plt.show()