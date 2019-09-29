#%% [markdown]
#  #  Portflio Comparrison Challenge
# 
# In this challenge, I evaluate the performance among various algorithmic, hedge, and mutual fund portfolios and compare them against the S&P 500 over a 6 year period.
# 
# Then, I create a portoflio of my own consisting of four stocks picked at random. Cody, CMG, NTFLX & EL. At compare it to the algorithim, hedge and mutual fund portfolios. 
# 

#%%
get_ipython().system('pip install yfinance')
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
get_ipython().run_line_magic('matplotlib', 'inline')
import yfinance as yf 
import seaborn as sns

#%% [markdown]
# ## Whale Returns
# 
# Read the Whale Portfolio daily returns and clean the data

#%%
# Reading whale returns
whale_returns_csv = Path("whale_returns.csv")
whale_returns_df = pd.read_csv(whale_returns_csv, index_col="Date", infer_datetime_format=True, parse_dates=True)
whale_returns_df = whale_returns_df.sort_index(ascending= True)


#%%
# Count nulls
whale_returns_df.isnull().mean() * 100


#%%
# Drop nulls
whale_returns_df = whale_returns_df.dropna()
whale_returns_df.isnull().sum()

#%% [markdown]
# ## Algorithmic Daily Returns
# 
# Read the algorithmic daily returns and clean the data

#%%
# Reading algorithmic returns
algo_returns_csv = Path("algo_returns.csv")
algo_returns_df = pd.read_csv(algo_returns_csv, index_col="Date", infer_datetime_format=True, parse_dates=True)
algo_returns_df = algo_returns_df.sort_index(ascending=True)


#%%
# Count nulls
algo_returns_df.isnull().mean() * 100


#%%
# Drop nulls
algo_returns_df = algo_returns_df.dropna()
algo_returns_df.isnull().sum()


#%%
# Reading S&P 500 Closing Prices
sp500_history_csv = Path("sp500_history.csv")
sp500_history_df = pd.read_csv(sp500_history_csv, index_col="Date", infer_datetime_format=True, parse_dates=True)


#%%
# Check Data Types
sp500_history_df.dtypes


#%%
# Fix Data Types
sp500_history_df['Close'] = sp500_history_df['Close'].str.replace('$', '')
sp500_history_df['Close'] = sp500_history_df['Close'].astype('float')


#%%
# Calculate Daily Returns
SP_daily_returns = sp500_history_df.pct_change()


#%%
# Drop nulls
SP_daily_returns = SP_daily_returns.dropna()


#%%
# Rename Column
SP_daily_returns.rename(columns = {'Close':'S&P 500'}, inplace = True) 
SP_daily_returns = SP_daily_returns.sort_index(ascending=True)


#%%
# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat([whale_returns_df,algo_returns_df,SP_daily_returns], axis="columns", join="inner")
combined_df.head()

#%% [markdown]
# ---
#%% [markdown]
# # Portfolio Analysis
# 
# In this section, you will calculate and visualize performance and risk metrics for the portfolios.
# 
# From the below charts I make the following conculsionsL
# 1. Algo 1 has performed better than the other portfolios, when viewing their cumulative returns and outperforms the S&P 500.
# 2. BERKSHIRE HATHAWAY INC is the riskiest of portfolios from the analysis. 
# 3. SOROS FUND MANAGEMENT LLC closely mimics the S&P 500. 
# 4. BERKSHIRE HATHAWAY INC does not seem to move with the S&P 500, and is negatively correlated to the S&P 500 -0.013856.
# 5. The algorithmic strategies do not outperform booth the market and the whale portfolios according to the Sharpe ratio plot. 
# 
#%% [markdown]
# ## Performance
# 
# Calculate and Plot the daily returns and cumulative returns. 

#%%



#%%
combined_df.plot(figsize=(20,10),title=('Daily Returns'))


#%%
# Plot cumulative returns
cumulative_returns = (1 + combined_df).cumprod()
cumulative_returns.plot(figsize=(20,10),title=('Cumulative Returns'))

#%% [markdown]
# ---
#%% [markdown]
# ## Risk
# 
# Determine the _risk_ of each portfolio:
# 
# 1. Create a box plot for each portfolio. 
# 2. Calculate the standard deviation for all portfolios
# 4. Determine which portfolios are riskier than the S&P 500
# 5. Calculate the Annualized Standard Deviation

#%%
combined_df.plot.box(figsize=(20,10),title=('Box Plot'))


#%%
# Daily Standard Deviations
# Calculate the standard deviation for each portfolio. Which portfolios are riskier than the S&P 500?
daily_std = combined_df.std()
daily_std.head(7)


#%%
# Determine which portfolios are riskier than the S&P 500
daily_std = daily_std.sort_values(ascending=False)
daily_std.head(7)


#%%
# Calculate the annualized standard deviation (252 trading days)
annualized_std = daily_std * np.sqrt(252)
annualized_std.head(7)

#%% [markdown]
# ---
#%% [markdown]
# ## Rolling Statistics
# 
# Risk changes over time. Analyze the rolling statistics for Risk and Beta. 
# 
# 1. Calculate and plot the rolling standard deviation for the S&PP 500 using a 21 day window
# 2. Calcualte the correlation between each stock to determine which portfolios may mimick the S&P 500
# 2. Calculate and plot a 60 day Beta for Berkshire Hathaway Inc compared to the S&&P 500

#%%
# Calculate and plot the rolling standard deviation for the S&PP 500 using a 21 day window
combined_df.rolling(window=21).std().plot(figsize=(20,10),title=('21 day Rolling standard deviation for the S&P 500'))


#%%
# Correlation
price_correlation = combined_df.corr()
price_correlation


#%%
rolling_covariance = combined_df['BERKSHIRE HATHAWAY INC'].rolling(window=60).cov(combined_df['S&P 500'])
rolling_covariance.plot(figsize=(20, 10), title='60 Day Covariance BERKSHIRE HATHAWAY INC')


#%%
rolling_variance = combined_df['S&P 500'].rolling(window=60).var()
rolling_variance.plot(figsize=(20, 10), title='Rolling 60-Day Variance of S&P 500 Returns')


#%%



#%%
# Calculate Beta for a single portfolio compared to the total market (S&P 500)
rolling_beta = rolling_covariance / rolling_variance
rolling_beta.plot(figsize=(20, 10), title='Rolling 60-Day Beta of BERKSHIRE HATHAWAY INC')

#%% [markdown]
# ### Challenge: Exponentially Weighted Average 
# 
# An alternative way to calculate a rollwing window is to take the exponentially weighted moving average. This is like a moving window average, but it assigns greater importance to more recent observations.

#%%
#EWA for all portfolios
ewm_combined_df = combined_df.ewm(halflife = 21).std()
ewm_combined_df.plot(figsize=(20, 10), title='Exponentially Weighted Average')


#%%
#EWA for Bershkire Hathaway
ewm_combined_df = combined_df['BERKSHIRE HATHAWAY INC'].ewm(halflife = 21).mean()
ewm_combined_df.plot(figsize=(20, 10), title='Exponentially Weighted Average')

#%% [markdown]
# ---
#%% [markdown]
# ## Sharpe Ratios
# 

#%%
# Annualzied Sharpe Ratios
all_portfolio_std = combined_df.std()
all_portfolio_std.head(7)


#%%
sharpe_ratios = (combined_df.mean() * 252) / (all_portfolio_std * np.sqrt(252))
sharpe_ratios.head(7)

#%% [markdown]
#  plot() these sharpe ratios using a barplot.
#  On the basis of this performance metric, do our algo strategies outperform both 'the market' and the whales?

#%%
# Visualize the sharpe ratios as a bar plot
sharpe_ratios.plot.bar(title='Sharpe Ratios')

#%% [markdown]
# ---
#%% [markdown]
# # Portfolio Returns
# 
# In this section, I build my own portfolio of stocks, calculate the returns, and compare the results to the Whale Portfolios and the S&P 500. 
# 
# I find: 
# 1. EL and NTFLX have a corelation of .9
# 2. My portfolio does not beat the market or the Aglo portfolios. 
# 3. My portfolio is negatively corelated to the S&P 500 with the largest STD. 
#%% [markdown]
# ## Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.

#%%
Coty_data = yf.download('COTY','2015-01-01','2019-01-01')
Coty_data.drop(['Open','High', 'Low', 'Adj Close','Volume'], axis = 1, inplace=True)
Coty_data.columns = ['COTY']
Coty_data.head(25)


#%%
cmg_data = yf.download('CMG','2015-01-01','2019-01-01',index_col="Date", infer_datetime_format=True, parse_dates=True)
cmg_data.drop(['Open','High', 'Low', 'Adj Close','Volume'], axis = 1, inplace=True)
cmg_data.columns = ['CMG']
cmg_data.head()


#%%



#%%
nflx_data = yf.download('NFLX','2015-01-01','2019-01-01')
nflx_data.drop(['Open','High', 'Low', 'Adj Close','Volume'], axis = 1, inplace=True)
nflx_data.columns = ['NFLX']
nflx_data.head()


#%%
el_data = yf.download('EL','2015-01-01','2019-01-01')
el_data.drop(['Open','High', 'Low', 'Adj Close','Volume'], axis = 1, inplace=True)
el_data.columns = ['EL']
el_data.head()


#%%
# Concatenate all stocks into a single DataFrame
MyPortfolio_df = pd.concat([Coty_data, cmg_data, nflx_data, el_data], axis="columns", join="inner")
MyPortfolio_df.head()


#%%
# Drop Nulls
MyPortfolio_df.dropna()

#%% [markdown]
# ## Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock

#%%
# Calculate weighted portfolio returns
weights = [1/3, 1/3, 1/3, 1/3]
MP_daily_returns = MyPortfolio_df.pct_change()
MP_daily_returns = MP_daily_returns.dot(weights).dropna()
MP_daily_returns.head()

#%% [markdown]
# ## Join your portfolio returns to the DataFrame that contains all of the portfolio returns

#%%
mp_correlation = MyPortfolio_df.corr()
mp_correlation


#%%
MP_Whale_combined_df = pd.concat([MP_daily_returns, combined_df], axis="columns", join="inner")
MP_Whale_combined_df.sort_index(inplace=True)
MP_Whale_combined_df.columns = ['ALEXIS S LLC', 'SOROS FUND MANAGEMENT LLC','PAULSON & CO.INC.', 'TIGER GLOBAL MANAGEMENT LLC','BERKSHIRE HATHAWAY INC','Algo 1','Algo 2','S&P 500']
MP_Whale_combined_df.head()


#%%
# Only compare dates where the new, custom portfolio has dates
MP_Whale_combined_df.dropna(inplace=True)

#%% [markdown]
# ## Re-run the performance and risk analysis with your portfolio to see how it compares to the others

#%%
MP_Whale_combined_df.plot(figsize=(20,10),title=('Combined Daily Returns'))


#%%
# Plot cumulative returns
MP_Whale_cumulative_returns = (1 + MP_Whale_combined_df).cumprod()
MP_Whale_cumulative_returns.plot(figsize=(20,10),title=('Combined Cumulative Returns'))


#%%
# Risk Assesment
MP_Whale_combined_df.plot.box(figsize=(20,10),title=('ALEXIS S LLC Box Plot'))


#%%
#Standard Deviation
MP_Whale_combined_std = MP_Whale_combined_df.std()
MP_Whale_combined_std.head()


#%%
#Sort STD
MP_Whale_daily_std = MP_Whale_combined_std.sort_values(ascending=False)
MP_Whale_daily_std.head()


#%%
#Analayse
mp_Whale_annualized_std = MP_Whale_combined_std * np.sqrt(252)
mp_Whale_annualized_std.head(8)


#%%
# Rolling Data 60 days 
MP_Whale_combined_60 = MP_Whale_combined_df.rolling(window=60).mean().plot(figsize=(20,10),title=('60 day Rolling standard deviation Alexis S'))


#%%
# Betaprice - Correlation
mp_correlation = MP_Whale_combined_df.corr()
mp_correlation


#%%
#rolling covarriance
mp_rolling_covariance = MP_Whale_combined_df['ALEXIS S LLC'].rolling(window=60).cov(combined_df['S&P 500'])


#%%
#rolling variance
mp_rolling_variance = MP_Whale_combined_df['S&P 500'].rolling(window=60).var()
mp_rolling_variance.plot(figsize=(20, 10), title='Rolling 60-Day Variance of S&P 500 Returns')


#%%
# Calculate Beta for my single portfolio compared to the total market (S&P 500)
mp_rolling_beta = mp_rolling_covariance / mp_rolling_variance
mp_rolling_beta.plot(figsize=(20, 10), title=' Beta of ALEXIS S LLC')


#%%
# Annualzied Sharpe Ratios
mp_sharpe_ratios = (MP_Whale_combined_df.mean() * 252) / (MP_Whale_combined_std * np.sqrt(252))
mp_sharpe_ratios.head()


#%%
# Visualize the sharpe ratios as a bar plot
mp_sharpe_ratios.plot.bar(title='ALEXIS LLC Sharpe Ratios')

#%% [markdown]
# ## Correlation Analysis to determine which stocks (if any) are correlated

#%%
#Heat map of correlation 
sns.heatmap(mp_correlation, vmin=-1, vmax=1,)


