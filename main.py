import datetime as dt
import pandas as pd
import pandas_datareader.data as reader
import streamlit as st
from duneanalytics import DuneAnalytics
from helper import extract_frame_from_dune_data, calc_beta, annualize_tot_ret #, row_style

# set_page_config() can only be called once per app, and must be called as the 
# first Streamlit command in your script.
st.set_page_config(page_title='Beta, Sharpe Ratio, and Excess Return', layout='wide', page_icon='🔔') 

# access dune
dune = DuneAnalytics(st.secrets["DUNE_USERNAME"], st.secrets["DUNE_PASSWORD"])
dune.login()
dune.fetch_auth_token()

# query daily prices for GLP and TriCrypto
glp_arbi_prices = dune.query_result(dune.query_result_id(query_id=1069389))
tricrypto_prices = dune.query_result(dune.query_result_id(query_id=1145739))
df_glp_arbi_prices = (extract_frame_from_dune_data(glp_arbi_prices, 'date')
    .rename({'price':'GLP'}, axis=1))
df_tricrypto_prices = (extract_frame_from_dune_data(tricrypto_prices, 'date')
    .rename({'price':'TriCrypto'}, axis=1))

# download daily prices for SP500, Tips, Bond, Gold, Reit, BTC, and ETH from Yahoo
start = dt.date(2021, 8, 31) # when GLP price first became available
    # TriCrypto price became available on 2021-06-09, earlier than GLP
end = dt.datetime.now()
tickers = ['^GSPC', 'TIP', 'BND', 'VNQ', 'GLD', 'BTC-USD', 'ETH-USD']
df_prices = (
    reader.get_data_yahoo(tickers, start, end)['Adj Close']
            .rename({'^GSPC':'SP500', 'TIP':'Inflation-Linked Bonds', 
                    'BND':'Nominal Bonds', 'VNQ':'Real Estate', 'GLD':'Gold', 
                    'BTC-USD':'BTC', 'ETH-USD':'ETH'}, axis=1))
df_prices.columns.name = None

# download monthly risk free rates 
rfs = reader.DataReader('F-F_Research_Data_Factors', 'famafrench', start, end)[0].RF
# convert Period index to datetime
rfs.index = pd.to_datetime(rfs.index.to_timestamp(how='end').strftime('%Y-%m-%d'))

# calculate daily, weekly, monthly returns for prices downloaded from yahoo
daily_rets = dict()
monthly_rets = dict()
weekly_rets = dict()
for col in df_prices.columns:
    daily_rets_ticker = df_prices[col].pct_change().dropna()
    monthly_rets_ticker = daily_rets_ticker.resample('M').agg(lambda x: (1+x).prod()-1).iloc[1:-1] # drop 1st and last row since they may not be a full month
    weekly_rets_ticker = daily_rets_ticker.resample('W').agg(lambda x: (1+x).prod()-1).iloc[1:-1] # drop 1st and last row since they may not be a full week
    # collect results
    daily_rets[col] = daily_rets_ticker
    monthly_rets[col] = monthly_rets_ticker
    weekly_rets[col] = weekly_rets_ticker
daily_rets = pd.DataFrame(daily_rets)    
monthly_rets = pd.DataFrame(monthly_rets)
weekly_rets = pd.DataFrame(weekly_rets)

# calculate daily, weekly, monthly returns for GLP
daily_rets_glp = df_glp_arbi_prices.GLP.pct_change().dropna()
monthly_rets_glp = daily_rets_glp.resample('M').agg(lambda x: (1+x).prod()-1).iloc[:-1] # drop last row since it may not be a full month, do not drop 1st row since it's a full month for GLP
weekly_rets_glp = daily_rets_glp.resample('W').agg(lambda x: (1+x).prod()-1).iloc[1:-1] # drop first and last rows since they may not be a full week

# calculate daily, weekly, monthly returns for TriCrypto
daily_rets_tri = df_tricrypto_prices.TriCrypto.pct_change().dropna()
monthly_rets_tri = daily_rets_tri.resample('M').agg(lambda x: (1+x).prod()-1).iloc[1:-1] # drop 1st and last row since they may not be a full month
weekly_rets_tri = daily_rets_tri.resample('W').agg(lambda x: (1+x).prod()-1).iloc[1:-1] # drop first and last rows since they may not be a full week

# join all returns
daily_rets = daily_rets.join(daily_rets_glp).join(daily_rets_tri)
monthly_rets = monthly_rets.join(monthly_rets_glp).join(monthly_rets_tri)
weekly_rets = weekly_rets.join(weekly_rets_glp).join(weekly_rets_tri)

# calculate monthly excess returns
monthly_rets = monthly_rets.join(rfs)
for col in monthly_rets.columns.drop('RF'):
    newcol = col + ' - ' + 'RF'
    monthly_rets[newcol] = monthly_rets[col] - monthly_rets['RF']
# for a fair comparison, we want to ensure all assets have the same months. GLP and TriCrypto have the least 
# amount of history. It's misleading to compare, for example, BTC's beta calculated using more historical months with 
# GLP or TriCrypto's beta calculated using fewer months. 
excess_monthly_rets = monthly_rets.dropna().loc[:, monthly_rets.columns.str.endswith('- RF')]
# remove ' - RF' from the column names for better display
excess_monthly_rets.columns = excess_monthly_rets.columns.str.replace(' - RF', '')

# Calculate Beta, Sharpe Ratio, and Excess Return (Ann) using Excess Monthly Returns
#   - Treat SP500 as benchmark
#   - GLP and TriCrypto Yields are excluded
market = 'SP500'
tokens = excess_monthly_rets.columns.drop(market)
betas = [calc_beta(excess_monthly_rets, token, market).round(3) for token in tokens]
df_betas = pd.Series(betas, index=tokens).sort_values().to_frame().rename({0:'Beta'}, axis=1)

sharpe_ratios = (excess_monthly_rets.mean() / excess_monthly_rets.std()).round(3)
df_sharpes = sharpe_ratios.sort_values(ascending=False).to_frame().rename({0:'Sharpe Ratio'}, axis=1)

tot_ret = (1+excess_monthly_rets).prod()-1
dur_years = len(excess_monthly_rets) / 12
ann_excess_rets = annualize_tot_ret(tot_ret, dur_years).round(3) * 100
df_ann_excess_rets = ann_excess_rets.sort_values(ascending=False).to_frame().rename({0:'Excess Return (Ann)'}, axis=1)

# display tables
col1, col2, col3 = st.columns(3)
with col1:
    st.header('Sharpe Ratio')
    st.dataframe(df_sharpes.style.format(precision=3))
with col2:
    st.header('Excess Return (Ann)')
    st.dataframe(df_ann_excess_rets.style.format({'Excess Return (Ann)': '{:,.1f}%'.format}))
with col3:
    st.header('Beta (vs. SP500)')
    st.dataframe(df_betas.style.format(precision=3))

msg = ('Data from {} {} {} through {} {} {}. GLP and TriCrypto yields are excluded.'
        .format(start.day, start.strftime('%B'), start.year, end.day, end.strftime('%B'), end.year))
st.caption(msg)

msg1 = "It's best to view Beta a multiplier. For example, if an asset has beta 0.9 against SP500, it means if SP500 return increases by 1%, we can expect the asset return increase by 0.9%."
msg2 = "On the other hand, if SP500 return decreases by 1%, we can expect the asset return decrease by 0.9%."
st.write(msg1 + '\n' + msg2)
st.write("Read my [article]() on how to use this dashboard.")

st.subheader('About')
st.markdown("Check out my Dune dashboards: [@coindataschool](https://dune.com/coindataschool)")
st.write("Follow me on twitter: [@coindataschool](https://twitter.com/coindataschool)")
st.write("Buy me a coffee: 0x783c5546c863f65481bd05fd0e3fd5f26724604e")
