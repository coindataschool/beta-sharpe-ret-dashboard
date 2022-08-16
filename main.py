import datetime as dt
import pandas as pd
import pandas_datareader.data as reader
import streamlit as st
from duneanalytics import DuneAnalytics
from helper import extract_frame_from_dune_data, calc_beta, annualize_tot_ret

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
df_glp_prices = (extract_frame_from_dune_data(glp_arbi_prices, 'date')
    .rename({'price':'GLP'}, axis=1))
df_tri_prices = (extract_frame_from_dune_data(tricrypto_prices, 'date')
    .rename({'price':'TriCrypto'}, axis=1))
# TriCrypto price became available on 2021-06-09 and GLP on 2021-08-31. 
# let's cut TriCrypto's price data using 2021-08-31. This will ensure the 
# monthly returns to be calculated over the same months.
df_tri_prices = df_tri_prices.loc[df_glp_prices.index[0]:, :]

# download daily prices from Yahoo
# we want to use the start date of the asset with the least amount of history
# as the start date of the period we want to download data for all assets. 
# This saves time.
start = dt.date(2021, 8, 31) # GLP price has the youngest history and it 
    # first became available on 2021-08-31.
today = dt.datetime.now(tz=dt.timezone.utc)
end = dt.date(today.year, today.month, 1)
tickers_names = {
    '^GSPC': 'SP500',
    'VNQ': 'Real Estate',           
    'TIP': 'Inflation-Linked Bonds',   
    'BND': 'Nominal Bonds', 
    'GLD': 'Gold',
    '^SPGSCI': 'Broad Commodities',
    'BTC-USD':'BTC', 
    'ETH-USD':'ETH'
}
tickers = list(tickers_names.keys())
# yahoo price reader downloads prices since `start` (including `start`) when 
# running on streamlit cloud. But when running on my local machine, it also
# downloads prices on the day before `start`. I guess it has to do 
# with my timezone and local time?
df_prices = (reader.get_data_yahoo(tickers, start, end)['Adj Close']
                .rename(tickers_names, axis=1))
df_prices.columns.name = None

# drop the last row since end date is the first day of the current month, 
# keeping it will result a fake current month return
df_prices = df_prices.iloc[:-1]

# download monthly risk free rates 
# these rates are already multiplied by 100, so we divide them by 100 to 
# make them on the same scale as the returns we will calculate. 
rfs = reader.DataReader('F-F_Research_Data_Factors', 'famafrench', start, end)[0].RF / 100

# calculate monthly returns 
# because `df_prices`` includes price for the day before `start`, we use 
# `last()` to calculate the monthly returns. 
monthly_rets = df_prices.resample('M').last().pct_change()
monthly_rets_glp = df_glp_prices.resample('M').last().pct_change()
monthly_rets_tri = df_tri_prices.resample('M').last().pct_change()
monthly_rets = monthly_rets.join(monthly_rets_glp).join(monthly_rets_tri)

# convert index to monthly period so that we can join with the risk free rates
monthly_rets = monthly_rets.to_period('M')
monthly_rets = monthly_rets.join(rfs)

# calculate monthly excess returns
for col in monthly_rets.columns.drop('RF'):
    newcol = col + ' - ' + 'RF'
    monthly_rets[newcol] = monthly_rets[col] - monthly_rets['RF']
# ensure all assets have the same months for fair comparison.  
excess_monthly_rets = monthly_rets.dropna().loc[:, monthly_rets.columns.str.endswith('- RF')]
# remove ' - RF' from the column names for better display
excess_monthly_rets.columns = excess_monthly_rets.columns.str.replace(' - RF', '')

# Calculate Beta, Sharpe Ratio, and Excess Return (Ann) using Excess Monthly Returns
#   - Treat SP500 as benchmark
#   - GLP and TriCrypto Yields are excluded
market = 'SP500'
tokens = excess_monthly_rets.columns

betas = []
pvals = []
r2s = []
for token in tokens:
    res = calc_beta(excess_monthly_rets, token, market)
    betas.append(res['beta'])
    pvals.append(res['p-val'])
    r2s.append(res['R2'])
df_betas = pd.Series(betas, index=tokens).sort_values().to_frame().rename({0:'Beta'}, axis=1)
df_pvals = pd.Series(pvals, index=tokens).sort_values().to_frame().rename({0:'p-Value'}, axis=1)
df_r2s = pd.Series(r2s, index=tokens).sort_values().to_frame().rename({0:'R2'}, axis=1)
df_betas = df_betas.join(df_pvals).join(df_r2s)    

sharpe_ratios = excess_monthly_rets.mean() / excess_monthly_rets.std()
df_sharpes = sharpe_ratios.sort_values(ascending=False).to_frame().rename({0:'Sharpe Ratio'}, axis=1)

tot_ret = (1+excess_monthly_rets).prod()-1
dur_years = len(excess_monthly_rets) / 12
ann_excess_rets = annualize_tot_ret(tot_ret, dur_years) * 100
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

fst_ret_yrmon = excess_monthly_rets.index.min()
lst_ret_yrmon = excess_monthly_rets.index.max()
msg = ('Monthly returns data from {} {} through {} {}. GLP and TriCrypto yields are excluded. The lower the p-Value and the bigger the R2, the more reliable the Beta.'
        .format(fst_ret_yrmon.strftime('%B'), fst_ret_yrmon.year, 
                lst_ret_yrmon.strftime('%B'), lst_ret_yrmon.year))
st.caption(msg)

st.write("Beta is a multiplier. For example, a beta of 0.9 against SP500 indicates that we can expect the asset's excess return to increase/decrease by 0.9% for every 1% increase/decrease in SP500's excess return.")
st.write("Sharpe ratio is volatility adjusted return in expectation.")
msg1 = "R2 ranges between 0 and 1. An R2 of 1 indicates all of the asset's fluctuations are explained by SP500's fluctuations, while an R2 of 0 indicates no correlation. "
msg2 = "A tiny R2 indicates the Beta is not reliable. We can expect p-Value and R2 to improve as time passes and more data become available."
st.write(msg1 + msg2)
st.write("You can read my [article](https://coindataschool.substack.com/p/beta-sharpe-ratio-excess-return?sd=pf) for more explanations.")

st.subheader('About')
st.markdown("Check out my Dune dashboards: [@coindataschool](https://dune.com/coindataschool)")
st.write("Follow me on twitter: [@coindataschool](https://twitter.com/coindataschool)")
st.markdown("Buy me a coffee with eth coins: `0x783c5546c863f65481bd05fd0e3fd5f26724604e`")
st.markdown("[Tip me sat](https://tippin.me/@coindataschool)")
 



