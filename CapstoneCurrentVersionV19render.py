#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import dash
from dash import Dash, dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Hard-code early BTC data that is not widely available; see comments on each price 
early_BTC_data = {2009: {'Open':0.01,'Close':0.01,'Avg Price':0.01,'Low':0.01,'High':0.01}, # Arbitrary value using a penny as the price since BTC was basically unpriced/miniscule in price at this time
                  2010: {'Open':0.18,'Close':0.18,'Avg Price':0.18,'Low':0.18,'High':0.18}, # Arbitrary value using the average of the first price (0.05) and last price (0.30) of 2010
                  2011: {'Open':2.28,'Close':2.28,'Avg Price':2.28,'Low':2.28,'High':2.28}, # Arbitrary value using the average of the first price (0.30) and last price (4.25) of 2011
                  2012: {'Open':9.09,'Close':9.09,'Avg Price':9.09,'Low':9.09,'High':9.09}, # Arbitrary value using the average of the first price (4.72) and last price (13.45) of 2012
                  2013: {'Open':383.76,'Close':383.76,'Avg Price':383.76,'Low':383.76,'High':383.76}, # Arbitrary value using the average of the first price (13.51) and last price (754.01) of 2013
                  2014: {'Open':618.73,'Close':618.73,'Avg Price':618.73,'Low':618.73,'High':618.73}, # Arbitrary value using the average of the first price of 2014 (771.40) and September 16, 2014 price (466.06), which is the first day that yfinance does not have (754.01)
                 }
# Define the start and end dates for hard-to-get BTC prices
start_date = datetime(2009, 1, 3)
end_date = datetime(2014, 9, 16)

# Build the hard-to-get BTC price dataframe
records = []
for date in pd.date_range(start=start_date, end=end_date):
    # Grab the year of the current date to properly query the early_BTC_data dict
    year = date.year
    records.append({'Date':date.strftime('%Y-%m-%d %H:%M:%S'),
                    'Open':early_BTC_data[year]['Open'],
                    'Close':early_BTC_data[year]['Close'],
                    'Avg Price':early_BTC_data[year]['Avg Price'],
                    'Low':early_BTC_data[year]['Low'],
                    'High':early_BTC_data[year]['High']})
new_records = pd.DataFrame(records)  

# Retrieve all Bitcoin prices on Yahoo finance, beginning on September 17th, 2014
BTC_Ticker = yf.Ticker('BTC-USD')
BTC_df = BTC_Ticker.history(period='max')

# Reset the index, which is the date data by default; adding the "Date" column and data to the DataFrame
BTC_df.reset_index(inplace=True)
BTC_df.rename(columns={'index': 'Date'}, inplace=True)

# Format the BTC DataFrame; add the average price column, which is used as the BTC price for an entire day in this project
BTC_df['Date'] = BTC_df['Date'].dt.strftime('%Y-%m-%d')
BTC_df.drop(['Stock Splits','Dividends','Volume'],inplace=True,axis=1)

# Join the pricing dataframes
BTC_df = pd.concat([new_records,BTC_df], ignore_index=True, axis = 0)
BTC_df = BTC_df.reset_index(drop=True)

# Calculate Avg Price with the following formula; use this as the price for the entire date provided in the same entry
BTC_df['Avg Price'] = round(((BTC_df['Open'] + BTC_df['Close']) / 2),2)

# Cast the date column to a datetime date type
BTC_df['Date'] = pd.to_datetime(BTC_df['Date'])

# Strip the timestamp (time portion) from the date column of the new_records dataframe that was added 
BTC_df['Date'] = BTC_df['Date'].dt.date
lowest_date = BTC_df['Date'].min()
highest_date = BTC_df['Date'].max()

# Find any missing dates; loop through the missing dates and set the missing date equal to the previous row
# The yfinance library sometimes skips dates, and it seems to remedy itself a few days later
# The gaps must be filled in with some sort of data to avoid errors
date_range = pd.date_range(lowest_date,highest_date)
missing_dates = date_range.difference(BTC_df['Date'])
for date in missing_dates:
    date = date.date()
    prev_row_index = BTC_df[BTC_df['Date'] < date].index[-1]
    prev_row_data = BTC_df.loc[prev_row_index]
    new_row = prev_row_data.copy()
    new_row['Date'] = date
    BTC_df = pd.concat([BTC_df.loc[:prev_row_index], new_row.to_frame().T, BTC_df.loc[prev_row_index+1:]], ignore_index=True)
BTC_df.reset_index(inplace=True)

# Cast the date column to a string for proper comparison with the Blockstream API's date data
BTC_df['Date'] = BTC_df['Date'].astype(str)
BTC_df.drop(columns=['index'], inplace=True)

# The Dash documentation says not to use global variables if they are going to be manipulated by user action
# If multiple sessions of the application are running, the global variable may store data from both, causing issues
# Instead, using a class seems to be the best way around this
class SessionState:
    def __init__(self):
        self.raw_all_wallet_df = pd.DataFrame(columns=['Date', 'Amount', 'Low', 'Avg Price', 'High', 'Wallet'])
        self.filtered_all_wallet_df = pd.DataFrame(columns=['Date', 'Amount', 'Low', 'Avg Price', 'High','Low Value','Avg Value','High Value'])

# Create an instance of the SessionState class
session_state = SessionState()

# Function to validate wallet addresses
# If the address exists and has greater than 0 associated transactions, update Dash displays and add the wallets data to the session_state's wallet dfs
def validate_wallet(bitcoin_address):
    # Check if the wallet address has already been entered
    unique_wallet_addresses = set(session_state.raw_all_wallet_df['Wallet'].unique())
    if bitcoin_address in unique_wallet_addresses:
        return False

    # Blockstream.info API endpoint URL
    url = f"https://blockstream.info/api/address/{bitcoin_address}/txs"

    # Store all the transactions (with metadata) of the wallet; gather all the unique transaction IDs from transactions in the wallet
    transactions = []
    txid_set = set()

    try:
        # Make calls of 25 (the limit of transactions returned) transactions via the Blockstream API until there are no more unique transaction IDs for the given public Bitcoin wallet
        while True:
            response = requests.get(url)
            response.raise_for_status()  # Check for any HTTP errors
            transactions_data = response.json()
            new_tx_count = 0
            last_new_txid = ''
            for transaction in transactions_data:
                txid = transaction['txid']
                if txid not in txid_set:
                    new_tx_count+=1
                    txid_set.add(txid)
                    transactions.append(transaction)
                    last_new_txid = txid

            if new_tx_count==0:
                break

            # Each call to the Blockstream API returns 25 transactions; if the 25th transaction from a call is passed, the next page of transactions is returned
            url = f"https://blockstream.info/api/address/{bitcoin_address}/txs/chain/{last_new_txid}"

    # Mark wallet as invalid if there is an error retrieving the wallet's data from the API
    except requests.exceptions.RequestException as e:
        return False
    except ValueError as ve:
        return False
        
    # Mark wallet as invalid if the wallet has no transactions associated with it
    if len(transactions)==0:
        return False
    
    # Add the wallet's data to both the raw and filtered dataframes in the session_state object; return True as the wallet is valid
    get_wallet_data(bitcoin_address,transactions)
    return True

# Create a dataframe of the transactions and add them to the session_state's raw and filtered dataframes respectively
def get_wallet_data(bitcoin_address,transactions):
    global BTC_df # Dash documentation suggests not using global variables if the variable will be changed by the user; BTC_df will never be changed
    wallet_df = pd.DataFrame(columns=['Date','Low','Avg Price','High','Amount','Wallet'])
    total_balance = 0
    
    # Iterate through the transactions list backwards, which will allow the oldest transactions to be evaluated first
    for transaction in reversed(transactions):
        txid = transaction['txid']
        
        # Retrieve the timestamp of the transaction in UTC
        date = datetime.utcfromtimestamp(transaction['status']['block_time'])

        # Check if the address is in the inputs or outputs to determine sending or receiving
        for input_tx in transaction['vin']:
            if input_tx['prevout']['scriptpubkey_address'] == bitcoin_address:
                total_balance -= input_tx['prevout']['value']
        for output in transaction['vout']:
            if output['value']!=0 and output['scriptpubkey_address'] == bitcoin_address:
                total_balance += output['value']

        total_balance_in_BTC = total_balance/100000000
        
        # Grab the date portion of the datetime timestamp only as a string
        # Find the date in the BTC_df to pull the pricing data for that date
        check_date = str(date)[:10]
        price_entry = BTC_df[BTC_df['Date'] == check_date]
        avg_price = price_entry['Avg Price'].iloc[0] 
        low_price = price_entry['Low'].iloc[0]
        high_price = price_entry['High'].iloc[0]
        
        # Create the new row
        new_row = pd.DataFrame([{'Date':date,'Low':low_price,'Avg Price':avg_price,'High':high_price,'Amount':total_balance_in_BTC,'Wallet':bitcoin_address}])

        # Add the new row to the wallet_df
        wallet_df = pd.concat([wallet_df,new_row],ignore_index=True)

    # Format floats to 2 decimal places in the Value column
    pd.set_option('float_format', '{:f}'.format)

    # Next, the df is normalized to include all dates from Bitcoin's inception to today
    # If the date doesn't exist in the dataframe, an entry for the date is added, and it is set to the previous days latest entry
    # This is needed in order to have Bitcoin portfolio-level holdings
    # If this normalization was not done, dates that appear in one wallet but not the others would show as massive outliers in the time series and inaccurately portray Bitcoin holdings for that day
    # Convert the 'Date' column to datetime objects and define start and end dates for the loop
    wallet_df['Date'] = pd.to_datetime(wallet_df['Date'])
    start_date = datetime(2009, 1, 3) # The date the Bitcoin blockchain went live  
    end_date = datetime.today()

    # Initialize an empty DataFrame to store intermediary results
    result_df = pd.DataFrame(columns=wallet_df.columns)

    # Initialize the previous_entry Series with 0s and the bitcoin_address as values
    previous_entry = pd.Series({'Amount':0,'Low':0,'Avg Price':0,'High':0,'Wallet':bitcoin_address})
    new_entries = []
    
    # Loop through the date range
    while start_date <= end_date:
        date_check = start_date.date()
        # If the date does not exist in the dataframe, add an entry using the previous_entry Series (as described previously)
        if date_check not in wallet_df['Date'].dt.date.values:
            new_entries.append({'Date':start_date,
                                'Amount':previous_entry['Amount'],
                                'Low':BTC_df.loc[BTC_df['Date'] == str(date_check), 'Low'].values[0],
                                'Avg Price':BTC_df.loc[BTC_df['Date'] == str(date_check), 'Avg Price'].values[0],
                                'High':BTC_df.loc[BTC_df['Date'] == str(date_check), 'High'].values[0],
                                'Wallet':previous_entry['Wallet']})   
        # The date exists; update the previous_entry to reflect this date's entry
        else:
            previous_entry = wallet_df.loc[wallet_df['Date'].dt.date == date_check].iloc[-1]
        start_date += timedelta(days=1)

    result_df = pd.DataFrame(new_entries)

    # Concatenate the original DataFrame and the result DataFrame
    wallet_df = pd.concat([wallet_df, result_df],axis=0)

    # Sort the DataFrame by 'Date' in ascending order; after this, the dataframe now contains the original entries and additional entries for missing dates 
    wallet_df = wallet_df.sort_values(by='Date').reset_index(drop=True)

    # The wallet_df now contains the original entries and additional entries for any missing dates in the range of Bitcoin's existence
    # Concat the normalized wallet_df to the session_state's raw_all_wallet_df, which is stored throughout the life of the Dash session 
    session_state.raw_all_wallet_df = pd.concat([session_state.raw_all_wallet_df,wallet_df],axis=0)
    session_state.raw_all_wallet_df = session_state.raw_all_wallet_df.reset_index(drop=True)
    
    # Create a copy of the session_state object's raw_all_wallet_df for use/maniuplation
    all_wallet_df = pd.DataFrame(session_state.raw_all_wallet_df).copy()
    
    # Set the 'Date' column to datetime instead of strings and sort the 'Date" column in descending order
    all_wallet_df['Date'] = pd.to_datetime(all_wallet_df['Date'])
    all_wallet_df = all_wallet_df.sort_values(by='Date', ascending=False)

    # Initialize an empty DataFrame to store the filtering results (latest timestamp entry for each date of each unique wallet)
    latest_entries = pd.DataFrame(columns=all_wallet_df.columns)

    # Iterate through the sorted DataFrame and select the last entry for each combination of 'Wallet' and 'Date'
    for date, group in all_wallet_df.groupby(['Wallet', all_wallet_df['Date'].dt.date]):
        latest_entry = group.head(1)  # Select the first row, which is the latest entry
        latest_entries = pd.concat([latest_entries,latest_entry],axis=0)

    # Reset the index of the resulting DataFrame; this dataframe contains the last entry for each unqiue wallet for each date
    latest_entries = latest_entries.reset_index(drop=True)

    # Convert the date column to datetime objects for sorting / summation
    latest_entries['Date'] = latest_entries['Date'].dt.date
    
    # Sum the BTC "amount" of all dates grouping by date, low price, avg price, and high price
    session_state.filtered_all_wallet_df = latest_entries.groupby(['Date','Low','Avg Price','High'])[['Amount']].sum(numeric_only=True).reset_index()
    
# This function generates the first graph in the Dash application, which shows either portfolio-level and individual-wallet-level data depending on user selections
# This graph can be filtered by time (by clicking buttons on the page like a normal stock chart) or by using the price type dropdown (24hr-average, 24hr-low, 24hr-high)
# This function also generates the 'value' column of the dataframe passed by taking the price column passed, mapping it, and multiplying it by the 'amount' column of the DataFrame
# The current usd value of the portfolio is also returned in this function
def generate_graph(filtered_df_in, price_type, radio_value, button_id=None):
    filtered_df = None 
    if filtered_df_in.empty:
        if radio_value=='individual':
            filtered_df = pd.DataFrame([{'Date':datetime.today().date(),'Amount':0,'Low Value':0,'Avg Value':0,'High Value':0,'Low':0,'Avg Price':0,'High':0,'Wallet':'abcdefghijklmnopqrstuvwxyz'}])
        elif radio_value=='portfolio':
            filtered_df = pd.DataFrame([{'Date':datetime.today().date(),'Amount':0,'Low Value':0,'Avg Value':0,'High Value':0,'Low':0,'Avg Price':0,'High':0}])
    else:
        filtered_df = filtered_df_in.copy()
    price_map = {'24hr-low':'Low', '24hr-average':'Avg Price', '24hr-high':'High'}
    price_col = price_map[price_type]
    filtered_df['Value'] = filtered_df['Amount'] * filtered_df[price_col]
    usd_val = filtered_df['Value'].iloc[-1]
   
    # Specify RGB color to be used in much of the graph
    darker_gold_color = 'rgb(184,134,11)'
    
    # If the "view" radio value in the top right of the application is "individual," display a time series graph with each wallet as a trace
    if radio_value=='individual':
        fig_title = 'USD Value Per Wallet Time Series'
        if button_id!=None:
            button_id = button_id.split('_')[0]
            fig_title += f' {button_id}'
        else:
            filtered_df = session_state.raw_all_wallet_df.copy()
        filtered_df['Value'] = filtered_df['Amount'] * filtered_df[price_col]
        
        # Create a list of 27 unique colors to be used as the trace color when a wallet is added; adding more than 27 wallets will break the application because a color will not be specified
        # More colors can be added, but for the demo of this capstone, more than 27 wallets are not needed to display application functionality
        color_sequence = ['lightgrey','rgb(184,134,11)', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd','#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                          '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5','#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5','#393b79', '#ff9c00', '#5254a3',
                          '#d94801', '#ff5800']
        
        # Create the time series graph with each wallet having a trace
        fig = px.line(filtered_df, x='Date', y='Value', color='Wallet', title=fig_title,hover_data={"Value": ":$.2f", "Date": True},
              category_orders={'Wallet': filtered_df['Wallet'].unique()},
              color_discrete_map={wallet: color_sequence[i % len(color_sequence)] for i, wallet in enumerate(filtered_df['Wallet'].unique())})
        
        # Customize the layout for sleek y-axis ticks
        fig.update_layout(
            legend=dict(
                x=1.1,  # Set the legend's x position to 1.1 (right)
                y=1.2,  # Set the legend's y position to 1.2 (top)
                xanchor='right',  # Specify legend's location
                yanchor='top'
            ),
            yaxis=dict(
                title='USD Value',
                tickmode='linear',
                tick0=filtered_df['Value'].min(),  # Set the lowest tick to the minimum value
                dtick=(filtered_df['Value'].max() - filtered_df['Value'].min()) / 4,  # Calculate tick interval for 5 ticks
                tickformat='$,.0f',  # Format y-axis ticks as currency with 2 decimal places
                showgrid=False,
                gridcolor='lightgray',
            ),
            xaxis=dict(
                hoverformat='%b %d, %Y',
                title='Date',
                showgrid=False
            ),
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color=darker_gold_color)
        )

    # If the "view" radio value in the top right of the application is "portfolio," display a portfolio-level graph with 2 traces: total BTC amount and total USD value
    elif radio_value=='portfolio':
        # Use default title if there is no title passed
        fig_title = 'Bitcoin Amount and USD Value Time Series'
        if button_id!=None:
            button_id = button_id.split('_')[0]
            fig_title += f' {button_id}'

        # Generate the graph
        fig = px.line(filtered_df, x='Date', y='Value', labels={'Date': 'Date', 'Value': 'Value'},
                  title=fig_title,hover_data={"Value": ":$.2f", "Date": True},color_discrete_sequence=['lightgrey'])

        # Add Bitcoin Amount as a new trace
        fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Amount'], mode='lines',
                                 name='Amount', yaxis='y2',line=dict(color=darker_gold_color,shape='spline'),
                                 hovertemplate='Date=%{x|%b %d, %Y}<br>Amount=%{y:,.8f}'))

        # Customize the layout for sleek y-axis ticks
        fig.update_layout(
            legend=dict(
                x=1.1, # Set the legend's x position to 1.1 (right)
                y=1.2, # Set the legend's y position to 1.2 (top)
                xanchor='right', # Specify legend's location
                yanchor='top' 
            ),
            yaxis=dict(
                title='USD Value',
                tickmode='linear',
                tick0=filtered_df['Value'].min(), # Set the lowest tick to the minimum value 
                dtick=(filtered_df['Value'].max() - filtered_df['Value'].min()) / 4, # Calculate tick interval for 5 ticks
                tickformat='$,.0f', # Format y-axis ticks as currency with 2 decimal places
                showgrid=False, 
                gridcolor='lightgray',  
            ),
            yaxis2=dict(
                title='BTC Amount',
                overlaying='y',
                side='right',
                tickmode='linear',
                tick0 = filtered_df['Amount'].min(), # Set the lowest tick to the minimum amount
                dtick=(filtered_df['Amount'].max() - filtered_df['Amount'].min()) / 4, # Calculate tick interval for 5 ticks
                showgrid=False,       
                tickformat=',.8f', # Format y2-axis ticks with 8 decimal places
            ),
            xaxis=dict(
                hoverformat='%b %d, %Y',
                title='Date',
                showgrid=False),
            paper_bgcolor='black',  
            plot_bgcolor='black',   
            font=dict(color=darker_gold_color)
        )
        
        # Add another trace so that "Value" appears on the legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Value',
                                 marker=dict(color='lightgrey'), showlegend=True))
    # Show the plot
    return fig,usd_val

# Initialize the Dash app object
app = Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])

# Initialize the Dash app server via Render.com
server = app.server

# Specify style for html.buttons in the Dash app layout
button_style = {'background-color': 'black', 'color': 'rgb(184, 134, 11)', 'border': '1px solid rgb(184, 134, 11)'}

# Create the popover content for the "i" icon in the top right of the application's navbar (header)
popover_content_info = dbc.Popover([
        dbc.PopoverHeader("Application Quick Start Guide", style={'font-weight':'bold','background-color': 'rgb(184, 134, 11)', 'color': 'black'}),
        dbc.PopoverBody([
            html.Strong("Preface: Please read the attached User Guide with this Milestone 3 submission for detailed explanations on using the application."),
            html.Br(),
            html.Strong("1. Begin by entering all your public wallet addresses first."),
            html.Div("You can copy public wallet addresses by hovering over the wallet icon to the right of this i icon."),
            html.Strong("2. Examine and play with the graph displayed on the page."),
            html.Div("You can filter this graph with the time buttons in the top left; you can also change the view (individual-wallet-level or portfolio-level) in the top right."),
            html.Strong("3. Change the price calculation method via the dropdown in the middle right of the page."),
            html.Div("This will allow you to change the underlying price calculation of your Bitcoin's USD value to the daily low, average (of open and closing price), or high price using Yahoo Finance data."),
            html.Strong("4. Select a prediction year to see what your current BTC amount's future value could be. ONLY DO THIS ONCE ALL WALLETS ARE INPUT."),
            html.Div("Notice that a trace in the projection graph is shown for the linear regression prediction based on the low, average, and high BTC historical prices separately."),
            html.Strong("5. Click the 'clear wallets' button on the left side panel to reset the application entirely."),
        ], style = {'color':'white'}),
    ],
    trigger='hover',
    target='app_info',
    style={'max-width': '1000px','background-color': 'black','color': 'white'}
)

# Create the popver content for the "wallets" icon in the top right of the application's navbar (header)
popover_content_wallets = dbc.Popover([
        dbc.PopoverHeader("Public Bitcoin Wallet Addresses for Testing", style={'font-weight':'bold','background-color': 'rgb(184, 134, 11)', 'color': 'black'}),
        dbc.PopoverBody([
            html.Strong("Small BTC Amount History:"),
            html.Div("35nVM2jFH4VhhnEqvQnVVVs9b6U3pJnzmG 1FV3sVEhib1KF8WqwMmJmmHLD9UAGVKQiU 1KaxPqBzRr76EmueqPgKsdwrPxCrNTDspu"),
            html.Div("17BLucvnjQuMgvGLSarDqC8nxrj8PoAEnV 1MpeoHWC82iaix8X7k79esYwH8P3SNZk6G 19eL91vPS3eHQiL5wAvRP6HAZj3AdY4rv2"),
            html.Strong("Medium BTC Amount History:"),
            html.Div("1Ric8cLznTzfEou6XsQakshST5VXJJKkf 17jGZpvEUGbSDkvt8AqniGMbbek12VZtZc 1BbqgqEqEZ2jvTWCvVPjmi8xaVzsFjcorP"),
            html.Div("1JXN3G3Z8DiuUgxvGEKAqk2kvWs7T3wL2E 1466GDyUBh7BkqjXqAuQk6SaBJp83iyMRf"),
            html.Strong("Large BTC Amount History:"),
            html.Div("19XMqP6XgFMBLAQmCFnxo7eZd2zMFHVF4a 1CmmGYZBrSLMrxAiJupY2aF4gHyJe2VzJu 1LHXajb4UGW6x6i9VkvcxRiaNVLBFyqUz2 1GoR3H3kSc6cG3YomaVRqKtBJRanqrha5Z"),
            html.Strong("Very Large BTC Amount History"),
            html.Div("17twDmWFPbecR6TtZPaD172A82b7PJStxW")
        ],style = {'color':'white'}),
    ],
    trigger='hover',
    target='bitcoin_wallets',
    style={'max-width': '1000px','background-color': 'black','color': 'white'}
)

# Create the navbar (header) of the application
navbar = dbc.Navbar([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3('Bitcoin Public Wallet Address Portfolio Tracker', style={'display':'inline-block'})
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink(html.I(className = "fa-solid fa-circle-info"), id = 'app_info', style = {'font-size':'30px'})),
                    dbc.NavItem(dbc.NavLink(html.I(className = "fa-solid fa-wallet"), id = 'bitcoin_wallets', style = {'font-size':'30px'})),
                    dbc.NavItem(dbc.NavLink(html.I(className = "fa-brands fa-bitcoin"), style={'font-size': '30px'}))
                ], navbar = True, pills = True),
                popover_content_info,
                popover_content_wallets
            ])
        ], className = 'ms-auto'),
    ], fluid = True)
], id = 'navbar', color='rgb(184, 134, 11)')

# Create the side card object that goes on the left side of the page for wallet address input/display
side_card = dbc.Card(dbc.CardBody([
    dbc.Label("Enter A Bitcoin Public Wallet Address", html_for = 'wallet_addresses_text_input', style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
    dcc.Input(id = 'wallet_addresses_text_input', placeholder = 'Enter wallet address', style=button_style),
    html.Button('Submit', id = 'wallet_address_submit_button', n_clicks = 0, style=button_style),
    html.Div([
        html.Div("0 Addresses Added:",id='wallet_addresses_counter_and_label',style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
        html.Div(id = 'wallet_addresses'),
        html.Br(),
        html.Button('Clear Wallets', id = 'wallet_addresses_clear_button', n_clicks = 0, style=button_style)
    ], style = {'height':'100%'})
]), style = {'height': '100vh', 'background-color':'black','border':'2px solid rgb(184, 134, 11)'})

# Design the Dash application's layout; this layout is heavily formatted using dbc.Row and dbc.Col objects
app.layout = html.Div([
    dbc.Row([
        dbc.Col([navbar],width=12)
    ]),
    html.Br(),
    dbc.Row([
        # Left Side Panel for Bitcoin Address Entry
        dbc.Col(side_card, width=2), 
        dbc.Col([
            dbc.Row([
                # Chart Filtering Buttons
                dbc.Col([
                    html.Div([
                        html.Br(),
                        html.Br(),
                        html.Button('1D',id='1D_button', style=button_style),
                        html.Button('5D',id='5D_button', style=button_style),
                        html.Button('1M',id='1M_button', style=button_style),
                        html.Button('3M',id='3M_button', style=button_style),
                        html.Button('6M',id='6M_button', style=button_style),
                        html.Button('YTD',id='YTD_button', style=button_style),
                        html.Button('1Y',id='1Y_button', style=button_style),
                        html.Button('5Y',id='5Y_button', style=button_style),
                        html.Button('ALL',id='ALL_button', style=button_style)
                    ])
                ], width=5),
                dbc.Col([
                    html.Div([
                        html.Br(),
                        html.Br(),
                        # Radio Items for Portfolio / Individual Wallets Time Series View
                        dbc.Row([
                            dbc.Col([
                                html.Div("View:", style={'font-weight':'bold', 'color':'rgb(184,134,11)'}),
                            ], width={'size':1, 'offset':5}),
                            dbc.Col([
                                dcc.RadioItems(
                                    id='filter_view_radio_items',
                                    options=[
                                        {'label':'Portfolio', 'value':'portfolio'},
                                        {'label':'Individual Wallets', 'value':'individual'}
                                    ],
                                    value='portfolio',
                                    labelStyle={'display':'inline', 'margin-right':'10px'},
                                    inline=True,
                                    style={'color':'rgb(184,134,11)'}
                                )
                            ], width=6)
                        ])
                    ])
                ], width = 7)
            ]),
            html.Div([
                # Object that contains the wallet_graph on the page; uses the generated empty figure initially
                dcc.Graph(id='wallet_graph',
                          figure=generate_graph(session_state.filtered_all_wallet_df,'24hr-average','portfolio')[0]
                         )
            ], style={'border': '2px solid rgb(184,134,11)'}), 
            html.Br(),
            # Portfolio metrics that go below the wallet_graph
            dbc.Row([
                dbc.Col([
                    dbc.Label("Current Bitcoin Balance:",html_for='current_bitcoin_balance',style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
                    dcc.Input(
                        id='current_bitcoin_balance',
                        value=0,
                        readOnly=True,
                        style={'width': '200px','height':'35px','text-align': 'center'}
                    )
                ], width ={'offset':1,'size':3}),
                dbc.Col([
                    dbc.Label("Current Bitcoin USD Value:",html_for='current_bitcoin_usd_value',style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
                    dcc.Input(
                        id='current_bitcoin_usd_value',
                        value=0,
                        readOnly=True,
                        style={'width': '200px','height':'35px','text-align': 'center'}
                    )
                ], width = 3),
                dbc.Col([
                    dbc.Label("Calculation Method:",html_for='price_type_dropdown',style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
                    dcc.Dropdown(
                        id='price_type_dropdown',
                        options=['24hr-low','24hr-high','24hr-average'],
                        value='24hr-average',
                        style={'width': '200px', 'height':'35px'}
                    )
                ], width = 3)
            ], justify = 'evenly'),  
            html.Br(),
            html.Br(),
            html.Br(),
            dbc.Row([
                dbc.Col([],width={'offset':1,'size':3}),
                dbc.Col([
                    dbc.Tooltip("Please only select a prediction year after inputting all wallets needed. After the initial prediction year selection, you will need to choose a new year for this graph to rerun for the new year.", target="projection_target_year_label", placement="top"),
                    dbc.Label("Price Prediction Target Year:",id='projection_target_year_label',html_for='projection_target_year',style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
                    dcc.Dropdown(
                        id='projection_target_year',
                        options=[{'label': year, 'value': year} for year in [str(year) for year in range(2024, 2040)]],
                        placeholder='Select a future year.',
                        style={'width':'200px','text-align': 'center'}
                    ), 
                ], width = 3),
                dbc.Col([],width=3)
            ], justify = 'evenly'),
            html.Br(),
            html.Div([
                # Object that contains the projection_graph on the page
                html.Div([dcc.Graph(id='projection_graph')],style={'border': '2px solid rgb(184,134,11)'}),
                html.Br(),
                # Portfolio projection metrics that go below the projection graph
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Bitcoin Balance Used in Projection:",html_for='projection_bitcoin_balance',style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
                        dcc.Input(
                            id='projection_bitcoin_balance',
                            value=0,
                            readOnly=True,
                            style={'width':'200px','height':'35px','text-align': 'center'}
                        )
                    ], width = {'offset':1,'size':3}),
                    dbc.Col([
                        dbc.Tooltip("This displays the prediction value based on average price as input. Please reference the above graph for predictions based on low/high price.", target="projection_btc_usd_value_label", placement="top"),
                        dbc.Label("Projected Bitcoin USD Value:",id='projection_btc_usd_value_label',html_for='projection_btc_usd_value',style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
                        dcc.Input(
                            id='projection_btc_usd_value',
                            value=0,
                            readOnly=True,
                            style={'width':'200px','height':'35px','text-align': 'center'}
                        )
                    ], width = 3),
                ], justify = 'evenly'),
            ], id='projection_graph_div',style={'display':'none'}),
            html.Br(),
            html.Br(),
            html.Br(),
        ], width=9)
    ])
], style={'width':'100%','background-color': 'black'})
    
# This callback / function changes the view of wallet_graph (main graph on the page) when a new "view" item in the top right of the application is selected
@app.callback(
    Output('wallet_graph','figure'),
    Input('filter_view_radio_items','value'),
    State('price_type_dropdown','value'),
    prevent_initial_call=True)
def change_graph_view(radio_value,price_type):
    fig = None
    if radio_value=='individual':
        fig = generate_graph(session_state.raw_all_wallet_df,price_type,radio_value)[0]
    elif radio_value=='portfolio':
        fig = generate_graph(session_state.filtered_all_wallet_df,price_type,radio_value)[0]
    return fig
        
# This callback / function updates the wallet address count on the left side card of the application whenever the len of the "wallet_address" object's children changes
@app.callback(
    Output('wallet_addresses_counter_and_label','children',allow_duplicate=True),
    Input('wallet_addresses','children'),
    prevent_initial_call=True)
def update_address_count(children):
    address_count = 0
    for child in children:
        if 'INVALID' not in child['props']['children']:
            address_count+=1
    return str(address_count)+' Addresses Added'

# This callback / function resets the entire application to its initial state; it passes the respective default values for each object back to each object on the page
@app.callback(
    Output('wallet_addresses_text_input','value',allow_duplicate=True),
    Output('wallet_addresses_counter_and_label','children',allow_duplicate=True),
    Output('wallet_addresses','children',allow_duplicate=True),
    Output('filter_view_radio_items','value',allow_duplicate=True), 
    Output('wallet_graph','figure',allow_duplicate=True),
    Output('current_bitcoin_balance','value',allow_duplicate=True),
    Output('current_bitcoin_usd_value','value',allow_duplicate=True),
    Output('price_type_dropdown','value',allow_duplicate=True),
    Output('projection_target_year','value',allow_duplicate=True),
    Output('projection_graph','figure',allow_duplicate=True),
    Output('projection_bitcoin_balance','value',allow_duplicate=True),
    Output('projection_btc_usd_value','value',allow_duplicate=True),
    Output('projection_graph_div','style',allow_duplicate=True),
    Input('wallet_addresses_clear_button','n_clicks'),
    prevent_initial_call=True)
def reset_application(n_clicks):
    if n_clicks>0:
        session_state.raw_all_wallet_df = pd.DataFrame(columns=['Date', 'Amount', 'Value', 'Low', 'Avg Price', 'High', 'Wallet'])
        session_state.filtered_all_wallet_df = pd.DataFrame(columns=['Date', 'Amount', 'Value', 'Low', 'Avg Price', 'High', 'Wallet'])
        return [None,
                '0 Addresses Added:',
                [],
                'portfolio',
                generate_graph(session_state.filtered_all_wallet_df,'24hr-average','portfolio')[0],
                0,
                0,
                '24hr-average',
                None,
                generate_graph(session_state.filtered_all_wallet_df,'24hr-average','portfolio')[0],
                0,
                0,
                {'display':'none'}]

# This callback / function changes the "wallet_graph" (main graph on the page) to display value based on the pricing method selected
@app.callback(
    Output('wallet_graph','figure',allow_duplicate=True), # By default, objects should not used as output in multiple callbacks; have to allow duplicates manually
    Output('current_bitcoin_balance','value',allow_duplicate=True),
    Output('projection_bitcoin_balance','value',allow_duplicate=True),
    Output('current_bitcoin_usd_value','value',allow_duplicate=True),
    Input('price_type_dropdown','value'),
    State('wallet_addresses','children'),
    State('filter_view_radio_items','value'),
    prevent_initial_call=True)
def change_price_calculation(price_type,wallet_addresses,radio_value):
    fig,curr_usd_value = generate_graph(session_state.filtered_all_wallet_df,price_type,radio_value)
    if len(wallet_addresses)>0:
        curr_btc_balance = session_state.filtered_all_wallet_df['Amount'].iloc[-1]
        curr_usd_value = f"${curr_usd_value:.2f}"
        return fig,curr_btc_balance,curr_btc_balance,curr_usd_value
    else:
        return fig,0,0,0

# This callback / function tracks the user wallets added and processes them (adds them to the page, generates graphs, generates session_state's DataFrames for the wallet)
@app.callback(
    Output('wallet_addresses_text_input','value',allow_duplicate=True),
    Output('wallet_addresses','children',allow_duplicate=True),
    Output('wallet_graph','figure',allow_duplicate=True), # By default, objects should not used as output in multiple callbacks; have to allow duplicates manually
    Output('current_bitcoin_balance','value',allow_duplicate=True),
    Output('projection_bitcoin_balance','value',allow_duplicate=True),
    Output('current_bitcoin_usd_value','value',allow_duplicate=True),
    Input('wallet_address_submit_button','n_clicks'),
    State('wallet_addresses_text_input','value'),
    State('wallet_addresses','children'),
    State('price_type_dropdown','value'),
    State('filter_view_radio_items','value'),
    prevent_initial_call=True
)
def update_portfolio_display(n_clicks,input_value,wallet_addresses,price_type,radio_value):
    input_value = input_value.strip()
    if wallet_addresses is None:
        wallet_addresses = []
    if n_clicks > 0 and input_value:
        if validate_wallet(input_value): 
            input_value = '-'+str(input_value)
        else: 
            input_value = '-INVALID'+str(input_value)
        wallet_addresses.append(html.Div(input_value,style={'color': 'rgb(184, 134, 11)'}))
        input_value = ''
    fig,curr_usd_value = generate_graph(session_state.filtered_all_wallet_df,price_type,radio_value)
    
    # Grab the current btc balance / usd value by querying these respective columns in the last row of the session_state's filtered_all_wallet_df
    curr_btc_balance = session_state.filtered_all_wallet_df['Amount'].iloc[-1]
    curr_usd_value = f"${curr_usd_value:.2f}"
    return input_value,wallet_addresses,fig,curr_btc_balance,curr_btc_balance,curr_usd_value

# This callback / function the "wallet_graph" similar to a traditional stock-chart; identify what timespan button was clicked by user and filter the time series graph accordingly
@app.callback(
    Output('wallet_graph','figure',allow_duplicate=True),
    Input('1D_button','n_clicks'),
    Input('5D_button','n_clicks'),
    Input('1M_button','n_clicks'),
    Input('3M_button','n_clicks'),
    Input('6M_button','n_clicks'),
    Input('YTD_button','n_clicks'),
    Input('1Y_button','n_clicks'),
    Input('5Y_button','n_clicks'),
    Input('ALL_button','n_clicks'),
    State('price_type_dropdown','value'),
    State('filter_view_radio_items','value'),
    prevent_initial_call=True)
def time_filter_graph(clicks_1d, clicks_5d, clicks_1m, clicks_3m, clicks_6m, clicks_ytd, clicks_1y, clicks_5y, clicks_all, price_type, radio_value):
    # Check and store which input triggered the callback
    ctx = dash.callback_context
    button_id = None if not ctx.triggered else ctx.triggered[0]['prop_id'].split('.')[0]
        
    # Initialize the days_offset for filtering to 0; grab the earliest and latest dates in the session_state's filtered_all_wallet_df 
    days_offset = 0
    earliest_date = None
    latest_date = None
    if radio_value=='individual':
        earliest_date = session_state.raw_all_wallet_df['Date'].min()
        latest_date = session_state.raw_all_wallet_df['Date'].max()
    elif radio_value=='portfolio':
        earliest_date = session_state.filtered_all_wallet_df['Date'].min()
        latest_date = session_state.filtered_all_wallet_df['Date'].max()

    # Determine days_offset by what button is clicked
    if button_id == '1D_button':
        days_offset = 1
    elif button_id == '5D_button':
        days_offset = 5
    elif button_id == '1M_button':
        days_offset = 30
    elif button_id == '3M_button':
        days_offset = 90
    elif button_id == '6M_button':
        days_offset = 180
    elif button_id == 'YTD_button':
        # Get the current year and highest date in dataframe (current date); this could be hardcoded, but I want it to be dynamic 
        current_year = datetime.now().year

        # Calculate the difference in days between january 1st of the same year and the latest date in the session_state's filtered_all_wallet_df dataframe
        start_of_year = datetime(current_year,1,1)
        if radio_value == 'individual':
            latest_date = pd.to_datetime(latest_date)
            days_offset = (latest_date - pd.Timestamp(start_of_year)).days
        elif radio_value=='portfolio':
            days_offset = (latest_date - start_of_year.date()).days 
        
    elif button_id == '1Y_button':
        days_offset = 365
    elif button_id == '5Y_button':
        days_offset = 1825
    elif button_id == 'ALL_button':
        days_offset = (latest_date - earliest_date).days
    
    # Retrieve the day equal to the latest_date minus the days_offset number of days
    offset_days_back = (latest_date - pd.DateOffset(days=days_offset)).date()
    filtered_df = None
    if radio_value=='individual':
        offset_days_back = pd.to_datetime(offset_days_back)
        filtered_df = session_state.raw_all_wallet_df[(session_state.raw_all_wallet_df['Date'] >= offset_days_back) & (session_state.raw_all_wallet_df['Date'] <= latest_date)]
    elif radio_value=='portfolio':
        filtered_df = session_state.filtered_all_wallet_df[(session_state.filtered_all_wallet_df['Date'] >= offset_days_back) & (session_state.filtered_all_wallet_df['Date'] <= latest_date)]
    
    # Generate the graph including button_id as a parameter to distinguish from the initial call in the update_portflio_display function
    fig = generate_graph(filtered_df,price_type,radio_value,button_id)[0]
    return fig

# This callback / function updates the projection graph based on the user's selected projection year
@app.callback(
    Output('projection_graph_div','style',allow_duplicate=True),
    Output('projection_graph','figure',allow_duplicate=True),
    Output('projection_btc_usd_value','value',allow_duplicate=True),
    Input('projection_target_year','value'),
    State('wallet_addresses','children'),
    State('current_bitcoin_balance','value'),
    State('filter_view_radio_items','value'),
    State('price_type_dropdown','value'),
    prevent_initial_call=True
)
def update_projection_display(input_year,wallet_addresses,btc_bal,radio_value,price_type):
    empty_df = pd.DataFrame([{'Date':datetime.today().date(),'Amount':0,'Value':0,'Low':0,'Avg Price':0,'High':0}])
    empty_fig = generate_graph(empty_df,'24hr-average',radio_value)[0]
    if input_year==None:
        return {'display':'none'},empty_fig,0
    if wallet_addresses==None:
        return {'display':'block'},empty_fig,0
    if input_year.isnumeric() and len(wallet_addresses)>0:
        # Make a copy of the BTC_df, which was created at the start of the script and contains Bitcoin pricing data from yfinance
        df = BTC_df.copy() # Filter the hard-coded data before Sept 17, 2014, the first pricing date from yfinance, out
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df['Date'] >= '2014-09-17']

        # Split the data into x and y variables for the linear regression model
        df['Date_numeric'] = df['Date'].apply(lambda x: x.toordinal())
        X = df['Date_numeric'].values.reshape(-1, 1)
        y1 = df['Avg Price']
        y2 = df['Low']
        y3 = df['High']

        # Create the 3 separate linear regression models (based on low price, average price, and high price)
        model_avg = LinearRegression()
        model_low = LinearRegression()
        model_high = LinearRegression()

        # Fit the model on the historical pricing data
        model_avg.fit(X, y1)
        model_low.fit(X, y2)
        model_high.fit(X, y3)

        # Create a sequence of dates from the starting date to the future date (the end of the user-selected year) 
        future_date = f'{input_year}-12-31'
        future_date = pd.to_datetime(future_date) 
        date_range = pd.date_range(start=df['Date'].max(), end=future_date, freq='D') # Use the maximum date from the BTC_df (the current day) as the starting point of predictions dataframe
        ordinal_dates = [date.toordinal() for date in date_range] # Ordinals must be used so that dates are treated as integers in the model

        # Make predictions for each day in the date range
        predictions = {'Date': date_range,
                       'Avg Price': model_avg.predict([[ordinal_date] for ordinal_date in ordinal_dates]),
                       'Low Price': model_low.predict([[ordinal_date] for ordinal_date in ordinal_dates]),
                       'High Price': model_high.predict([[ordinal_date] for ordinal_date in ordinal_dates])}
        predicted_df = pd.DataFrame(predictions)

        # Calculate the value of the user's portfolio based on the prediction multplied by the user's current BTC balance
        predicted_df['Avg Value'] = predicted_df['Avg Price'] * btc_bal
        predicted_df['Low Value'] = predicted_df['Low Price'] * btc_bal
        predicted_df['High Value'] = predicted_df['High Price'] * btc_bal
        
        # Calculate the value of the user's portfolio based on their current data multiplied by the user's current BTC balance
        prev = session_state.filtered_all_wallet_df.copy()
        prev['Avg Value'] = prev['Avg Price'] * btc_bal
        prev['Low Value'] = prev['Low'] * btc_bal
        prev['High Value'] = prev['High'] * btc_bal

        # Append the predicted_df to the actual dataframe (prev)
        predicted_df = pd.concat([prev,predicted_df],axis=0)

        # Create a time series graph to display the 3 predictions (low, average, high values)
        fig = px.line(predicted_df, x='Date', y=['Avg Value', 'Low Value', 'High Value'], title='Bitcoin Value Over Time',
                      color_discrete_map={'Avg Value': 'lightgrey', 'Low Value': 'rgb(184,134,11)', 'High Value': '#1f77b4'})
        proj_btc_val = predicted_df['Avg Value'].iloc[-1]
        proj_btc_val = f"${proj_btc_val:.2f}"

        # Format the time series graph
        fig.update_layout(
            legend_title_text='Price Calculation',  
            legend=dict(
                x=1.1,  
                y=1.2, 
                xanchor='right', 
                yanchor='top'
            ),
            yaxis=dict(
                title='USD Value',
                tickmode='linear',
                tick0=predicted_df['Avg Value'].min(),  # Set the lowest tick to the minimum value
                dtick=(predicted_df['Avg Value'].max() - predicted_df['Avg Value'].min()) / 4, # Calculate tick interval for 5 ticks
                tickformat='$,.0f',  # Format y-axis ticks as currency with 2 decimal places
                showgrid=False,
                gridcolor='lightgray',
            ),
            xaxis=dict(
                hoverformat='%b %d, %Y',
                title='Date',
                showgrid=False
            ),
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='rgb(184,134,11)')
        )
        return {'display':'block'},fig,proj_btc_val
    return {'display':'none'},empty_fig,0 

# Main function to run the Dash application / server
if __name__ == "__main__":
    app.run_server(debug=True) # THIS LINE OR SOMETHING SIMILAR WILL BE USED IN OTHER IDE's; NOTE THAT THE APP IS FORMATTED FOR EXTERNAL WINDOWS
    # I HAVE ONLY USED JUPYTER_LABS FOR THIS ASSIGNMENT; I CANNOT SPEAK ON OTHER IDE's
    #app.run_server(jupyter_mode='external',port=7953,debug=True)
    
    # To run this application on Jupyter without the Dash debug pop-up, remove debug=True
    # app.run_server(jupyter_mode='external',port=7953)


# In[13]:


# print("Pandas version:", pd.__version__)
# print("NumPy version:", np.__version__)
# print("Requests version:", requests.__version__)
# print("yfinance version:", yf.__version__)
# print("Dash version:", dash.__version__)
# print("Scikit-learn version:", sklearn.__version__)
# # Pandas version: 1.5.3
# # NumPy version: 1.24.3
# # Requests version: 2.31.0
# # yfinance version: 0.2.28
# # Dash version: 2.13.0
# # Scikit-learn version: 1.2.2


# In[14]:


# print(dbc.__version__)


# In[16]:


# print(plotly.__version__)


# In[10]:


# import sys
# print(sys.version)


# In[ ]:




