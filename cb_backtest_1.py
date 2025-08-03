import pandas as pd
import numpy as np
from datetime import datetime
from cb_pricing_test3 import CB_Pricer, PARAMS as default_params

def load_rate_data(rate_file='rate.csv'):
    try:
        # Read the rate data
        rate_df = pd.read_csv(rate_file, encoding='utf-8')
        rate_column = "台灣銀行-定期存款-一年-機動(%)"
        
        if rate_column not in rate_df.columns:
            print(f"Error: Column '{rate_column}' not found in rate.csv")
            print(f"Available columns: {list(rate_df.columns[:5])}...")
            return None
        
        # Select date and rate
        rate_data = rate_df[['月', rate_column]].copy()
        rate_data.columns = ['Date', 'Rate']
        
        # Convert date format 
        rate_data['Date'] = pd.to_datetime(rate_data['Date'], format='%YM%m')
        
        # Convert rate from percentage to decimal
        rate_data['Rate'] = pd.to_numeric(rate_data['Rate'], errors='coerce') / 100
        rate_data = rate_data.dropna()
        
        # Set date as index
        rate_data.set_index('Date', inplace=True)
        
        print(f"Loaded rate data: {len(rate_data)} monthly observations")
        print(f"Date range: {rate_data.index.min()} to {rate_data.index.max()}")
        print(f"Rate range: {rate_data['Rate'].min():.3f} to {rate_data['Rate'].max():.3f}")
        
        return rate_data
        
    except Exception as e:
        print(f"Error loading rate data: {e}")
        return None

def load_and_prepare_data(stock_path, cb_path, condition_path, rate_data=None):
    try:
        stock_df = pd.read_excel(stock_path, header=0)
        cb_df = pd.read_excel(cb_path, header=0)
        condition_df = pd.read_excel(condition_path, header=1)
        stock_df['時間'] = pd.to_datetime(stock_df['時間'])
        cb_df['時間'] = pd.to_datetime(cb_df['時間'])
        
        df = pd.merge(stock_df[['時間', '收盤價']], cb_df[['時間', '收盤價']], on='時間', suffixes=('_stock', '_cb'))
        df.rename(columns={'收盤價_stock': 'Stock_Close', '收盤價_cb': 'CB_Market_Price'}, inplace=True)
        df.set_index('時間', inplace=True)
        df['Stock_Return'] = df['Stock_Close'].pct_change()
        df['s_vol'] = df['Stock_Return'].rolling(window=60).std() * np.sqrt(252)
        conditions = condition_df.iloc[0].to_dict()
        
        conversion_price = float(conditions['轉換價格'])
        face_value = float(conditions.get('發行面額', 100000))
        call_price = float(conditions.get('發行人贖回權價格', face_value))
        previous_conversion_price = float(conditions.get('前次轉換價格', conversion_price))
        anti_dilution_price = float(conditions.get('反稀釋價格', conversion_price))
        anti_dilution_adjustment = float(conditions.get('反稀釋重設幅度', 0))
        anti_dilution_cumulative = float(conditions.get('反稀釋累計重設幅度', 0))
        previous_convertible_shares = float(conditions.get('可轉換股數', face_value / conversion_price))
        anti_dilution_shares = float(conditions.get('反稀釋轉換股數', face_value / conversion_price))
        dividend_base_date_str = conditions.get('配股基準日', '')
        
        if dividend_base_date_str and dividend_base_date_str != 'nan':
            try:
                conditions['Dividend_Base_Date'] = datetime.strptime(dividend_base_date_str, '%Y/%m/%d')
            except:
                conditions['Dividend_Base_Date'] = None
        else:
            conditions['Dividend_Base_Date'] = None
        
        conditions['previous_conversion_price'] = previous_conversion_price
        conditions['original_conversion_price'] = conversion_price
        conditions['original_conv_ratio'] = face_value / conversion_price
        conditions['anti_dilution_price'] = anti_dilution_price
        conditions['anti_dilution_adjustment'] = anti_dilution_adjustment
        conditions['anti_dilution_cumulative'] = anti_dilution_cumulative
        conditions['previous_convertible_shares'] = previous_convertible_shares
        conditions['anti_dilution_shares'] = anti_dilution_shares
        conditions['conv_ratio'] = conditions['original_conv_ratio']
        
        maturity_date_str = conditions['到期日']
        conditions['Maturity_Date'] = datetime.strptime(maturity_date_str, '%Y/%m/%d')
        df['T'] = (conditions['Maturity_Date'] - df.index).days / 365.25
        
        # Integrate rate data
        if rate_data is not None:
            # Resample rate data to daily
            daily_rate_data = rate_data.resample('D').ffill()
            df = df.merge(daily_rate_data, left_index=True, right_index=True, how='left')
            df.rename(columns={'Rate': 'r0'}, inplace=True)
            
            # Forward fill missing rates within each month
            df['r0'] = df['r0'].ffill()
            
            print(f"Rate data integrated: {df['r0'].notna().sum()} days have rate data")
        else:
            df['r0'] = None
        
        df_clean = df.dropna(subset=['Stock_Close', 'CB_Market_Price', 'T'])
        df_clean['s_vol'] = df_clean['s_vol'].bfill().ffill()
        
        # Fill missing rates with default
        if rate_data is not None:
            df_clean['r0'] = df_clean['r0'].fillna(default_params['r0'])
        else:
            df_clean['r0'] = default_params['r0']
        
        df = df_clean
        
        if df.shape[0] == 0:
            print("ERROR: All data was dropped! Check for missing values.")
            return None, None
        
        return df, conditions
    
    except FileNotFoundError as e:
        print(f"File not found error: {repr(e)}")
        return None, None
    
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        return None, None

def get_effective_conversion_params(current_date, conditions):
    
    base_price = conditions['original_conversion_price']
    base_ratio = conditions['original_conv_ratio']
    previous_price = conditions['previous_conversion_price']
    previous_shares = conditions['previous_convertible_shares']
    anti_dilution_price = conditions['anti_dilution_price']
    anti_dilution_shares = conditions['anti_dilution_shares']
    dividend_base_date = conditions.get('Dividend_Base_Date')
    
    if dividend_base_date and current_date >= dividend_base_date:
        effective_price = anti_dilution_price
        effective_ratio = anti_dilution_shares
    else:
        if previous_price != base_price:
            effective_price = base_price
            face_value = conditions.get('發行面額', 100000)
            effective_ratio = face_value / effective_price
        else:
            effective_price = base_price
            effective_ratio = base_ratio
    
    return effective_price, effective_ratio

def calculate_model_prices(df, conditions, max_n_steps=100):
    model_prices = []
    
    for index, row in df.iterrows():
        current_params = default_params.copy()
        current_params['S_0'] = row['Stock_Close']
        current_params['T'] = row['T']
        current_params['s_vol'] = row['s_vol']
        
        effective_price, effective_ratio = get_effective_conversion_params(index, conditions)
        current_params['conv_ratio'] = effective_ratio
        current_params['K'] = effective_price
        
        # Ignore call price
        current_params['call_price'] = 1e10  
        
        theoretical_N = int(np.round(row['T'] * 365))
        current_params['N'] = max(1, min(theoretical_N, max_n_steps))
        
        # Use dynamic rate data if available
        if row['r0'] is not None and not pd.isna(row['r0']):
            current_params['r0'] = row['r0']
        else:
            current_params['r0'] = default_params['r0']
        
        try:
            pricer = CB_Pricer(**current_params)
            cb_price, _, _, _, _, _ = pricer.price(verbose=False)
            model_prices.append(cb_price)
        except Exception as e:
            print(f"Error calculating model price for {index}: {e}")
            model_prices.append(np.nan)
    
    return model_prices

def run_backtest_with_rate_data():
    
    rate_data = load_rate_data('rate.csv')
    
    # Load data
    stock_file = '1101_Stock_price.xlsx'
    cb_file = '11011_CB_Price.xlsx'
    condition_file = '11011_condition.xlsx'
    data_df, cb_conditions = load_and_prepare_data(stock_file, cb_file, condition_file, rate_data)
    
    # Calculate model prices
    model_prices = calculate_model_prices(data_df, cb_conditions, max_n_steps=100)
    data_df['Model_Price'] = model_prices
    
    # Remove rows with NaN model prices
    data_df = data_df.dropna(subset=['Model_Price'])
    
    # Generate signals
    data_df['Signal'] = np.where(data_df['Model_Price'] > data_df['CB_Market_Price'], 1, -1)
    
    # Simulate trading
    print("Simulating trading...")
    portfolio = pd.DataFrame(index=data_df.index)
    portfolio['Cash'] = 1_000_000
    portfolio['Position'] = 0.0
    portfolio['Holdings_Value'] = 0.0
    portfolio['Total_Value'] = 1_000_000
    portfolio['Transaction_Costs'] = 0.0
    
    trades = []
    current_position = 0.0
    current_cash = 1_000_000
    commission_rate = 0.001425
    
    for i, (date, row) in enumerate(data_df.iterrows()):
        signal = row['Signal']
        cb_price = row['CB_Market_Price']
        
        # Update portfolio values
        portfolio.loc[date, 'Cash'] = float(current_cash)
        portfolio.loc[date, 'Position'] = float(current_position)
        portfolio.loc[date, 'Holdings_Value'] = float(current_position * cb_price)
        portfolio.loc[date, 'Total_Value'] = float(current_cash + (current_position * cb_price))
        portfolio.loc[date, 'Transaction_Costs'] = float(portfolio.loc[date, 'Transaction_Costs'] if i > 0 else 0)
        
        # Execute trades based on signals
        if signal == 1 and current_position == 0:  # Buy signal
            commission = current_cash * commission_rate
            available_cash = current_cash - commission
            position_size = available_cash / cb_price
            current_position = position_size
            current_cash = 0
            
            trades.append({
                'Date': date,
                'Type': 'Buy',
                'Price': cb_price,
                'Size': position_size,
                'Commission': commission
            })
            
            if i > 0:
                portfolio.loc[date, 'Transaction_Costs'] = portfolio.loc[data_df.index[i-1], 'Transaction_Costs'] + commission
            else:
                portfolio.loc[date, 'Transaction_Costs'] = commission
                
        elif signal == -1 and current_position > 0:  # Sell signal
            gross_proceeds = current_position * cb_price
            commission = gross_proceeds * commission_rate
            net_proceeds = gross_proceeds - commission
            current_cash = net_proceeds
            current_position = 0
            
            trades.append({
                'Date': date,
                'Type': 'Sell',
                'Price': cb_price,
                'Size': current_position,
                'Commission': commission
            })
            
            if i > 0:
                portfolio.loc[date, 'Transaction_Costs'] = portfolio.loc[data_df.index[i-1], 'Transaction_Costs'] + commission
            else:
                portfolio.loc[date, 'Transaction_Costs'] = commission
    
    # Calculate performance
    final_value = portfolio['Total_Value'].iloc[-1]
    net_profit = final_value - 1_000_000
    total_return = (net_profit / 1_000_000) * 100
    buy_hold_return = ((data_df['CB_Market_Price'].iloc[-1] / data_df['CB_Market_Price'].iloc[0]) - 1) * 100
    
    portfolio['Peak'] = portfolio['Total_Value'].cummax()
    portfolio['Drawdown'] = (portfolio['Total_Value'] - portfolio['Peak']) / portfolio['Peak'] * 100
    max_drawdown = portfolio['Drawdown'].min()
    
    portfolio['Daily_Return'] = portfolio['Total_Value'].pct_change()
    sharpe_ratio = (portfolio['Daily_Return'].mean() / portfolio['Daily_Return'].std()) * np.sqrt(252) if portfolio['Daily_Return'].std() != 0 else 0
    
    total_trades = len(trades)
    total_commission = sum(trade['Commission'] for trade in trades)
    
    # Results
    print("\n=== Convertible Bond Backtest Results ===")
    print(f"Strategy: Buy when Model Price > Market Price, sell when Model Price < Market Price")
    print(f"Initial Capital: $1,000,000")
    print(f"Commission Rate: 0.1425%")
    print(f"Max N Steps: 100")
    print(f"Rate Data: Taiwan Bank 1-year time deposit rate (monthly)")
    print()
    print(f"Total Return: {total_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Net Profit: ${net_profit:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Total Trades: {total_trades}")
    print(f"Total Commission: ${total_commission:,.2f}")
    
    # Show rate data summary
    if rate_data is not None:
         used_rate_data = data_df[data_df['r0'].notna()]
         
         if len(used_rate_data) > 0:
             actual_start = used_rate_data.index.min().strftime('%Y-%m')
             actual_end = used_rate_data.index.max().strftime('%Y-%m')
             print(f"\nRate Data Summary:")
             print(f"  - Used Date Range: {actual_start} to {actual_end}")
             print(f"  - Rate Range: {used_rate_data['r0'].min():.3f} to {used_rate_data['r0'].max():.3f}")
             print(f"  - Average Rate: {used_rate_data['r0'].mean():.3f}")
             print(f"  - Rate Coverage: {data_df['r0'].notna().sum()}/{len(data_df)} days")
         else:
             print(f"\nRate Data Summary: No rate data used in backtest period")
    
    return data_df, portfolio, trades

def main():
    print("Convertible Bond Arbitrage Backtest with Rate Data Integration")
    print("=" * 70)
    
    # Run the backtest
    result = run_backtest_with_rate_data()
    
    if result is not None:
        data_df, portfolio, trades = result
        print("\nBacktest completed successfully!")
        print(f"Data points processed: {len(data_df)}")
        print(f"Trades executed: {len(trades)}")
    else:
        print("Backtest failed to complete.")

if __name__ == '__main__':
    main() 