#Library
import time
import numpy as np


# Variables, could be changed OR modified to be able to accept inout files
PARAMS = {
    'S_0': 100.0,         # Initial stock price
    'K': 120.0,          # Strike price for stock
    'conv_ratio': 50,     # Conversion ratio
    'face_value': 1,      # Face value of the bond
    'call_price': 100000,      # Call price: Bond call price
    'T': 3.0,            # Time to maturity
    'r0': 0.10,          # Initial interest risk-free rate
    'r_vol': 0.10,       # Interest rate volatility
    's_vol': 0.20,       # Stock price volatility
    'pi': 0.5,           # Probability parameter for rf
    'N': 252             # Number of time steps (252 trading days)
}

class CB_Pricer:
    def __init__(self, S_0, K, conv_ratio, face_value, call_price,
                 T, N, r0, r_vol, s_vol, pi=0.5):
        """Initialize the convertible bond pricer with given parameters."""
        
        self.S_0 = S_0
        self.K = K
        self.conv_ratio = conv_ratio
        self.face_value = face_value
        self.call_price = call_price
        self.T = T
        self.N = N
        self.dt = 1/252
        self.r0 = r0
        self.r_vol = r_vol
        self.s_vol = s_vol
        self.pi = pi

        self.rf_tree = self._build_rf_tree()
        self.stock_tree = self._build_stock_tree()

    def _build_rf_tree(self):
        """Build the risk-free rate tree."""
        
        tree = np.zeros((self.N + 1, self.N + 1))
        u_r = np.exp(self.r_vol * np.sqrt(self.dt))
        d_r = 1 / u_r

        for i in range(self.N + 1):
            for j in range(i + 1):
                tree[i, j] = self.r0 * (u_r ** j) * (d_r ** (i - j))
        return tree

    def _build_stock_tree(self):
        """Build the stock price tree."""
        
        tree = np.zeros((self.N + 1, self.N + 1))
        u_s = np.exp(self.s_vol * np.sqrt(self.dt))
        d_s = 1 / u_s

        for i in range(self.N + 1):
            for j in range(i + 1):
                tree[i, j] = self.S_0 * (u_s ** j) * (d_s ** (i - j))
        return tree

    def price(self, verbose=False):
        """Main pricing function."""
        
        print("Starting valuation...")
        start_time = time.time()

        # Initialize arrays - now properly indexed for both rate and stock movements
        equity_values = np.zeros((self.N + 1, self.N + 1, self.N + 1))
        debt_values = np.zeros((self.N + 1, self.N + 1, self.N + 1))
        holding_values = np.zeros((self.N + 1, self.N + 1, self.N + 1))
        execute_values = np.zeros((self.N + 1, self.N + 1, self.N + 1))
        cb_values = np.zeros((self.N + 1, self.N + 1, self.N + 1))

        # Calculate up and down factors for stock
        u_s = np.exp(self.s_vol * np.sqrt(self.dt))
        d_s = 1 / u_s

        # Terminal values
        for i in range(self.N + 1):  # rate nodes
            for j in range(self.N + 1):  # stock nodes
                if j <= i:  # Only valid stock nodes at each time step
                    equity_values[self.N, i, j] = self.stock_tree[self.N, j] * self.conv_ratio
                    market_value = self.face_value
                    debt_values[self.N, i, j] = min(market_value, self.call_price)
                    execute_values[self.N, i, j] = max(debt_values[self.N, i, j], equity_values[self.N, i, j])
                    cb_values[self.N, i, j] = execute_values[self.N, i, j]  

                    if verbose:
                        print(f"n={self.N} i={i} j={j} | S={self.stock_tree[self.N, j]:.2f} r={self.rf_tree[self.N, i]:.2%} CB={cb_values[self.N, i, j]:.2f} H={holding_values[self.N, i, j]:.2f} Exe={execute_values[self.N, i, j]:.2f}")

        # Backward induction
        for n in range(self.N - 1, -1, -1):
            if verbose:
                print(f"\nTime Step n = {n}:")
                print("=" * 85)

            for i in range(n + 1):  # rate nodes at time n
                for j in range(n + 1):  # stock nodes at time n
                    # Current stock price and rate
                    stock_price = self.stock_tree[n, j]
                    rf = self.rf_tree[n, i]
                    p = (np.exp(rf * self.dt) - d_s) / (u_s - d_s)

                    # Calculate equity value (conversion value)
                    equity_values[n, i, j] = stock_price * self.conv_ratio
                    
                    # Calculate debt value 
                    market_value = self.face_value * np.exp(-rf * self.dt)
                    debt_values[n, i, j] = min(market_value, self.call_price)
                    
                    # Calculate execute value 
                    execute_values[n, i, j] = max(debt_values[n, i, j], equity_values[n, i, j])
                    
                    # Calculate holding value 
                    expected_future_value = 0
                    # Always include all four branches, using min() to handle boundaries
                    expected_future_value += self.pi * p * cb_values[n + 1, i + 1, j + 1]
                    expected_future_value += self.pi * (1 - p) * cb_values[n + 1, i + 1, j]
                    expected_future_value += (1 - self.pi) * p * cb_values[n + 1, i, j + 1]
                    expected_future_value += (1 - self.pi) * (1 - p) * cb_values[n + 1, i, j]
                    # Discount the expected future value
                    holding_values[n, i, j] = expected_future_value * np.exp(-rf * self.dt)
                    
                    # Calculate final CB value 
                    cb_values[n, i, j] = max(execute_values[n, i, j], holding_values[n, i, j])

                    if verbose:
                        print(f"n={n} i={i} j={j} | S={stock_price:.2f} r={rf:.2%} CB={cb_values[n, i, j]:.2f} H={holding_values[n, i, j]:.2f} Exe={execute_values[n, i, j]:.2f}")

        end_time = time.time()
        print(f"Valuation calculation time: {end_time - start_time:.4f} seconds")
        
        # Debug: Print the final CB value
        print(f"\nFinal CB Value at root node: {cb_values[0, 0, 0]:.4f}")

        return cb_values[0, 0, 0], equity_values, debt_values, cb_values, execute_values, holding_values

# Create a pricer instance and run the valuation
pricer = CB_Pricer(**PARAMS)
cb_price, equity_values, debt_values, cb_values, execute_values, holding_values = pricer.price(verbose=True)

print("-" * 30)
print(f"Calculated Convertible Bond Price (for N={PARAMS['N']}):")
print(f"  - CB Value: {cb_price:.4f}\n")
