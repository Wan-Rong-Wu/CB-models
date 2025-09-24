# cb_pricing_numpy.py
# This script implements a pricer for convertible bonds using a binomial tree approach.
# It includes detailed debug statements to trace the computation values step-by-step.


import time

import matplotlib.pyplot as plt
import numpy as np

# Variables, could be changed OR modified to be able to accept input files=========================================================
# Pricing day:2025/7/8
PARAMS = {
    "S_0": 24.35,  # Initial stock price
    "K": 35.2,  # Strike price for stock
    "conv_ratio": 2.840,  # Conversion ratio
    "face_value": 100,  # Face value of the bond
    "units": 1.000,  # Units of bond
    "call_price": 100,  # Call price: Bond call price
    "put_price": 100,  # Put price: Bond put price
    "T": 4.42739726,  # Time to maturity
    "N": 52,  # Number of time steps    #回測N要調整
    "r0": 0.10,  # Initial interest risk-free rate
    "r_vol": 0.10,  # Interest rate volatility
    "s_vol": 0.20,  # Stock price volatility
    "pi": 0.5,  # Probability parameter for rf
}


# ================================================================================
# CB Pricer class
# ===============================================================================
class CB_Pricer:
    def __init__(
        self,
        S_0,
        K,
        conv_ratio,
        face_value,
        units,
        call_price,
        put_price,
        T,
        N,
        r0,
        r_vol,
        s_vol,
        pi,
    ):
        """Initialize the convertible bond pricer with given parameters."""

        self.S_0 = S_0
        self.K = K
        self.conv_ratio = conv_ratio
        self.face_value = face_value
        self.units = units
        self.call_price = call_price
        self.put_price = put_price
        self.T = T
        self.N = N
        self.dt = T / N
        self.r0 = r0
        self.r_vol = r_vol
        self.s_vol = s_vol
        self.pi = pi

        self.rf_tree = self._build_rf_tree()
        self.stock_tree = self._build_stock_tree()

    # 1. Risk-free rate tree--------------------------------------------
    def _build_rf_tree(self):
        """Build the risk-free rate tree."""

        tree = np.zeros((self.N + 1, self.N + 1))
        u_r = np.exp(self.r_vol * np.sqrt(self.dt))
        d_r = 1 / u_r

        for i in range(self.N + 1):
            for j in range(i + 1):
                tree[i, j] = self.r0 * (u_r**j) * (d_r ** (i - j))
        return tree

    # 2. Stock price tree--------------------------------------------------
    def _build_stock_tree(self):
        """Build the stock price tree."""
        tree = np.zeros((self.N + 1, self.N + 1))
        u_s = np.exp(self.s_vol * np.sqrt(self.dt))
        d_s = 1 / u_s

        for i in range(self.N + 1):
            for j in range(i + 1):
                tree[i, j] = self.S_0 * (u_s**j) * (d_s ** (i - j))
        return tree

    # 3.CB pricing function-----------------------------------------------------
    def price(self, verbose=False):
        """Main pricing function."""

        print("Starting valuation...")
        start_time = time.time()

        # Initialize arrays - now properly indexed for both rate and stock movements
        equity_values = np.zeros((self.N + 1, self.N + 1, self.N + 1))
        debt_values = np.zeros((self.N + 1, self.N + 1, self.N + 1))
        market_values = np.zeros((self.N + 1, self.N + 1, self.N + 1))
        conv_values = np.zeros((self.N + 1, self.N + 1, self.N + 1))
        bond_values = np.zeros((self.N + 1, self.N + 1, self.N + 1))
        cb_values = np.zeros((self.N + 1, self.N + 1, self.N + 1))

        # Calculate up and down factors for stock
        u_s = np.exp(self.s_vol * np.sqrt(self.dt))
        d_s = 1 / u_s
        book_value = self.face_value * self.units

        # 3.1 Terminal values
        for i in range(self.N + 1):  # rate nodes
            for j in range(self.N + 1):  # stock nodes
                if j <= i:  # Only valid stock nodes at each time step
                    # 3.1.1 Conversion value (Stock value)
                    conv_values[self.N, i, j] = (
                        self.stock_tree[self.N, j] * self.conv_ratio
                    )
                    if conv_values[self.N, i, j] >= book_value:
                        conv_values[self.N, i, j] = conv_values[self.N, i, j]
                    else:
                        conv_values[self.N, i, j] = 0

                    # 3.1.2 Market value(equity + debt)
                    equity_values[self.N, i, j] = conv_values[self.N, i, j]

                    if conv_values[self.N, i, j] > 0:
                        debt_values[self.N, i, j] = 0
                    else:
                        debt_values[self.N, i, j] = book_value

                    market_values[self.N, i, j] = (
                        equity_values[self.N, i, j] + debt_values[self.N, i, j]
                    )

                    # 3.1.3 Bond value(min(Market value, Call value))
                    bond_values[self.N, i, j] = min(
                        market_values[self.N, i, j], self.call_price * self.units
                    )

                    # 3.1.4 CB value (max(Bond value, Conversion value))
                    cb_values[self.N, i, j] = max(
                        bond_values[self.N, i, j], conv_values[self.N, i, j]
                    )

                    if verbose:
                        print(
                            f"n={self.N} i={i} j={j} | S={self.stock_tree[self.N, j]:.2f} r={self.rf_tree[self.N, i]:.2%} CB={cb_values[self.N, i, j]:.2f} Bond={bond_values[self.N, i, j]:.2f} Conv={conv_values[self.N, i, j]:.2f}"
                        )

        # 3.2 Nodes value by backward induction
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

                    # 3.2.1 Conversion values
                    conv_values[n, i, j] = stock_price * self.conv_ratio
                    if conv_values[n, i, j] >= book_value:
                        conv_values[n, i, j] = conv_values[n, i, j]
                    else:
                        conv_values[n, i, j] = 0

                    # 3.2.2 Market values
                    equity_values[n, i, j] = (
                        p
                        * self.pi
                        * equity_values[n + 1, i + 1, j + 1]
                        * np.exp(-self.rf_tree[n + 1, i + 1] * self.dt)
                        + (1 - p)
                        * self.pi
                        * equity_values[n + 1, i + 1, j]
                        * np.exp(-self.rf_tree[n + 1, i + 1] * self.dt)
                        + p
                        * (1 - self.pi)
                        * equity_values[n + 1, i, j + 1]
                        * np.exp(-self.rf_tree[n + 1, i])
                        + (1 - p)
                        * (1 - self.pi)
                        * equity_values[n + 1, i, j]
                        * np.exp(-self.rf_tree[n + 1, i])
                    )
                    debt_values[n, i, j] = self.pi * book_value * np.exp(
                        -self.rf_tree[n + 1, i + 1] * self.dt
                    ) + (1 - self.pi) * book_value * np.exp(
                        -self.rf_tree[n + 1, i] * self.dt
                    )

                    # 3.2.3 Bond value
                    market_values[n, i, j] = (
                        equity_values[n, i, j] + debt_values[n, i, j]
                    )
                    if (
                        n >= 91
                    ):  # Repurchase    #目前是91天後可以賣回，但2025/7/8已可以買回應不用設
                        bond_values[n, i, j] = min(
                            market_values[n, i, j], self.call_price * self.units
                        )
                    else:
                        bond_values[n, i, j] = market_values[n, i, j]

                    if n >= 1095:  # Reverse Repurchase #回測天期要調整
                        bond_values[n, i, j] = max(
                            bond_values[n, i, j], self.put_price * self.units
                        )
                    else:
                        bond_values[n, i, j] = bond_values[n, i, j]

                    # 3.2.4 CB value
                    cb_values[n, i, j] = max(bond_values[n, i, j], conv_values[n, i, j])

                    if verbose:
                        print(
                            f"n={n} i={i} j={j} | S={stock_price:.2f} r={rf:.2%} CB={cb_values[n, i, j]:.2f} B={bond_values[n, i, j] * self.units:.2f} Conv={conv_values[n, i, j]:.2f}"
                        )

        end_time = time.time()
        print(f"Valuation calculation time: {end_time - start_time:.4f} seconds")

        # Debug: Print the final CB value
        print(f"\nFinal CB Value at root node: {cb_values[0, 0, 0]:.4f}")

        return (
            cb_values[0, 0, 0],
            equity_values,
            debt_values,
            cb_values,
            bond_values,
            conv_values,
        )


# ======= Plot helpers =======
def plot_binomial_tree(tree, title, fmt_value):
    """Generic binomial drawing utility for 2D tree arrays (shape (N+1, N+1))."""
    N = tree.shape[0] - 1
    plt.figure(figsize=(10, 7))
    # edges
    for n in range(N):
        for j in range(n + 1):
            x0, y0 = n, j - n / 2.0
            x1u, y1u = n + 1, (j + 1) - (n + 1) / 2.0
            x1d, y1d = n + 1, j - (n + 1) / 2.0
            plt.plot([x0, x1u], [y0, y1u])
            plt.plot([x0, x1d], [y0, y1d])
    # nodes
    for n in range(N + 1):
        for j in range(n + 1):
            x, y = n, j - n / 2.0
            plt.scatter([x], [y], s=55)
            plt.text(
                x, y + 0.10, fmt_value(tree[n, j]), ha="center", va="bottom", fontsize=8
            )
    plt.title(title)
    plt.xlabel("Time step (n)")
    plt.ylabel("Node index")
    plt.grid(True, axis="x", linestyle=":")
    plt.tight_layout()
    plt.show()


def plot_cb_slice(cb_vals, decisions, title, rate_index_fn=None):
    """
    Plot a 2D slice of the 3D CB lattice by fixing the rate index at each time.
    rate_index_fn: function n -> i (choose the rate state at time n). Default = mid (n//2).
    """
    N = cb_vals.shape[0] - 1
    if rate_index_fn is None:
        rate_index_fn = lambda n: n // 2  # mid-rate slice

    plt.figure(figsize=(12, 8))
    # edges
    for n in range(N):
        for j in range(n + 1):
            x0, y0 = n, j - n / 2.0
            x1u, y1u = n + 1, (j + 1) - (n + 1) / 2.0
            x1d, y1d = n + 1, j - (n + 1) / 2.0
            plt.plot([x0, x1u], [y0, y1u])
            plt.plot([x0, x1d], [y0, y1d])
    # nodes + labels
    for n in range(N + 1):
        i = rate_index_fn(n)
        for j in range(n + 1):
            x, y = n, j - n / 2.0
            v = cb_vals[n, i, j]
            tag = "Exe" if decisions[n, i, j] else "Hold"
            plt.scatter([x], [y], s=55)
            plt.text(
                x, y + 0.10, f"{v:.2f} ({tag})", ha="center", va="bottom", fontsize=8
            )

    plt.title(title + " — slice on rate state i = floor(n/2)")
    plt.xlabel("Time step (n)")
    plt.ylabel("Stock node index (j)")
    plt.grid(True, axis="x", linestyle=":")
    plt.tight_layout()
    plt.show()


# ======= Run & Plot =======
if __name__ == "__main__":
    # Create a pricer instance and run the valuation
    pricer = CB_Pricer(**PARAMS)
    cb_price, equity_values, debt_values, market_values, bond_values, conv_values = (
        pricer.price(verbose=False)
    )
    print("-" * 30)
    print(f"Calculated Convertible Bond Price (for N={PARAMS['N']}):")
    print(f"  - CB Value: {cb_price:.4f}\n")

    # Trees
    plot_binomial_tree(
        pricer.stock_tree, "Stock Price Binomial Tree ", lambda v: f"{v:.2f}"
    )
    plot_binomial_tree(pricer.rf_tree, "Risk-Free Rate Tree", lambda v: f"{v:.3%}")

    # # CB price (2D slice)
    # cb_vals, exe_vals, hold_vals, decisions = pricer.price_full()
    # plot_cb_slice(cb_vals, decisions, "Convertible Bond Price Tree")
