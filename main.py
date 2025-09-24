# cb_pricing_numba.py
# This script implements a pricer for convertible bonds using a binomial tree approach.
# It includes detailed debug statements to trace the computation values step-by-step.


import logging
import time
from pathlib import Path

import hvplot as hv
import hvplot.networkx as hvnx
import hvplot.polars  # noqa: This is a necessary import for hvplot with pandas
import networkx as nx
import numpy as np
import panel as pn
from numba import njit

import polars as pl

# Initialize panel extensions
hv.extension("bokeh")
pn.extension("bokeh")

# Configure logging
# Create formatters
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create handlers
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

file_handler = logging.FileHandler("cb_pricer.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all messages
    handlers=[stream_handler, file_handler],
)
logger = logging.getLogger(__name__)

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
# Numba-optimized computational kernels
# ================================================================================
@njit(parallel=True, cache=True)
def build_rf_tree_numba(r0, r_vol, dt, N):
    """Build the risk-free rate tree using numba acceleration."""
    tree = np.zeros((N + 1, N + 1))
    u_r = np.exp(r_vol * np.sqrt(dt))
    d_r = 1 / u_r

    for i in range(N + 1):
        for j in range(i + 1):
            tree[i, j] = r0 * (u_r**j) * (d_r ** (i - j))
    return tree


@njit(parallel=True, cache=True)
def build_stock_tree_numba(S_0, s_vol, dt, N):
    """Build the stock price tree using numba acceleration."""
    tree = np.zeros((N + 1, N + 1))
    u_s = np.exp(s_vol * np.sqrt(dt))
    d_s = 1 / u_s

    for i in range(N + 1):
        for j in range(i + 1):
            tree[i, j] = S_0 * (u_s**j) * (d_s ** (i - j))
    return tree


@njit(parallel=True, cache=True)
def price_backward_induction_numba(
    N,
    dt,
    rf_tree,
    stock_tree,
    conv_ratio,
    face_value,
    units,
    call_price,
    put_price,
    s_vol,
    pi,
):
    """Perform backward induction for CB pricing using numba acceleration."""
    # Initialize arrays
    equity_values = np.zeros((N + 1, N + 1, N + 1))
    debt_values = np.zeros((N + 1, N + 1, N + 1))
    market_values = np.zeros((N + 1, N + 1, N + 1))
    conv_values = np.zeros((N + 1, N + 1, N + 1))
    bond_values = np.zeros((N + 1, N + 1, N + 1))
    cb_values = np.zeros((N + 1, N + 1, N + 1))

    # Calculate up and down factors for stock
    u_s = np.exp(s_vol * np.sqrt(dt))
    d_s = 1 / u_s
    book_value = face_value * units

    # Terminal values
    for i in range(N + 1):  # rate nodes
        for j in range(N + 1):  # stock nodes
            if j <= i:  # Only valid stock nodes at each time step
                # Conversion value (Stock value)
                conv_values[N, i, j] = stock_tree[N, j] * conv_ratio
                if conv_values[N, i, j] < book_value:
                    conv_values[N, i, j] = 0

                # Market value(equity + debt)
                equity_values[N, i, j] = conv_values[N, i, j]

                if conv_values[N, i, j] > 0:
                    debt_values[N, i, j] = 0
                else:
                    debt_values[N, i, j] = book_value

                market_values[N, i, j] = equity_values[N, i, j] + debt_values[N, i, j]

                # Bond value(min(Market value, Call value))
                bond_values[N, i, j] = min(market_values[N, i, j], call_price * units)

                # CB value (max(Bond value, Conversion value))
                cb_values[N, i, j] = max(bond_values[N, i, j], conv_values[N, i, j])

    # Backward induction
    for n in range(N - 1, -1, -1):
        for i in range(n + 1):  # rate nodes at time n
            for j in range(n + 1):  # stock nodes at time n
                # Current stock price and rate
                stock_price = stock_tree[n, j]
                rf = rf_tree[n, i]
                p = (np.exp(rf * dt) - d_s) / (u_s - d_s)

                # Conversion values
                conv_values[n, i, j] = stock_price * conv_ratio
                if conv_values[n, i, j] < book_value:
                    conv_values[n, i, j] = 0

                # Market values
                equity_values[n, i, j] = (
                    p
                    * pi
                    * equity_values[n + 1, i + 1, j + 1]
                    * np.exp(-rf_tree[n + 1, i + 1] * dt)
                    + (1 - p)
                    * pi
                    * equity_values[n + 1, i + 1, j]
                    * np.exp(-rf_tree[n + 1, i + 1] * dt)
                    + p
                    * (1 - pi)
                    * equity_values[n + 1, i, j + 1]
                    * np.exp(-rf_tree[n + 1, i])
                    + (1 - p)
                    * (1 - pi)
                    * equity_values[n + 1, i, j]
                    * np.exp(-rf_tree[n + 1, i])
                )
                debt_values[n, i, j] = pi * book_value * np.exp(
                    -rf_tree[n + 1, i + 1] * dt
                ) + (1 - pi) * book_value * np.exp(-rf_tree[n + 1, i] * dt)

                # Bond value
                market_values[n, i, j] = equity_values[n, i, j] + debt_values[n, i, j]

                if n >= 91:  # Repurchase
                    bond_values[n, i, j] = min(
                        market_values[n, i, j], call_price * units
                    )
                else:
                    bond_values[n, i, j] = market_values[n, i, j]

                if n >= 1095:  # Reverse Repurchase
                    bond_values[n, i, j] = max(bond_values[n, i, j], put_price * units)

                # CB value
                cb_values[n, i, j] = max(bond_values[n, i, j], conv_values[n, i, j])

    return cb_values, equity_values, debt_values, bond_values, conv_values


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

        self.rf_tree = build_rf_tree_numba(r0, r_vol, self.dt, N)
        self.stock_tree = build_stock_tree_numba(S_0, s_vol, self.dt, N)

    # CB pricing function using numba-optimized kernels
    def price(self, verbose=False):
        """Main pricing function using numba acceleration."""

        logger.info("Starting valuation...")
        start_time = time.time()

        # Use numba-optimized backward induction
        cb_values, equity_values, debt_values, bond_values, conv_values = (
            price_backward_induction_numba(
                self.N,
                self.dt,
                self.rf_tree,
                self.stock_tree,
                self.conv_ratio,
                self.face_value,
                self.units,
                self.call_price,
                self.put_price,
                self.s_vol,
                self.pi,
            )
        )

        # Log verbose output if requested
        if verbose:
            # Terminal values logging
            for i in range(self.N + 1):
                for j in range(self.N + 1):
                    if j <= i:
                        logger.info(
                            f"n={self.N} i={i} j={j} | S={self.stock_tree[self.N, j]:.2f} r={self.rf_tree[self.N, i]:.2%} CB={cb_values[self.N, i, j]:.2f} Bond={bond_values[self.N, i, j]:.2f} Conv={conv_values[self.N, i, j]:.2f}"
                        )

            # Backward induction logging
            for n in range(self.N - 1, -1, -1):
                logger.info(f"\nTime Step n = {n}:")
                logger.info("=" * 85)
                for i in range(n + 1):
                    for j in range(n + 1):
                        stock_price = self.stock_tree[n, j]
                        rf = self.rf_tree[n, i]
                        logger.info(
                            f"n={n} i={i} j={j} | S={stock_price:.2f} r={rf:.2%} CB={cb_values[n, i, j]:.2f} B={bond_values[n, i, j] * self.units:.2f} Conv={conv_values[n, i, j]:.2f}"
                        )

        end_time = time.time()
        logger.info(f"Valuation calculation time: {end_time - start_time:.4f} seconds")

        # Print the final CB value
        logger.info(f"Final CB Value at root node: {cb_values[0, 0, 0]:.4f}")

        return (
            cb_values[0, 0, 0],
            equity_values,
            debt_values,
            cb_values,
            bond_values,
            conv_values,
        )

    def price_full(self, verbose=False):
        """Full pricing function that returns additional data for visualization."""

        # Get the basic pricing results
        (
            cb_price,
            equity_values,
            debt_values,
            cb_values_raw,
            bond_values,
            conv_values,
        ) = self.price(verbose)

        # Create arrays for visualization
        cb_values = cb_values_raw
        exe_values = np.zeros_like(cb_values)
        hold_values = np.zeros_like(cb_values)
        decisions = np.zeros_like(cb_values, dtype=bool)

        # Determine exercise vs hold decisions
        for n in range(self.N + 1):
            for i in range(n + 1):
                for j in range(n + 1):
                    if j <= i:
                        # Exercise value is the conversion value
                        exe_values[n, i, j] = conv_values[n, i, j]
                        # Hold value is the bond value
                        hold_values[n, i, j] = bond_values[n, i, j]
                        # Decision: True if exercise is optimal
                        decisions[n, i, j] = conv_values[n, i, j] > bond_values[n, i, j]

        return cb_values, exe_values, hold_values, decisions


# ======= HvPlot helpers =======
def create_binomial_tree_plot(tree, title, fmt_value, save_path=None):
    """Create an interactive binomial tree visualization using hvplot/networkx."""
    N = tree.shape[0] - 1
    G = nx.DiGraph()

    # Create nodes and edges
    node_labels = {}
    pos = {}

    for n in range(N + 1):
        for j in range(n + 1):
            node_id = f"n{n}_j{j}"
            y_pos = j - n / 2.0
            pos[node_id] = (n, y_pos)
            node_labels[node_id] = fmt_value(tree[n, j])
            G.add_node(node_id, value=tree[n, j], label=node_labels[node_id])

            # Add edges to next time step
            if n < N:
                # Up edge
                up_node = f"n{n + 1}_j{j + 1}"
                G.add_edge(node_id, up_node)
                # Down edge
                down_node = f"n{n + 1}_j{j}"
                G.add_edge(node_id, down_node)

    # Create the plot using hvplot
    plot = hvnx.draw(
        G,
        pos=pos,
        node_color="lightblue",
        node_size=300,
        labels="label",
        title=title,
        width=800,
        height=600,
        arrowhead_length=0.01,
        directed=True,
    ).opts(
        xlabel="Time Step (n)",
        ylabel="Node Index",
        toolbar="above",
        tools=["hover", "box_zoom", "pan", "reset", "save"],
    )

    # Save if path provided
    if save_path:
        pn.pane.HoloViews(plot).save(save_path)
        logger.info(f"Plot saved to {save_path}")

    return plot


def create_cb_tree_plot(cb_vals, decisions, title, rate_index_fn=None, save_path=None):
    """
    Create an interactive CB tree visualization using hvplot.
    rate_index_fn: function n -> i (choose the rate state at time n). Default = mid (n//2).
    """
    N = cb_vals.shape[0] - 1
    if rate_index_fn is None:

        def rate_index_fn(n):
            return n // 2  # mid-rate slice

    # Prepare data for plotting
    data = []
    for n in range(N + 1):
        i = rate_index_fn(n)
        for j in range(n + 1):
            y_pos = j - n / 2.0
            v = cb_vals[n, i, j]
            decision = "Exercise" if decisions[n, i, j] else "Hold"
            data.append(
                {
                    "time_step": n,
                    "node_index": j,
                    "y_position": y_pos,
                    "cb_value": v,
                    "decision": decision,
                    "label": f"{v:.2f} ({decision})",
                }
            )

    df = pl.DataFrame(data)

    # Create scatter plot with hvplot
    plot = df.hvplot.scatter(
        x="time_step",
        y="y_position",
        color="decision",
        size=100,
        hover_cols=["cb_value", "decision", "node_index"],
        title=title + " — slice on rate state i = floor(n/2)",
        xlabel="Time Step (n)",
        ylabel="Stock Node Index",
        width=900,
        height=600,
        legend="top_right",
    ).opts(tools=["hover", "box_zoom", "pan", "reset", "save"], toolbar="above")

    # Save if path provided
    if save_path:
        pn.pane.HoloViews(plot).save(save_path)
        logger.info(f"Plot saved to {save_path}")

    return plot


def create_dashboard(pricer, cb_vals, decisions, output_dir="plots"):
    """Create an interactive dashboard with all plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create individual plots
    stock_plot = create_binomial_tree_plot(
        pricer.stock_tree,
        "Stock Price Binomial Tree",
        lambda v: f"{v:.2f}",
        save_path=output_dir / "stock_tree.html",
    )

    rf_plot = create_binomial_tree_plot(
        pricer.rf_tree,
        "Risk-Free Rate Tree",
        lambda v: f"{v:.3%}",
        save_path=output_dir / "rf_tree.html",
    )

    cb_plot = create_cb_tree_plot(
        cb_vals,
        decisions,
        "Convertible Bond Price Tree",
        save_path=output_dir / "cb_tree.html",
    )

    # Create dashboard
    dashboard = pn.template.FastListTemplate(
        title="Convertible Bond Pricer Dashboard",
        sidebar=[
            f"# CB Pricer Results\n\n"
            f"**Initial Stock Price**: ${pricer.S_0:.2f}\n\n"
            f"**Strike Price**: ${pricer.K:.2f}\n\n"
            f"**Time to Maturity**: {pricer.T:.2f} years\n\n"
            f"**Time Steps**: {pricer.N}\n\n"
            f"**CB Value**: ${cb_vals[0, 0, 0]:.4f}"
        ],
        main=[
            pn.Column(
                "## Stock Price Tree",
                stock_plot,
                "## Risk-Free Rate Tree",
                rf_plot,
                "## Convertible Bond Price Tree",
                cb_plot,
            )
        ],
    )

    # Save dashboard
    dashboard_path = output_dir / "dashboard.html"
    dashboard.save(dashboard_path)
    logger.info(f"Dashboard saved to {dashboard_path}")

    return dashboard


# ======= Run & Plot =======
if __name__ == "__main__":
    # Create a pricer instance and run the valuation
    pricer = CB_Pricer(**PARAMS)

    logger.info("=" * 30)
    cb_price, equity_values, debt_values, market_values, bond_values, conv_values = (
        pricer.price(verbose=False)
    )
    logger.info(f"Calculated Convertible Bond Price (for N={PARAMS['N']}):")
    logger.info(f"  - CB Value: {cb_price:.4f}")
    logger.info("=" * 30)

    logger.info("=" * 30)
    # Get full pricing results
    cb_vals, exe_vals, hold_vals, decisions = pricer.price_full(verbose=False)
    logger.info("=" * 30)

    # Create and save interactive dashboard
    dashboard = create_dashboard(pricer, cb_vals, decisions, output_dir="plots")

    logger.info("All plots have been generated and saved to the ./plots directory.")
    logger.info(
        "Open plots/dashboard.html in a web browser to view the interactive dashboard."
    )
