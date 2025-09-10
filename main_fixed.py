# cb_pricing_fixed.py
# Fixed version with proper memory allocation and verbose logging

import logging
import time

import numpy as np
from numba import njit
from numba.typed import List

# Configure logging
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

file_handler = logging.FileHandler("cb_pricer_fixed.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
logger = logging.getLogger(__name__)

# Variables
PARAMS = {
    "S_0": 24.35,
    "K": 35.2,
    "conv_ratio": 2.840,
    "face_value": 100,
    "units": 1.000,
    "call_price": 100,
    "put_price": 100,
    "T": 4.42739726,
    "N": 1040,
    "r0": 0.10,
    "r_vol": 0.10,
    "s_vol": 0.20,
    "pi": 0.5,
}


# ================================================================================
# Original computational kernels (unchanged)
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


@njit(cache=True)
def get_value(arr, n, i, j):
    """Safe getter for 3D array values."""
    if n < len(arr) and i <= n and j <= n:
        # Map to proper index in triangular storage
        return arr[n][i * (n + 1) + j]
    return 0.0


@njit(cache=True)
def set_value(arr, n, i, j, value):
    """Safe setter for 3D array values."""
    if n < len(arr) and i <= n and j <= n:
        # Map to proper index in triangular storage
        arr[n][i * (n + 1) + j] = value


@njit(cache=True)
def create_triangular_arrays(N):
    """Create triangular arrays for storage."""
    equity_values = List()
    debt_values = List()
    market_values = List()
    conv_values = List()
    bond_values = List()
    cb_values = List()

    for n in range(N + 1):
        size = (n + 1) * (n + 1)
        equity_values.append(np.zeros(size, dtype=np.float64))
        debt_values.append(np.zeros(size, dtype=np.float64))
        market_values.append(np.zeros(size, dtype=np.float64))
        conv_values.append(np.zeros(size, dtype=np.float64))
        bond_values.append(np.zeros(size, dtype=np.float64))
        cb_values.append(np.zeros(size, dtype=np.float64))

    return (
        equity_values,
        debt_values,
        market_values,
        conv_values,
        bond_values,
        cb_values,
    )


@njit(parallel=True, cache=True)
def price_backward_induction_numba_fixed(
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
    """Fixed backward induction with proper memory allocation."""

    # Create triangular arrays
    equity_values, debt_values, market_values, conv_values, bond_values, cb_values = (
        create_triangular_arrays(N)
    )

    # Calculate up and down factors for stock
    u_s = np.exp(s_vol * np.sqrt(dt))
    d_s = 1 / u_s
    book_value = face_value * units

    # Terminal values at time N
    for i in range(N + 1):  # rate nodes
        for j in range(N + 1):  # stock nodes
            # Conversion value
            conv_val = stock_tree[N, j] * conv_ratio
            if conv_val < book_value:
                conv_val = 0
            set_value(conv_values, N, i, j, conv_val)

            # Market value
            if conv_val > 0:
                set_value(equity_values, N, i, j, conv_val)
                set_value(debt_values, N, i, j, 0)
            else:
                set_value(equity_values, N, i, j, 0)
                set_value(debt_values, N, i, j, book_value)

            equity_val = get_value(equity_values, N, i, j)
            debt_val = get_value(debt_values, N, i, j)
            market_val = equity_val + debt_val
            set_value(market_values, N, i, j, market_val)

            # Bond value
            bond_val = min(market_val, call_price * units)
            set_value(bond_values, N, i, j, bond_val)

            # CB value
            cb_val = max(bond_val, conv_val)
            set_value(cb_values, N, i, j, cb_val)

    # Backward induction
    for n in range(N - 1, -1, -1):
        for i in range(n + 1):  # Only iterate valid rate nodes
            for j in range(n + 1):  # Only iterate valid stock nodes
                # Current stock price and rate
                stock_price = stock_tree[n, j]
                rf = rf_tree[n, i]
                p = (np.exp(rf * dt) - d_s) / (u_s - d_s)

                # Conversion value
                conv_val = stock_price * conv_ratio
                if conv_val < book_value:
                    conv_val = 0
                set_value(conv_values, n, i, j, conv_val)

                # Market values from next time step
                i_up = min(i + 1, n + 1)
                j_up = min(j + 1, n + 1)

                rf_up = rf_tree[n + 1, i_up] if i_up <= n + 1 else rf
                rf_same = rf_tree[n + 1, i] if i <= n + 1 else rf

                # Get values from next time step
                eq_uu = get_value(equity_values, n + 1, i_up, j_up)
                eq_ud = get_value(equity_values, n + 1, i_up, j)
                eq_du = get_value(equity_values, n + 1, i, j_up)
                eq_dd = get_value(equity_values, n + 1, i, j)

                equity_val = (
                    p * pi * eq_uu * np.exp(-rf_up * dt)
                    + (1 - p) * pi * eq_ud * np.exp(-rf_up * dt)
                    + p * (1 - pi) * eq_du * np.exp(-rf_same * dt)
                    + (1 - p) * (1 - pi) * eq_dd * np.exp(-rf_same * dt)
                )
                set_value(equity_values, n, i, j, equity_val)

                debt_val = pi * book_value * np.exp(-rf_up * dt) + (
                    1 - pi
                ) * book_value * np.exp(-rf_same * dt)
                set_value(debt_values, n, i, j, debt_val)

                # Bond value
                market_val = equity_val + debt_val
                set_value(market_values, n, i, j, market_val)

                if n >= 91:  # Repurchase
                    bond_val = min(market_val, call_price * units)
                else:
                    bond_val = market_val

                if n >= 1095:  # Reverse Repurchase
                    bond_val = max(bond_val, put_price * units)

                set_value(bond_values, n, i, j, bond_val)

                # CB value
                cb_val = max(bond_val, conv_val)
                set_value(cb_values, n, i, j, cb_val)

    return (
        get_value(cb_values, 0, 0, 0),
        equity_values,
        debt_values,
        cb_values,
        bond_values,
        conv_values,
    )


# ================================================================================
# Fixed CB Pricer class
# ================================================================================
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

    def price(self, verbose=False):
        """Main pricing function using fixed memory allocation."""
        logger.info("Starting valuation with fixed memory allocation...")

        # Calculate actual memory usage
        total_elements = sum((n + 1) ** 2 for n in range(self.N + 1))
        memory_gb = 6 * total_elements * 8 / 1e9
        original_memory_gb = 6 * (self.N + 1) ** 3 * 8 / 1e9
        logger.info(
            f"Actual memory usage: ~{memory_gb:.2f} GB (vs {original_memory_gb:.2f} GB in original)"
        )
        logger.info(
            f"Memory savings: {(1 - memory_gb / original_memory_gb) * 100:.1f}%"
        )

        start_time = time.time()

        (
            cb_value,
            equity_values,
            debt_values,
            cb_values_arr,
            bond_values,
            conv_values,
        ) = price_backward_induction_numba_fixed(
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

        # Log verbose output if requested
        if verbose:
            # Terminal values logging
            logger.info(f"\nTerminal Values at Time Step n = {self.N}:")
            logger.info("=" * 85)
            for i in range(self.N + 1):
                for j in range(self.N + 1):
                    if j <= i:
                        cb_val = get_value(cb_values_arr, self.N, i, j)
                        bond_val = get_value(bond_values, self.N, i, j)
                        conv_val = get_value(conv_values, self.N, i, j)
                        logger.info(
                            f"n={self.N} i={i} j={j} | S={self.stock_tree[self.N, j]:.2f} "
                            f"r={self.rf_tree[self.N, i]:.2%} CB={cb_val:.2f} "
                            f"Bond={bond_val:.2f} Conv={conv_val:.2f}"
                        )

            # Backward induction logging
            for n in range(self.N - 1, -1, -1):
                logger.info(f"\nTime Step n = {n}:")
                logger.info("=" * 85)
                for i in range(n + 1):
                    for j in range(n + 1):
                        stock_price = self.stock_tree[n, j]
                        rf = self.rf_tree[n, i]
                        cb_val = get_value(cb_values_arr, n, i, j)
                        bond_val = get_value(bond_values, n, i, j)
                        conv_val = get_value(conv_values, n, i, j)
                        logger.info(
                            f"n={n} i={i} j={j} | S={stock_price:.2f} r={rf:.2%} "
                            f"CB={cb_val:.2f} B={bond_val:.2f} Conv={conv_val:.2f}"
                        )

        end_time = time.time()
        logger.info(f"Valuation calculation time: {end_time - start_time:.4f} seconds")
        logger.info(f"Final CB Value at root node: {cb_value:.4f}")

        return (
            cb_value,
            equity_values,
            debt_values,
            cb_values_arr,
            bond_values,
            conv_values,
        )


# ======= Run =======
if __name__ == "__main__":
    pricer = CB_Pricer(**PARAMS)

    logger.info("=" * 60)
    logger.info("Running Fixed Memory CB Pricer")
    logger.info(f"Parameters: N={PARAMS['N']}, T={PARAMS['T']:.4f} years")
    logger.info("=" * 60)

    cb_price, _, _, _, _, _ = pricer.price(verbose=True)

    logger.info("=" * 60)
    logger.info(f"Calculated Convertible Bond Price: {cb_price:.4f}")
    logger.info("=" * 60)
