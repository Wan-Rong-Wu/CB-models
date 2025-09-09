# Convertible Bond Pricer

A convertible bond pricing tool using binomial tree methods with Numba acceleration and interactive visualizations.

## Features

- **Binomial Tree Pricing Model**: Implements a comprehensive binomial tree approach for convertible bond valuation
- **Numba Acceleration**: Leverages JIT compilation for significant performance improvements in computational kernels
- **Interactive Visualizations**: Creates dynamic, browser-based plots using HvPlot and Panel
- **Comprehensive Logging**: Detailed logging system for debugging and tracking calculations
- **Flexible Parameters**: Easily configurable bond parameters including conversion ratio, call/put prices, and volatility

## Installation

### Prerequisites

- Python 3.12 or higher
- uv package manager (recommended) or pip

### Using uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd cb_pricer

# Install dependencies using uv
uv sync
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd cb_pricer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install from pyproject.toml:

```bash
pip install .
```

## Usage

### Basic Usage

Run the pricer with default parameters:

```bash
uv run python main.py
```

Or if using pip:

```bash
python main.py
```

### Output

The script generates:

1. **Console Output**: Calculation results and timing information
2. **Log File**: Detailed calculations saved to `cb_pricer.log`
3. **Interactive Plots**: HTML files saved in the `plots/` directory:
   - `stock_tree.html`: Interactive stock price binomial tree
   - `rf_tree.html`: Risk-free rate tree visualization
   - `cb_tree.html`: Convertible bond price tree with exercise decisions
   - `dashboard.html`: Combined dashboard with all visualizations

### Viewing Results

Open the dashboard in your web browser:

```bash
open plots/dashboard.html  # On macOS
# Or
xdg-open plots/dashboard.html  # On Linux
# Or
start plots/dashboard.html  # On Windows
```

### Modifying Parameters

Edit the `PARAMS` dictionary in `main.py` to customize:

```python
PARAMS = {
    "S_0": 24.35,      # Initial stock price
    "K": 35.2,         # Strike price for stock
    "conv_ratio": 2.840,  # Conversion ratio
    "face_value": 100,    # Face value of the bond
    "units": 1.000,       # Units of bond
    "call_price": 100,    # Bond call price
    "put_price": 100,     # Bond put price
    "T": 4.42739726,      # Time to maturity (years)
    "N": 52,              # Number of time steps
    "r0": 0.10,           # Initial risk-free rate
    "r_vol": 0.10,        # Interest rate volatility
    "s_vol": 0.20,        # Stock price volatility
    "pi": 0.5,            # Probability parameter for rf
}
```

### Verbose Mode

Enable detailed calculation logging:

```python
# In main.py, modify the price() call
cb_price, equity_values, debt_values, market_values, bond_values, conv_values = (
    pricer.price(verbose=True)  # Set to True for detailed output
)
```

## Performance Optimization

The pricer uses Numba JIT compilation for critical computational functions:

- `build_rf_tree_numba()`: Constructs risk-free rate tree
- `build_stock_tree_numba()`: Builds stock price tree
- `price_backward_induction_numba()`: Performs backward induction pricing

These optimizations provide significant speedup compared to pure Python implementations.

## Logging Configuration

Logs are configured with timestamps and multiple output handlers:

- **Console**: INFO level messages
- **File** (`cb_pricer.log`): All log levels including DEBUG

To adjust logging level:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more detail
    ...
)
```

## Project Structure

```bash
cb_pricer/
- main.py              # Main pricing script
- pyproject.toml       # Project dependencies
- README.md           # This file
- cb_pricer.log       # Generated log file
- plots/              # Generated visualization directory
    - stock_tree.html
    - rf_tree.html
    - cb_tree.html
    - dashboard.html
```

## Dependencies

- **numba**: JIT compilation for performance
- **numpy**: Numerical computations
- **hvplot**: Interactive plotting
- **panel**: Dashboard creation
- **networkx**: Graph structures for tree visualization
- **bokeh**: Plotting backend
- **polars**: Data manipulation

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
uv sync --refresh
```

### Numba Warnings

First-time execution may show Numba compilation warnings. These are normal and subsequent runs will be faster.

### Plot Display Issues

If plots don't display correctly, ensure you have a modern web browser with JavaScript enabled.

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]
