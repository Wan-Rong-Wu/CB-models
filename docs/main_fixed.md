# 記憶體優化的核心原理

## 問題分析：原始 3D 陣列的浪費

原始 `main.py` 使用 `(N+1, N+1, N+1)` 的 3D 陣列，但實際上：

- 時間步 `n=0`：只需要 `1×1 = 1` 個值
- 時間步 `n=500`：只需要 `501×501 = 251,001` 個值
- 時間步 `n=1040`：需要 `1041×1041 = 1,082,881` 個值

**實際使用 vs 分配的記憶體：**

原始分配：`N³ = 1041³ ≈ 11.3` 億個元素
實際需要：`Σ(n+1)²` from `n=0` to `N ≈ N³/3 ≈ 3.8` 億個元素
浪費率：`66%`

## 解決方案：三角形儲存結構

### 關鍵函數實作細節

1. `create_triangular_arrays` - 動態記憶體分配

   ```python
   @njit(cache=True)
   def create_triangular_arrays(N):
       """為每個時間步分配精確大小的陣列"""
       arrays = List()
       for n in range(N + 1):
           size = (n + 1) * (n + 1) # 時間步 n 需要 (n+1)×(n+1) 的空間
           arrays.append(np.zeros(size, dtype=np.float64))
       return arrays
   ```

   **改動原理：**

   - 每個時間步獨立分配記憶體
   - 使用 1D 陣列模擬 2D（利率 × 股價）空間
   - Numba List 確保 JIT 編譯相容

2. `get_value / set_value` - 索引映射

   ```python
   @njit(cache=True)
   def get_value(arr, n, i, j):
       """將 3D 索引 (n,i,j) 映射到三角形儲存結構"""
       if n < len(arr) and i <= n and j <= n:
           # 2D 到 1D 的映射：row * width + col
           return return arr[n][i * (n + 1) + j]
       return 0.0
   ```

   **索引映射公式：**

   - 3D 座標 (n, i, j) → 陣列 `arr[n]` 的第 `i*(n+1)+j` 個元素
   - 邊界檢查防止越界存取

3. `price_backward_induction_numba_fixed` - 主要改動

   **原始版本：**

   ```python
   # 分配巨大的 3D 陣列
   equity_values = np.zeros((N+1, N+1, N+1))  # 54GB!
   # 直接存取
   equity_values[n, i, j] = value
   ```

   **優化版本：**

   ```python
   # 分配三角形結構
   equity_values = create_triangular_arrays(N)  # 18GB
   # 透過映射函數存取
   set_value(equity_values, n, i, j, value)
   ```

   **關鍵改動點：**

   1. 終端值計算（第 143-170 行）：保持相同邏輯，只改變存取方式
   2. 向後歸納（第 173-229 行）：

   - 只迭代有效節點 `range(n+1)`
   - 使用 `get_value / set_value` 存取資料
   - 邊界處理確保不越界

### 驗證準確性的方法

1. 與原始版本對比

   ```python
   # 執行對比測試
   import subprocess

   # 運行原始版本（較小的 N）
   N_test = 100 # 使用較小的 N 避免記憶體問題
   result_original = run_original_with_N(N_test)
   result_fixed = run_fixed_with_N(N_test)

   # 驗證結果一致性
   assert abs(result_original - result_fixed) < 1e-10
   ```

2. 收斂性測試

   ```python
   # 測試不同 N 值的收斂
   N_values = [50, 100, 200, 400, 800]
   results = []

   for N in N_values:
   price = run_pricer_with_N(N)
   results.append(price)

   # 檢查收斂趨勢
   for i in range(1, len(results)):
   diff = abs(results[i] - results[i-1])
   print(f"N={N_values[i]}: Price={results[i]:.4f},
   Δ={diff:.6f}")
   ```

3. 邊界條件測試

   ```python
   # 測試極端情況
   test_cases = [
   {"S_0": 100, "K": 1}, # 深度價內
   {"S_0": 1, "K": 100}, # 深度價外
   {"T": 0.01}, # 接近到期
   {"s_vol": 0.01}, # 低波動率
   {"s_vol": 1.0}, # 高波動率
   ]
   ```

4. 基準測試對比

   ```python
   # 與已知解析解或市場價格對比
   # 例如：當轉換價值遠高於債券價值時，CB 應接近轉換價值
   if conv_ratio _ S_0 >> face_value:
   expected ≈ conv_ratio _ S_0
   assert abs(cb_price - expected) / expected < 0.01
   ```

5. 單元測試

   ```python
   # 測試關鍵函數
   def test_index_mapping():
       """確保索引映射正確"""
       arr = create_triangular_arrays(10)
       set_value(arr, 5, 3, 2, 42.0)
       assert get_value(arr, 5, 3, 2) == 42.0

   def test_boundary_conditions():
       """測試邊界處理"""
       arr = create_triangular_arrays(10) # 不應崩潰
       value = get_value(arr, 15, 10, 10)
       assert value == 0.0
   ```

## 建議的完整驗證流程

1. 先用小 `N` 值（如 `N=100`）比對兩版本結果
2. 逐步增加 `N`，觀察收斂性
3. 檢查 `verbose` 輸出的中間值是否合理
4. 與市場數據或其他定價工具對比

這樣的優化保持了算法的數學正確性，只是改變了資料的儲存方式，因此理
論上應該產生完全相同的結果。
