import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
import matplotlib.pyplot as plt

# ==== 构造数据 ====
X = np.random.uniform(-2, 2, (200, 2))
y = np.sin(X[:, 0]) + X[:, 1] ** 2

# ==== 拟合符号回归模型 ====
est = SymbolicRegressor(
    population_size=500,
    generations=20,
    stopping_criteria=1e-4,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=1.0,
    verbose=1,
    function_set=['add', 'sub', 'mul', 'div', 'sin', 'sqrt'],
    parsimony_coefficient=0.01,
    random_state=42
)
est.fit(X, y)

# ==== 输出公式 ====
print(f"Discovered Equation:\n{est._program}")

# ==== 可视化 ====
pred = est.predict(X)
plt.scatter(y, pred, s=10)
plt.xlabel("True y")
plt.ylabel("Predicted y")
plt.title("GP Symbolic Regression")
plt.grid(True)
plt.savefig("gp_symbolic_regression.png")