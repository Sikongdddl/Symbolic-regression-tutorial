import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline

# 1. 生成数据：y = 3x + 2x^2 + 0.5 + 噪声
np.random.seed(0)
n_samples = 100
X = np.sort(np.random.rand(n_samples, 1) * 2 - 1, axis=0)  # x ∈ [-1, 1]
true_y = 0.5 + 3 * X + 2 * X**2
noise = np.random.normal(0, 0.2, size=true_y.shape)
y = true_y + noise

# 2. 构造模型：2阶多项式 + LassoCV（自动调节 alpha）
model = make_pipeline(PolynomialFeatures(degree=2, include_bias=True),
                      LassoCV(cv=5))
model.fit(X, y)

# 3. 可视化拟合效果
X_plot = np.linspace(-1, 1, 100).reshape(-1, 1)
y_pred = model.predict(X_plot)

plt.scatter(X, y, color='blue', label='Noisy observations')
plt.plot(X_plot, y_pred, color='red', label='Lasso regression fit')
plt.title('Lasso Polynomial Regression (Symbolic Regression)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('lasso_polynomial_regression.png')

# 4. 打印出模型表达式（即符号形式）
lasso = model.named_steps['lassocv']
poly = model.named_steps['polynomialfeatures']
feature_names = poly.get_feature_names_out(['x'])

print("Recovered symbolic expression:")
terms = []
for name, coef in zip(feature_names, lasso.coef_):
    if abs(coef) > 1e-4:
        terms.append(f"({coef:.3f})*{name}")
if abs(lasso.intercept_) > 1e-4:
    terms.insert(0, f"{lasso.intercept_:.3f}")
print("y = " + " + ".join(terms))