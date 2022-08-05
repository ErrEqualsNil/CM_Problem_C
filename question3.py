import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("data/有插层.csv")
achieve_distance = np.array(data['接收距离(cm)']).reshape(-1, 1)
hot_wind_speed = np.array(data['热风速度(r/min)']).reshape(-1, 1)
thickness = np.array(data['厚度mm']).reshape(-1, 1)
porosity = np.array(data['孔隙率（%）']).reshape(-1, 1)
compression_resilience = np.array(data["压缩回弹性率（%）"]).reshape(-1, 1)
air_permeability = np.array(data['透气性 mm/s']).reshape(-1, 1)
filtration_resistance = np.array(data['过滤阻力Pa']).reshape(-1, 1)
filtration_efficiency = np.array(data['过滤效率（%）']).reshape(-1, 1)


#  厚度与孔隙率的关系
plt.figure(figsize=(10, 10))
transform = PolynomialFeatures(degree=2)
transform.fit(thickness)
x = transform.transform(thickness)
model = LinearRegression()
model.fit(x, porosity)
pred_y = model.predict(x)

outliers_idx = []
for i in range(len(thickness)):
    threshold = 3  # 自定的
    if abs(pred_y[i] - porosity[i]) > threshold:
        outliers_idx.append(i)

for idx in outliers_idx:
    plt.plot(thickness[idx], porosity[idx], 'r.', label="outliers")
thickness = np.delete(thickness, outliers_idx).reshape(-1, 1)
porosity = np.delete(porosity, outliers_idx).reshape(-1, 1)
plt.plot(thickness, porosity, 'g.', label="original data")

model = LinearRegression()
transform = PolynomialFeatures(degree=2)
transform.fit(thickness)
x = transform.transform(thickness)
model.fit(x, porosity)
pred_y = model.predict(x)
print(f"porosity = {model.coef_[0][2]} * thickness^2 + {model.coef_[0][1]} * thickness + {model.intercept_[0]}")

loss = mean_squared_error(porosity, pred_y)
print(f"loss: {loss}")

sorted_x = np.sort(thickness, axis=0)
transform = PolynomialFeatures(degree=2)
transform.fit(sorted_x)
log_sorted_x = transform.transform(sorted_x)
sorted_pred_y = model.predict(log_sorted_x)
plt.plot(sorted_x, sorted_pred_y, label="Regression Model")
plt.xlabel("thickness(mm)")
plt.ylabel("porosity(%)")
plt.legend()
plt.title("Thickness - Porosity Regression Result (Interlayer)")
plt.savefig("results/Thickness_Porosity_Interlayer.png")
