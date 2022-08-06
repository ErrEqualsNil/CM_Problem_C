import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("data/无插层.csv")
achieve_distance = np.array(data['接收距离(cm)']).reshape(-1, 1)
hot_wind_speed = np.array(data['热风速度(r/min)']).reshape(-1, 1)
thickness = np.array(data['厚度mm']).reshape(-1, 1)
porosity = np.array(data['孔隙率（%）']).reshape(-1, 1)
compression_resilience = np.array(data["压缩回弹性率（%）"]).reshape(-1, 1)
air_permeability = np.array(data['透气性 mm/s']).reshape(-1, 1)
filtration_resistance = np.array(data['过滤阻力Pa']).reshape(-1, 1)
filtration_efficiency = np.array(data['过滤效率（%）']).reshape(-1, 1)

# 工艺参数与厚度的关系
plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")

ax.plot(achieve_distance.reshape(-1), hot_wind_speed.reshape(-1), thickness.reshape(-1), 'g.', label="original data")

model = LinearRegression()
x = np.concatenate([achieve_distance, hot_wind_speed, hot_wind_speed ** 2], axis=1)
model.fit(x, thickness)
pred_y = model.predict(x)
print(f"filtration_resistance = {model.coef_[0][0]} * achieve_distance + {model.coef_[0][1]} * hot_wind_speed + {model.coef_[0][2]} * hot_wind_speed^2 + {model.intercept_[0]}")

loss = mean_squared_error(thickness, pred_y)
print(f"loss: {loss}")

ax.plot_trisurf(achieve_distance.reshape(-1), hot_wind_speed.reshape(-1), pred_y.reshape(-1), cmap='Blues', label="Regression Model")
ax.set_xlabel("achieve_distance(cm)")
ax.set_ylabel("hot_wind_speed(r/min)")
ax.set_zlabel("thickness(mm)")
ax.set_title("achieve_distance & hot_wind_speed - thickness Regression Result (No-Interlayer)")
ax.view_init(10, 30)
plt.savefig("results/achieve_distance-&-hot_wind_speed-thickness-No_Interlayer.png")

# 过滤阻力与厚度的关系
# plt.figure(figsize=(10, 10))
# model = LinearRegression()
# model.fit(thickness, filtration_resistance)
# pred_y = model.predict(thickness)
#
# outliers_idx = []
# for i in range(len(thickness)):
#     threshold = 10  # 自定的
#     if abs(pred_y[i] - filtration_resistance[i]) > threshold:
#         outliers_idx.append(i)
#
# plt.plot(thickness[outliers_idx], filtration_resistance[outliers_idx], 'r.', label="outliers")
# thickness = np.delete(thickness, outliers_idx).reshape(-1, 1)
# filtration_resistance = np.delete(filtration_resistance, outliers_idx).reshape(-1, 1)
# plt.plot(thickness, filtration_resistance, 'g.', label="original data")
#
# model = LinearRegression()
# model.fit(thickness, filtration_resistance)
# pred_y = model.predict(thickness)
# print(f"filtration_resistance = {model.coef_[0]} * thickness + {model.intercept_[0]}")
#
# loss = mean_squared_error(filtration_resistance, pred_y)
# print(f"loss: {loss}")
#
# sorted_x = np.sort(thickness, axis=0)
# sorted_pred_y = model.predict(sorted_x)
# plt.plot(sorted_x, sorted_pred_y, label="Regression Model")
# plt.xlabel("thickness(mm)")
# plt.ylabel("filtration_resistance(Pa)")
# plt.legend()
# plt.title("thickness - filtration_resistance Regression Result (No-Interlayer)")
# plt.savefig("results/thickness-filtration_resistance-No_interlayer.png")


# 过滤阻力与过滤效率的关系
# plt.figure(figsize=(10, 10))
# model = LinearRegression()
# model.fit(filtration_resistance, filtration_efficiency)
# pred_y = model.predict(filtration_resistance)
#
# outliers_idx = []
# for i in range(len(filtration_resistance)):
#     threshold = 10  # 自定的
#     if abs(pred_y[i] - filtration_efficiency[i]) > threshold:
#         outliers_idx.append(i)
#
# plt.plot(filtration_resistance[outliers_idx], filtration_efficiency[outliers_idx], 'r.', label="outliers")
# filtration_resistance = np.delete(filtration_resistance, outliers_idx).reshape(-1, 1)
# filtration_efficiency = np.delete(filtration_efficiency, outliers_idx).reshape(-1, 1)
# plt.plot(filtration_resistance, filtration_efficiency, 'g.', label="original data")
#
# model = LinearRegression()
# model.fit(filtration_resistance, filtration_efficiency)
# pred_y = model.predict(filtration_resistance)
# print(f"filtration_efficiency = {model.coef_[0]} * filtration_resistance + {model.intercept_[0]}")
#
# loss = mean_squared_error(filtration_efficiency, pred_y)
# print(f"loss: {loss}")
#
# sorted_x = np.sort(filtration_resistance, axis=0)
# sorted_pred_y = model.predict(sorted_x)
# plt.plot(sorted_x, sorted_pred_y, label="Regression Model")
# plt.xlabel("filtration_resistance(Pa)")
# plt.ylabel("filtration_efficiency(%)")
# plt.legend()
# plt.title("filtration_resistance - filtration_efficiency Regression Result (No-Interlayer)")
# plt.savefig("results/filtration_resistance-filtration_efficiency-No_interlayer.png")

#  厚度与孔隙率的关系
# plt.figure(figsize=(10, 10))
# transform = PolynomialFeatures(degree=2)
# transform.fit(thickness)
# x = transform.transform(thickness)
# model = LinearRegression()
# model.fit(x, porosity)
# pred_y = model.predict(x)
#
# outliers_idx = []
# for i in range(len(thickness)):
#     threshold = 3  # 自定的
#     if abs(pred_y[i] - porosity[i]) > threshold:
#         outliers_idx.append(i)
#
# for idx in outliers_idx:
#     plt.plot(thickness[idx], porosity[idx], 'r.', label="outliers")
# thickness = np.delete(thickness, outliers_idx).reshape(-1, 1)
# porosity = np.delete(porosity, outliers_idx).reshape(-1, 1)
# plt.plot(thickness, porosity, 'g.', label="original data")
#
# model = LinearRegression()
# transform = PolynomialFeatures(degree=2)
# transform.fit(thickness)
# x = transform.transform(thickness)
# model.fit(x, porosity)
# pred_y = model.predict(x)
# print(f"porosity = {model.coef_[0][2]} * thickness^2 + {model.coef_[0][1]} * thickness + {model.intercept_[0]}")
#
# loss = mean_squared_error(porosity, pred_y)
# print(f"loss: {loss}")
#
# sorted_x = np.sort(thickness, axis=0)
# transform = PolynomialFeatures(degree=2)
# transform.fit(sorted_x)
# log_sorted_x = transform.transform(sorted_x)
# sorted_pred_y = model.predict(log_sorted_x)
# plt.plot(sorted_x, sorted_pred_y, label="Regression Model")
# plt.xlabel("thickness(mm)")
# plt.ylabel("porosity(%)")
# plt.legend()
# plt.title("Thickness - Porosity Regression Result (Interlayer)")
# plt.savefig("results/Thickness_Porosity_Interlayer.png")

