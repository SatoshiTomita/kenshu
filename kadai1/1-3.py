import numpy as np
import matplotlib.pyplot as plt

data = np.array([[1,2,3], [4,5,6], [7,8,9]]) # ただの一例です

min_value = 0
max_value = 2 * np.pi
points_per_round = 100 # １周あたり何個の点のデータにするか
num_rounds = 5 # 何周分のデータにするか
# (points_per_round * num_rounds) 個のデータになる
theta = np.linspace(min_value, max_value * num_rounds, points_per_round * num_rounds)

print(theta) # データの中身がそのまま出てきます
print(theta.shape) # (points_per_round * num_rounds,) (上記の設定で書いたら250)

x = np.cos(theta)
y = np.sin(theta*2)

