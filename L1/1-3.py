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

print(f'x.shape: {x.shape}') # もちろん2つとも(points_per_round * num_rounds,)です。
print(f'y.shape: {y.shape}')
print(f'x.max(): {x.max()}') # 最大値を確認できます。当たり前ですが、1.0は超えません。
print(f'y.max(): {y.max()}')
print(f'x.min(): {x.min()}') # 最小値を確認できます。-1.0は下回りません。
print(f'y.min(): {y.min()}')