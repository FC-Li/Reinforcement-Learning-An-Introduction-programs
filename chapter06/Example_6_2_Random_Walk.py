#!coding: utf-8
import numpy as np 
from matplotlib import pyplot as plt 

# 到达一个状态的奖励
reward = {x: 0.0 for x in "ZABCDE"}
reward["Y"] = 1

def TD(episodes, alpha=0.1, gamma=1):
    value = {x: 0.5 for x in "ABCDE"}
    value["Z"] = 0.0
    value["Y"] = 0.0
    for _ in range(episodes):
        end = False
        current_state = "C"
        # A episode
        while not end:
            current_state_index = "ZABCDEY".index(current_state)
            next_state = np.random.choice(["ZABCDEY"[current_state_index+1], 
                "ZABCDEY"[current_state_index-1]])
            value[current_state] = value[current_state] + alpha * (reward[next_state] + gamma * value[next_state] - value[current_state])
            if next_state == "Y" or next_state == "Z":
                end = True
            else:
                current_state = next_state
    return value

def MC(episodes, alpha=0.1):
    value = {x: 0.5 for x in "ABCDE"}
    value["Z"] = 0.0
    value["Y"] = 0.0
    for _ in range(episodes):
        end = False
        current_state = "C"
        # A episode
        state_sequence = [current_state]
        while not end:
            current_state_index = "ZABCDEY".index(current_state)
            next_state = np.random.choice(["ZABCDEY"[current_state_index+1], 
                "ZABCDEY"[current_state_index-1]])
            state_sequence.append(next_state)
            if next_state == "Y" or next_state == "Z":
                for index, state in enumerate(state_sequence[:-1]):
                    # 由于在此问题中只有到达 Y 状态才会有奖励，即中间的奖励都是0，又 gamma = 1，所以 G_t 的求解
                    # 并不需要累计折扣求和，只需要看一下终止状态，根据终止状态得到 G_t。
                    G_t = reward[next_state]
                    value[state] = value[state] + alpha * (G_t - value[state])
                end = True
            else:
                current_state = next_state
    return value

def RMSE(value):
    target_value = {
    "A": 1/6.0,
    "B": 2/6.0,
    "C": 3/6.0,
    "D": 4/6.0,
    "E": 5/6.0,
    }
    se = 0
    for x in "ABCDE":
        se += (target_value[x] - value[x])**2
    mse = se / 5
    return np.sqrt(mse)

for alpha in [0.01, 0.02, 0.03]:
    rmse_mc = []
    rmse_td = []
    for episodes in range(1, 100, 1):
        rmse_mc_ = 0
        rmse_td_ = 0
        for _ in range(100):
            rmse_mc_ += RMSE(MC(episodes, alpha=alpha))
            rmse_td_ += RMSE(TD(episodes, alpha=alpha))

        rmse_mc.append(rmse_mc_ / 100)
        rmse_td.append(rmse_td_ / 100)
    plt.plot(rmse_mc, label="mc %s" % alpha)
    plt.plot(rmse_td, label="td %s" % alpha)
plt.xlabel("episodes")
plt.ylabel("rmse")
plt.legend()
plt.show()