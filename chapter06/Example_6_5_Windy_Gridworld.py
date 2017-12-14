import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.misc

class Block(object):
    """docstring for Block"""
    def __init__(self, x, y, channel, size=1):
        super(Block, self).__init__()
        self.x = x
        self.y = y
        self.size = size
        self.channel = channel

class GridWorld(object):
    """docstring for GridWorld"""
    def __init__(self, x_size, y_size, target_x, target_y, init_x, init_y, wind_distribution=None):
        super(GridWorld, self).__init__()
        self.x_size, self.y_size = x_size, y_size
        self.target = Block(target_x, target_y, 0)
        self.hero = Agent(init_x, init_y, 1, self)
        self.wind_distribution = wind_distribution

    def render(self):
        canvas = np.ones([self.y_size+2, self.x_size+2, 3])
        canvas[1:-1, 1:-1, :] = 0
        canvas[self.target.x+1:self.target.x+self.target.size+1, 
            self.target.y+1:self.target.y+self.target.size+1, 
            self.target.channel] = 1
        canvas[self.hero.x+1:self.hero.x+self.hero.size+1, 
            self.hero.y+1:self.hero.y+self.hero.size+1, 
            self.hero.channel] = 1
        r = scipy.misc.imresize(canvas[:,:,0], [128, 128, 1], interp="nearest")
        g = scipy.misc.imresize(canvas[:,:,1], [128, 128, 1], interp="nearest")
        b = scipy.misc.imresize(canvas[:,:,2], [128, 128, 1], interp="nearest")
        return np.stack([r, g, b], axis=2)

class Agent(Block):
    """docstring for Agent"""
    def __init__(self, init_x, init_y, channel, gridworld, epsilon=0.1, alpha=0.1, gamma=1):
        super(Agent, self).__init__(init_x, init_y, channel)
        self.gridworld = gridworld
        # self.Q = np.random.randn(self.gridworld.x_size, self.gridworld.y_size, 4)
        self.Q = np.zeros((self.gridworld.x_size, self.gridworld.y_size, 4))
        self.Q[self.gridworld.target.x, self.gridworld.target.y, :] = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.init_x = init_x
        self.init_y = init_y

    def learn(self, episodes, max_step=np.inf):
        for _ in range(episodes):
            step = 0
            terminated = False
            self.trajectory = []
            # self.x = np.random.randint(self.gridworld.x_size)
            # self.y = np.random.randint(self.gridworld.y_size)
            self.x = self.init_x
            self.y = self.init_y
            self.current_state = (self.x, self.y)
            self.action = self.choose_action(self.current_state, True)
            self.trajectory.append(self.current_state)
            while step < max_step and not terminated:
                self.next_state = self.move(self.action)
                self.next_action = self.choose_action(self.next_state, True)
                if self.next_state == (self.gridworld.target.x, self.gridworld.target.y):
                    reward = 1
                    terminated = True
                else:
                    reward = -1
                # 五元组(current_state, action, reward, next_state, next_action)
                self.Q[self.current_state[0], self.current_state[1], self.action] += \
                    self.alpha * (reward + self.gamma * self.Q[self.next_state[0], self.next_state[0], self.next_action] 
                        - self.Q[self.current_state[0], self.current_state[1], self.action])
                step += 1
                self.trajectory.append(self.next_state)
                self.current_state = (self.x, self.y)
                self.action = self.next_action

    def step(self, origin_x, origin_y):
        self.x = origin_x
        self.y = origin_y
        self.current_state = (self.x, self.y)
        self.action = self.choose_action(self.current_state, True)
        self.next_state = self.move(self.action)
        terminated = self.next_state == (self.gridworld.target.x, self.gridworld.target.y)
        return self.next_state, terminated

    def choose_action(self, state, greedy=True):
        if greedy and np.random.rand() < self.epsilon:
            action = np.random.choice([0, 1, 2, 3])
            return action
        action = np.argmax(self.Q[state[0], state[1]])
        return action

    def move(self, direction):
        if self.gridworld.wind_distribution is None:
            wind = 0
        else:
            wind = self.gridworld.wind_distribution[self.y]
        self.x = max(self.x - wind, 0)
        if direction == 0:
            self.y = max(self.y - 1, 0)
        elif direction == 1:
            self.y = min(self.y + 1, self.gridworld.y_size - 1)
        elif direction == 2:
            self.x = max(self.x - 1, 0)
        elif direction == 3:
            self.x = min(self.x + 1, self.gridworld.x_size - 1)
        else:
            pass
        return (self.x, self.y)

gw = GridWorld(10, 10, 6, 7, 6, 0, [1,0,0,0,1,0,0,1,0,0])
gw.hero.learn(100)
next_state, terminated = gw.hero.step(6, 0)

fig = plt.figure()
im = plt.imshow(gw.render(), animated=True)

def updatefig(*args):
    global next_state, terminated
    if not terminated:
        next_state, terminated = gw.hero.step(*next_state)
    im.set_array(gw.render())
    return im,

ani = animation.FuncAnimation(fig, updatefig, frames=100, interval=50, blit=True)
# Set up formatting for the movie files
Writer = animation.writers["ffmpeg"]
writer = Writer(fps=15, metadata=dict(artist="Robert"), bitrate=1800)
ani.save("im.mp4", writer=writer)
# plt.show()
