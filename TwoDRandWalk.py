# %%
import numpy as np
import matplotlib.pyplot as plt

# random walk process,only modify p_l,p_u,steps,and graphics stuff

p_l = 0.5
p_r = 1 - p_l
p_u = 0.5
p_d = 1 - p_u
simulations = 10000

x_hists = [[]]  # for animation plotting purposes
y_hists = [[]]

Squaredistances = []
delta_xs = []
delta_ys = []


def GetDelta(hist_array):
    return hist_array[len(hist_array) - 1] - hist_array[0]


def GetSquareDist(x_hist, y_hist):
    dx = GetDelta(x_hist)
    dy = GetDelta(y_hist)
    return dx**2 + dy**2


steps = 1000

for n in range(0, simulations):
    x = 0
    y = 0
    x_hist = [0]  # for animation plotting purposes
    y_hist = [0]
    for i in range(0, steps):
        drawx = np.random.random()
        drawy = np.random.random()
        if drawx <= p_l:
            x += 1
            x_hist.append(x)
        else:
            x -= 1
            x_hist.append(x)
        if drawy <= p_l:
            y += 1
            y_hist.append(y)
        else:
            y -= 1
            y_hist.append(y)
        # plt.xlim(-50,50)
        # plt.ylim(-50,50)
        # plt.scatter(x,y,color="red",marker="x")
        # plt.plot(x_hist, y_hist)
        # plt.pause(0.3)
    x_hists.append(x_hist)
    y_hists.append(y_hist)
    delta_xs.append(GetDelta(x_hist))
    delta_ys.append(GetDelta(y_hist))
    Squaredistances.append(GetSquareDist(x_hist, y_hist))

plt.figure(0)
plt.xlim(-50, 50)
plt.ylim(-50, 50)
# plt.scatter(x,y,color="red",marker="x")
for j in range(0, len(x_hists)):
    plt.plot(x_hists[j], y_hists[j])

plt.figure(1)
plt.hist(delta_xs)
print("mean dx = %5.2f" % (np.mean(x_hist)))
plt.figure(2)
plt.hist(delta_ys)
print("mean dy = %5.2f" % (np.mean(y_hist)))
plt.figure(3)
plt.hist(Squaredistances)
print("mean SD = %5.2f" % (np.mean(Squaredistances)))

# %%
