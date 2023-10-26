#%%
import numpy as np
import matplotlib.pyplot as plt

#this block is an animation for a random walk process,only modify p_l,p_u,steps,and graphics stuff
p_l = 0.5
p_r = 1-p_l
p_u = 0.5
p_d = 1-p_u
simulations = 10

x_hists = [[]] #for animation plotting purposes
y_hists = [[]] 

steps = 1000

for n in range(0,simulations):
 x = 0
 y = 0
 x_hist = [0] #for animation plotting purposes
 y_hist = [0] 
 for i in range(0,steps):
    drawx = np.random.random()
    drawy = np.random.random()
    if drawx <= p_l :
        x += 1
        x_hist.append(x)
    else:
        x -= 1
        x_hist.append(x)
    if drawy <= p_l :
        y += 1
        y_hist.append(y)
    else:
        y -= 1
        y_hist.append(y)
    #plt.xlim(-50,50)
    #plt.ylim(-50,50)
    #plt.scatter(x,y,color="red",marker="x")
    #plt.plot(x_hist, y_hist)
    #plt.pause(0.3)
 x_hists.append(x_hist)
 y_hists.append(y_hist)

plt.xlim(-50,50)
plt.ylim(-50,50)
#plt.scatter(x,y,color="red",marker="x")
for j in range(0,len(x_hists)) :
 plt.plot(x_hists[j], y_hists[j])
    
# %%
