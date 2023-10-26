import numpy as np
#don't know if it is optimized, still serves its purpose, maybe you could save a draw
counter = 0
iterations = 100000

for i in range(0,iterations) :
  draw1 = np.random.randint(1,4) #winning door
  draw2 = np.random.randint(1,4) #door chosen by contestant
  draw3 = np.random.randint(1,4) #loosing door eliminated
  while draw2 == draw3 | draw1 == draw3 : #assuring that you actually eliminate a loosing door
      draw3 = np.random.randint(1,4)
  if draw2 == draw1 :
      counter += 1
print((counter/iterations)-(1/3))