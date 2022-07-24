from turtle import update
import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from PLResults import update_player_ratings

#make dummy data
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
y=np.random.rand(len(x))
plt.figure()
plt.plot(scatter_variables_x() ,scatter_variables_y())
plt.bar(x, y, alpha=0.2)
plt.title(f"Top 10 Premier League Teams 17/18")
plt.xlabel("Teams")
plt.ylabel("Rating")
plt.xticks(x, [str(i) for i in y], rotation=90)

#set parameters for tick labels
plt.tick_params(axis='x', which='major', labelsize=3)

plt.tight_layout()