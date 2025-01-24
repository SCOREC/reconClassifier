import postgkyl as pg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


path='reconClassifier-data/gem_challenge/rt-5m-gem_elc_1.bp'
gdata = pg.GData(path)
values = gdata.get_values()
print(type(values),values.shape)
print(values)
print("Max value:", values.max())
print("Min value:", values.min())
threshold = 0
bit_field = (values < threshold).astype(int)
print(bit_field)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

coords_1 = np.where(bit_field == 1)
coords_0 = np.where(bit_field == 0)

ax.scatter(coords_1[0], coords_1[1], coords_1[2], c='red', marker='.', label='1')
ax.scatter(coords_0[0], coords_0[1], coords_0[2], c='blue', marker='.', label='0')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()
plt.savefig("my_plot.png")
