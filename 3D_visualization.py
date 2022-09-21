import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
import os
import cv2

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')


red_list_t = []
red_list_col = []
red_list_row = []
blue_list_t = []
blue_list_col = []
blue_list_row = []
for i in range(1286, 1306):
	t = i - 1286
	print(t)
	# filename = os.path.join('inputs', 'video3', '00' + str(i), 'modified', '0_0.png')
	filename = os.path.join('I-ES4LFV', 'video3', '00' + str(i) + '.png')
	image = cv2.imread(filename)


	for row in range(image.shape[0]):
		for col in range(image.shape[1]):
			pixel = image[row, col]

			if pixel[0] == 255:
				red_list_t.append(t)
				red_list_row.append(row)
				red_list_col.append(col)

			elif pixel[2] == 255:
				blue_list_t.append(t)
				blue_list_row.append(row)
				blue_list_col.append(col)


red_list_t = np.array(red_list_t)
red_list_col = np.array(red_list_col)
red_list_row = np.array(red_list_row)
blue_list_t = np.array(blue_list_t)
blue_list_col = np.array(blue_list_col)
blue_list_row = np.array(blue_list_row)

ax.scatter(red_list_t, red_list_col, red_list_row, c='r',s=1)
ax.scatter(blue_list_t, blue_list_col, blue_list_row, c='b',s=1)



# ax.set_title('Scene 2')

# Set axes label
ax.set_xlabel('t', labelpad=20)
ax.set_ylabel('x', labelpad=20)
ax.set_zlabel('y', labelpad=20)

plt.show()