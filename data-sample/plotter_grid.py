from matplotlib import image
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
# Open the image form working directory
file = '/home/liluan/桌面/DL_Group9/data-sample/grid.jpg'
image = image.imread(file)
# summarize some details about the image
print(image.dtype)
print(image.shape)
# display the array of pixels as an image
pyplot.imshow(image)
pyplot.show()
print()
npimg = np.array(image.data.tolist())
print(type(npimg))
plt.imshow(npimg)
plt.show()