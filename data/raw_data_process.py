import scipy.io as sio
import skimage.io

# Convert raw '.tif' to '.mat'
image_path = 'C:\\Users\\Kikki\\Archive\\dataset\\DC_mall\\dc.tif'
image = skimage.io.imread(image_path)

print(image)
sio.savemat('C:\\Users\\Kikki\\Archive\\dataset\\DC_mall\\dc.mat', {'image': image})
