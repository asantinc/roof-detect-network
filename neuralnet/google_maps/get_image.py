from cStringIO import StringIO
import numpy as np
from scipy import misc, ndimage
from PIL import Image
import urllib



def pull_satellite_imgs(latitude=1.054759, longitude=32.466644, num=20, increase=True):
	for i in range(num):
		url = "http://maps.googleapis.com/maps/api/staticmap?center={0},{1}&size=800x800&zoom=19&maptype=satellite&format=jpg".format(latitude, longitude)
		buffer = StringIO(urllib.urlopen(url).read())
		image = Image.open(buffer)
		img = np.array(image)
		misc.imsave('lat{0}_long{1}.jpg'.format(latitude, longitude), img)
		if increase:
			latitude += 0.01
			longitude -= 0.01
		else:
			latitude -= 0.01
			longitude -= 0.01


if __name__ == '__main__':
	pull_satellite_imgs(latitude=1.054759, longitude=32.466644, num=20, increase=True)
	pull_satellite_imgs(latitude=0.328277, longitude=32.409728, num=20, increase=False)
