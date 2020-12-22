import struct
import colorsys
from PIL import Image

frame_s = 135
frame_e = 200

width = 1024
height = 1024

for f in range(frame_s, frame_e):
	print("Frame %d" % f)	
	
	fp = open("out/%d.bin" % f, "rb")

	data = []
	for i in fp.read():
		if i == 0:
			data.append( (0, 0, 0) )
		else:
			data.append( tuple(map(lambda a : int(a*255), colorsys.hsv_to_rgb(i/255, 1, 1))) )

	img = Image.new("RGB", (width*3, height*3))

	img.putdata(data)
	
	img = img.resize((width, height), Image.BICUBIC)

	img = img.transpose(Image.FLIP_TOP_BOTTOM)

	img.save("anim/%d.png" % f)
