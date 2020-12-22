from PIL import Image

frames = 88

imgs = []

for f in range(0, frames):
	imgs.append( Image.open("anim/%d.png" % f) )

imgs[0].save("out.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0)
