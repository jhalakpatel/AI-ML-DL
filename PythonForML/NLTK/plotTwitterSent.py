import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
style.use('ggplot')

fig = plt.figure()
axl = fig.add_subplot(1, 1, 1)

def animate(i):
	pullData = open('twitter-out.txt', 'r').read()
	lines = pullData.split('\n')
	xarr = []
	yarr = []

	x = 0
	y = 0

	for l in lines:
		x += 1
		if 'pos' in l:
			y += 1
		elif 'neg' in l:
			y-=1

		# update the xarr and yarr
		xarr.append(x)
		yarr.append(y)

	axl.clear()
	axl.plot(xarr, yarr)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()