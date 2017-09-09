import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [3, 7, 4]

x2 = [1, 2, 3]
y2 = [10, 14, 22]

plt.plot(x, y, label="First line")
plt.plot(x2, y2, label="second line")
plt.xlabel('Plot Number')
plt.ylabel('Variable')
plt.title('Interesting graph\n Check this out')
plt.legend()
plt.show()
