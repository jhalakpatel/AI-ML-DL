from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
style.use('fivethirtyeight')

def create_dataset(num_data_point, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(num_data_point):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


# simple best bit line using the slopes and intercept - simple python based implementation - simple way to return value in python - no need to worry about the data type
def best_fit_line_and_intercept(xs, ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) / 
           ((mean(xs)*mean(xs)) - mean(xs*xs)) )
    b = mean(ys) - m * mean(xs)
    return m, b

# simple function to calculate squared error
def squared_error(y_orig, ys_line):
    return sum((ys_line-y_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

#xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
#ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)
xs, ys = create_dataset(40, 80, 2, correlation='neg')
m, b = best_fit_line_and_intercept(xs, ys)
regression_line = [(m*x) + b for x in xs]
r_squared = coefficient_of_determination(ys, regression_line) 
print('Error : ', r_squared)
plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()
