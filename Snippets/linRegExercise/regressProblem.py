# CMSC 352 Exercise
# Univariate regression using vectors and numpy.

from random import *
from numpy import *
import numpy.random

from pylab import *
#import matplotlib.animation as animation

def readDat(fname):
    '''
    Read rows having two points per row, (x, y).  Return array of data.
    fname: name of input file
    '''
    fd = open(fname,'r')
    dat = []
    for line in fd.readlines():
        if line[0] == '#':
            continue
        x = line.split()
        x.insert(0,'1.0') #prepend a 1.0 to the list
        vals = [float(val) for val in x]
        dat.append( vals ) # convert to floats and append
    fd.close()
    return array(dat) # returns a numpy array, not a list


def gradStep(dat,p):
    '''
    Calculate one update of the parameters in p via gradient descent
    algorithm.
    :param dat: all input data
    :param p: parameters of linear equation to be adjusted
    : return : estimate of parameters
    '''
    sum = zeros(2) # sum of changes for the two parameters
    # TODO:
    # You must complete the gradient step: dJ/dp = (px - y) * x.
    # You should loop over all data items, adding to
    # sum maintained for the parameters you use.  Don't forget the bias term.
    # return the two parameters averaged over the size of the training data.

    return sum


def cost(dat,p):
    '''Cost function (sum of squares of differences)'''

    sum = 0
    for x in dat:
        sum += ( dot(x[0:2], p) - x[2] )**2
    return sum / (2.0 * len(dat))

def regress(dat,limit=0.001):
    '''Use linear regression to predict col2 from col1 in dat.
    :param dat: all data
    :param limit: maximum bound on error
    :returns : parameters and errors, both as lists
    '''
    p = (numpy.random.rand(2) - 0.5) # two random values ((these are the parameters to determine)
    np = p.copy()
    alpha = 1.0e-4 # choose this value carefully.

    err = cost(dat,p)
    print(err)
    olderr = 2 * err
    paramsList = []
    errorList = []
    while olderr - err > limit: # terminate when error change is small
        dp = gradStep(dat,p) # calculate gradient step
        np += -alpha * dp # add change to parameters
        p = np.copy()

        olderr = err
        err = cost(dat,p) # determine cost (error)
        paramsList.append(p)
        errorList.append(err)
        print ("Error %f PreviousError %f" % (err,olderr))

    return paramsList,errorList



dat = readDat('oldfaithLtd.m')
fig = figure()
errorPlot = fig.add_subplot(2,1,1) # plot for error
linePlot = fig.add_subplot(2,1,2) # plot for data
linePlot.plot(dat[:,1],dat[:,2],'yo')
plt.show()

# perform regression to specified limit
params,error = regress(dat,1.0e-10)
# plot error
errorPlot.plot(error,'r')
