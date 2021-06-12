import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pylab
pylab.rcParams['figure.figsize'] = (8, 4)
import seaborn as sns
from collections import OrderedDict

# Function to build synthetic data
def sample(rSeed, periodLength, colNames):

    np.random.seed(rSeed)
    date = pd.to_datetime("1st of Dec, 1999")   
    cols = OrderedDict()

    for col in colNames:
        cols[col] = np.random.normal(loc=0.0, scale=1.0, size=periodLength)
    dates = date+pd.to_timedelta(np.arange(periodLength), 'D')

    df = pd.DataFrame(cols, index = dates)
    return(df)

# Dataframe with synthetic data
df = sample(rSeed = 123, colNames = ['X1', 'X2'], periodLength = 50)

# sns.distplot with multiple layers
for var in list(df):
    #myPlot = sns.distplot(df[var])
    myPlot = sns.histplot(df[var])
plt.show()
plt.close()
data = []
for idx, var in enumerate(list(df)):
    import ipdb; ipdb.set_trace()
    lines2D = [obj for obj in myPlot.findobj() if str(type(obj)) == "<class 'matplotlib.lines.Line2D'>"]
    x, y = lines2D[idx].get_data()[0], lines2D[idx].get_data()[1]
    # Store as dataframe 
    data.append(pd.DataFrame({'x':x, 'y':y}))

plt.plot(data[0].x, data[0].y)
import ipdb; ipdb.set_trace()
plt.show()


'''
data = []
for idx, var in enumerate(list(df)):
    myPlot = sns.distplot(df[var])
    
    
    
    mySpecificPlot = sns.histplot(df[var])

    lines2D = [obj for obj in myPlot.findobj() if str(type(obj)) == "<class 'matplotlib.lines.Line2D'>"]
    dislines2D = [obj for obj in mySpecificPlot.findobj() if str(type(obj)) == "<class 'matplotlib.lines.Line2D'>"]

    import ipdb; ipdb.set_trace()


    #import ipdb; ipdb.set_trace()
    specificmyPlot = mySpecificPlot._axes[0][0]
    # Fine Line2D objects
    lines2D = [obj for obj in myPlot.findobj() if str(type(obj)) == "<class 'matplotlib.lines.Line2D'>"]
    dislines2D = [obj for obj in specificmyPlot.findobj() if str(type(obj)) == "<class 'matplotlib.lines.Line2D'>"]
    #import ipdb; ipdb.set_trace()
    #lines2D = [obj for obj in myPlot.findobj() if str(type(obj)) == "<class 'matplotlib.lines.Line2D'>"]
    
    for ax in mySpecificPlot._axes.flat:
        print(ax.lines)
        for line in ax.lines:
            print(line.get_xdata())
            print(line.get_ydata())
        import ipdb; ipdb.set_trace()


    # Retrieving x, y data
    x, y = lines2D[idx].get_data()[0], lines2D[idx].get_data()[1]
    # Store as dataframe 
    data.append(pd.DataFrame({'x':x, 'y':y}))
    #import ipdb; ipdb.set_trace()
'''
