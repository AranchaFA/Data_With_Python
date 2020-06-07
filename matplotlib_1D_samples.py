import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Changing style to look similar to R ggplot2 package
plt.style.use('ggplot')

# VARIABLE DECLARATION (they are data arrays, and functions on that data)
x: np.ndarray = np.linspace(0, 20, 1000) # 1000 evenly spaced values from 0 to 20
x2 = np.linspace(0, 20, 100)
print(type(x))
print(type(x2)) # Type is recognized both we declare typed var or not
cos_x = np.cos(x)
sin_x2 = np.sin(x2)
tan_x = np.tan(x)

# OPTION 1: SHOWING QUICK AND SIMPLE PLOT (we can use directly plt.plot() without create figure and axis)
#plt.plot(x, x*2, x, x*4)
plt.plot(x2, x2**3, label='x**3')
plt.plot(x2, x2**2, marker='.', color='grey', label='x**2')
plt.legend() # This .legend() affects only plots above

# OPTION 2: CREATING FIGURE AND AXIS (basic elements for all plots)
figure = plt.figure() # 'Empty' constructor, but we can add some parameters to customize its appearance
axis = plt.axes() # 'Empty', same as above

# ASSIGN VARIABLE-FUNCTION TO AXIS created above
# LINE PLOT
'''
    Representing 3 functions at a time (they will have same visual features):
    y1(x) = 2x
    y2(x) = 4x
    y3(2x) = 16x
'''
axis.plot(x, 2*x, x, 4*x, 2*x, 16*x, linewidth=1.2, ls='--', label='f(x)')
axis.plot(x, cos_x, linewidth=1.2, color="yellow", label='cos(x)')
axis.plot(x, tan_x, color="pink", label='tan(x)')
# SCATTER PLOT
axis.scatter(x2, sin_x2, marker='^', color="green", label='sin(x)')
# We can plot directly from pandas DataFrame and Series
serie = pd.Series(np.random.random(20)*20)
data_frame = pd.DataFrame(np.random.random(50)*20)
# HISTOGRAM
data_frame.plot(kind='hist', bins=20, color="orange", label='Hist') # Label isn't changed :( ?¿
'''
# BAR PLOT
plt.figure() # To show bar plot in a new figure, remove if we want hist+bar in the same graph
serie.plot(kind='bar', color='violet', label='Bar')
plt.legend(fontsize=5)
# BOX PLOT
plt.style.use('classic')
data_frame.plot(kind='box', label='Box')
plt.legend(fontsize=20)
'''
# POLIGON filled plot
plt.figure()
plt.fill(x2, sin_x2, color='darkgreen')

# CUSTOMIZE PLOT APPEARANCE
# axis.method() -> Acts on figure object plots
# plt.method() -> Acts on last graph (polygonal if Bar and Box plots are comented)
# Tittle
axis.set_title('Trigonometric (and more) functions', color='b', fontsize=10)
plt.title('Histogram plot', color='b', fontsize=10)
# axis.set(attrname = attrvalue, attrname2 = ...) or axis.set_attrname(attrvalue) or plt.attrname(attrvalue)
# Axis limits (to show tan(x) (infinite limits function) convenient, we have to limit plot axis)
axis.margins(x=0.0, y=0.5) # x=0% y=50% of graphic rise
axis.set(xlim=[0, 22], ylim=[-3, 7])
# Axis labels
axis.set_xlabel('x')
axis.set_ylabel('y(x)')
plt.ylabel('Frequency')
plt.xlabel('Value')
# Show legend (labels for each function)
axis.legend(fontsize=7)
plt.legend(fontsize=12)

# SAVING plot as image file
figure.savefig('trig.png') # Save figure's plot
plt.savefig('sample_plot.png') # Save last graph (polygonal if Bar and Box plots are comented)

# SHOW plot
plt.show() # All graphs and their plots are showed

# If we save plot image here, file will be created AFTER close show window and it will be EMPTY
#plt.savefig('trigomometric_plot.png')
