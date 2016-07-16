
# coding: utf-8

# # EXPLORATORY DATA ANALYSIS   
# Notebook containing my exploratory data analysis for the UCI Activities of Daily Living (ADL) data set.  Due to the number of plots generated, most will be output to file and compiled in a slide pack that will accompany this notebook.

# In[ ]:

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic(u'matplotlib inline')


# In[ ]:

# Load in the csv saved from the File Extraction notebook that contains the accelerometer files for each activity
dataset = pd.read_csv("../Data/MasterFileDF.csv")


# In[ ]:

dataset.head()


# In[ ]:

# Inspect how many data sets we have for each activity
dataset.count()


# Based upon the number of datasets for 'eat_meat' and 'eat_soup', there would be insufficient samples to create a training and test set and therefore should be removed.  We could use these later in the model validation to see whether the model identifies them as an unknown activity. For the remainder of the exploratory data analysis, we'll keep all activities.

# ## Data Manipulation
# Process the individual files in organise them for easy use during the feature engineering and model building phases

# In[ ]:

# Define a function that ingests the UCI txt file, and converts each x,y,z stream to 'g' based upon the process
# contained in MANUAL.txt
""""
Acceleration data recorded in the dataset are coded according to the following mapping:
	[0; +63] = [-1.5g; +1.5g]
The conversion rule to extract the real acceleration value from the coded value is the following:
	real_val = -1.5g + (coded_val/63)*3g
"""""
def convertAccel(filepath):
    # open the file
    df = pd.read_csv(filepath,sep=' ',names = ['x','y','z'],header=None)
    # convert each value using equation
    df = df.applymap(lambda x: -1.5 + (x/63.0)*3)
    # output the file as filename-converted.txt
    return df
    


# In[ ]:

# function to creates a list of tuples with format (filename,[accelerometer data],activity)

def createTuple(filename):
    filepath = '../Raw_Data/' + str(filename)  # create full file path to location of file in Data folder
    df = convertAccel(filepath)  # return a 3 column dataframe of the converted x,y,z accelerations
    activity = str(filename).split("-")[7]  # extract the activity name from the filename string
    output = (filename,(df.x,df.y,df.z),activity)
    return output
    


# In[ ]:

# function that loops through a dataframe of filenames and returns a list of tuples, one tuple for each filename
def createDataList(fileDataframe):
    dataList = []  # initialise a new empty list to fill with tuples
    for ADL in fileDataframe.columns:  # loop through each activity column
        for x in range(0,fileDataframe[ADL].count()): # loop through each filename that is a string (stops at NaN)
            summary = createTuple(fileDataframe[ADL][x])  # create the filename summary tuple
            dataList.append(summary)
    return dataList


# In[ ]:

# create a list of tuples, where each tuple is an activity file and the accelerometer data has been converted to g's
dataList = createDataList(dataset)


# In[ ]:

# Output an example of one of the tuples
dataList[1]


# The accelerometer data is taken from a sensor unit mounted on the right hand of the subject with the x,y,z axes as shown in the image below

# ![](../image.jpg)

# ## Feature Generation

# In[ ]:

# function to calculate the root mean square
def rms(x):   
    return np.sqrt(x.dot(x)/x.size) 


# In[ ]:

# function to create all the model features; loop through each tuple in list
def createFeatures(activityTuple):
    
    features = pd.DataFrame(index = range(0,len(activityTuple)),columns = ['x_max','y_max','z_max','tot_max',                                                                            'x_min','y_min','z_min','tot_min',                                                                           'x_mean','y_mean','z_mean','tot_mean',                                                                           'x_rms','y_rms','z_rms','tot_rms','totTime'                                                                           ,'activity'])
    for row in range(0,len(activityTuple)):
        x,y,z = activityTuple[row][1]
        activity = activityTuple[row][2]
        totAccel = (x**2 + y**2 + z**2)**0.5
        features['x_max'][row] = float(np.max(x))
        features['y_max'][row] = np.max(y)
        features['z_max'][row] = np.max(z)
        features['tot_max'][row] = np.max(totAccel)
        features['x_min'][row] = np.min(x)
        features['y_min'][row] = np.min(y)
        features['z_min'][row] = np.min(z)
        features['tot_min'][row] = np.min(totAccel)
        features['x_mean'][row] = np.mean(x)
        features['y_mean'][row] = np.mean(y)
        features['z_mean'][row] = np.mean(z)
        features['tot_mean'][row] = np.mean(totAccel)
        features['x_rms'][row] = rms(x)
        features['y_rms'][row] = rms(y)
        features['z_rms'][row] = rms(z)
        features['tot_rms'][row] = rms(totAccel)
        features['totTime'][row] = len(x)*(1/32.0)  # The activity length in seconds is no. samples * 1/sampling rate
        features['activity'][row] = str(activity)
        
    return features


# In[ ]:

# Generate a dataframe of features
featuresDF = createFeatures(dataList)


# In[ ]:

featuresDF.head()


# In[ ]:

featuresDF.dtypes


# In[ ]:

# The features dataframe has been initialised as all objects whereas all columns other than activity should be float.  
# Convert to correct data type so can create plots in exploratory analysis
col_names = featuresDF.columns
col_names = col_names[0:(len(col_names)-1)]  # drop activity from col_names
featuresDF[col_names] = featuresDF[col_names].astype(float)
featuresDF.activity = featuresDF.activity.astype(str)


# In[ ]:

featuresDF.dtypes


# ## Generate Plots of Features and Raw Time Series Data

# In[ ]:

# Loop through the activities and generate X,Y,Z acceleration time series plots for each activity
plt.ioff()
for activity in dataset.columns:
    dataList = []  # initialise a new empty list to fill with tuples
    dfX = pd.DataFrame()
    dfY = pd.DataFrame()
    dfZ = pd.DataFrame()
    for filename in dataset[activity].dropna():
        output = createTuple(filename)     
        dataList.append(output)
    for row in range(0,len(dataList)):
        x,y,z = dataList[row][1]
        colNameX = "X"+ str(row)
        colNameY = "Y"+ str(row)
        colNameZ = "Z"+ str(row)
        # Don't know why setting the column names in the dataframe initialisation isn't working
        # interim fix, can come back later and correct this with more time to review
        x_list = pd.DataFrame(x, dtype=float)
        x_list.columns = [colNameX]
        dfX = pd.concat([dfX,x_list],axis = 1)
        y_list = pd.DataFrame(y, dtype=float)
        y_list.columns = [colNameY]
        dfY = pd.concat([dfY,y_list],axis = 1)
        z_list = pd.DataFrame(z, dtype=float)
        z_list.columns = [colNameZ]
        dfZ = pd.concat([dfZ,z_list],axis = 1)
    
    #sns.set_style("white")
    sns.set(font_scale=2)
    plt.figure(figsize=(18.5,10.5))
    plt.plot(np.arange(0,len(dfX)*(1/32.0),(1/32.0)),dfX)
    sns.axlabel('Seconds', 'Acceleration [g]')
    title = activity + " x accel"
    plt.title(title)
    plt.savefig("../Plots/Time/"+title +".png",dpi=200,facecolor='white')
    plt.close()
    plt.figure(figsize=(18.5,10.5))
    plt.plot(np.arange(0,len(dfY)*(1/32.0),(1/32.0)), dfY)
    sns.axlabel('Seconds', 'Acceleration [g]')
    title = activity + " y accel"
    plt.title(title)
    plt.savefig("../Plots/Time/"+title +".png",dpi=200)
    plt.close()
    plt.figure(figsize=(18.5,10.5))
    plt.plot(np.arange(0,len(dfZ)*(1/32.0),(1/32.0)), dfZ)
    sns.axlabel('Seconds', 'Acceleration [g]')
    title = activity + " z accel"
    plt.title(title)
    plt.savefig("../Plots/Time/"+title +".png",dpi=200)
    plt.close()

plt.ion()


# In[ ]:

# Generate Boxplots for each feature, grouped by activity and output to .png files in the Plots/Box/ folder
plt.ioff()
sns.set()
sns.set(style="whitegrid")
sns.set_context("talk")
for name in col_names:
    g = sns.boxplot(x="activity", y=name, data=featuresDF)
    plt.xticks(rotation=90) 
    g.set_title('boxplot of ' + name)
    g.set_ylabel("Acceleration [g]")
    plt.tight_layout()
    plt.savefig("../Plots/Box/"+str(name) +".png",dpi=200)
    plt.close()
plt.ion()    


# In[ ]:

# Plot a seaborn pairplot for relationship between features.  Save in Plots/Pair/ folder
sns.set()
sns.set(style="whitegrid")
sns.set_context("poster")
g = sns.pairplot(featuresDF)
g.fig.title('Scatter Plot of Basic Features') 
g.set(xticklabels=[])
g.set(yticklabels=[])
plt.savefig("../Plots/Pair/"+"FeaturePairPlot.png",dpi=200)


# In[ ]:

# For each feature, print a histogram for each activity for True/False (is/is not) that activity to see how 
# much the data overlaps and whether there are any good features that show good seperation between the two sets
# save in Plots/Histo/ folder
df = featuresDF

plt.ioff()
sns.set()
sns.set(style="white")
sns.set_context("talk")
for feature in col_names:
    for activity in df['activity'].unique():
        fig,ax = plt.subplots()
        fmax = int(df[feature][df['activity']==activity].max())+1
        fmin = int(df[feature][df['activity']==activity].min())-1
        binsize = np.arange(fmin,fmax,((fmax-fmin)/50.0))
        sns.distplot(df[feature][df['activity']==activity],ax=ax,kde=False,label=('is '+ str(activity)),bins=binsize,                     color='b',norm_hist=True)
        sns.distplot(df[feature][df['activity']!=activity],ax=ax,kde=False,label=('is not '+ str(activity)),                     bins=binsize,color='r',norm_hist=True)
        plt.title("Histogram of " + str(feature))
        plt.legend(frameon=True)
        plt.savefig("../Plots/Histo/"+ str(activity)+"-"+str(feature)+".png",dpi=200)
        plt.close()        
plt.ion()


# In[ ]:

featuresDF.count()


# In[ ]:



