###############################################################################
#####            Draft Script for facial expressions analysis             #####
#####            see (link post)                                          ##### 
#####            (C) 2021 Guillermo Hidalgo Gadea                         #####
###############################################################################

# These are the libraries you will need to have installed in your environment
import math
import tkinter.filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm


# Let us start the analysis by loading the csv file from the pose-3d output of Anipose
file = tkinter.filedialog.askopenfile(title='Select the csv file in the pose-3d directory', mode="r")
data = pd.read_csv(file, header=0)
data.info()

# Apart from the x-, y-, z-coordinates we are interested in, the anipose output also contains other variables we want to discard for now
scores = data.loc[:, data.columns.str.contains('score')]
scores.describe()

scores.hist()

scores.boxplot()

errors = data.loc[:, data.columns.str.contains('error')]
errors.describe()

errors.hist()

errors.boxplot()

# Let's filter the coordinates and ignore the other variables
coords = data.loc[:,~data.columns.str.contains('score|error|ncams|fnum|center|M_')]

# Now we want to define facial expressions in egocentric coordinates relative to the nasal bone 
# Note that the reference point is arbitrary, but the nasal bone is a great reference, as it moves only by head movements and not by facial expressions.
centered_coords = coords.copy()
for i in range(centered_coords.shape[1]):
    if '_x' in centered_coords.columns[i]:
        centered_coords.loc[:,centered_coords.columns[i]] = centered_coords.loc[:,centered_coords.columns[i]].subtract(coords.loc[:,"nose1_x"].values)
    elif '_y' in centered_coords.columns[i]:
        centered_coords.loc[:,centered_coords.columns[i]] = centered_coords.loc[:,centered_coords.columns[i]].subtract(coords.loc[:,"nose1_y"].values)
    elif '_z' in centered_coords.columns[i]:
        centered_coords.loc[:,centered_coords.columns[i]] = centered_coords.loc[:,centered_coords.columns[i]].subtract(coords.loc[:,"nose1_z"].values)
    else:
        pass

emoface_egocentric = centered_coords.to_numpy()


# Additionally to the relative coordinates for facial expression, we also may want to calculate some useful features of head movement
features = centered_coords.copy()

# Head position as coordinates of nasal bone reference
features['position_x'] = coords['nose1_x']
features['position_y'] = coords['nose1_y']
features['position_z'] = coords['nose1_z']

pos_x, = plt.plot(features['position_x'], label='x')
pos_y, = plt.plot(features['position_y'], label='y')
pos_z, = plt.plot(features['position_z'], label='z')
plt.xlabel('Time [frames]')
plt.ylabel('Position [pixel]')
plt.legend()


# Velocity of head movement as frame-to-frame difference in position
features['velocity_x'] = np.append([0],np.diff(features['position_x'], n=1)) 
features['velocity_y'] = np.append([0],np.diff(features['position_y'], n=1))
features['velocity_z'] = np.append([0],np.diff(features['position_z'], n=1))

vel_x, = plt.plot(features['velocity_x'], label='x')
vel_y, = plt.plot(features['velocity_y'], label='y')
vel_z, = plt.plot(features['velocity_z'], label='z')
plt.xlabel('Time [frames]')
plt.ylabel('Velocity [pixel/s]')
plt.legend()


# Acceleration of head movement as frame-to-frame difference in velocity
features['acceleration_x'] = np.append([0],np.diff(features['velocity_x'], n=1))
features['acceleration_y'] = np.append([0],np.diff(features['velocity_y'], n=1))
features['acceleration_z'] = np.append([0],np.diff(features['velocity_z'], n=1))

acc_x, = plt.plot(features['acceleration_x'], label='x')
acc_y, = plt.plot(features['acceleration_y'], label='y')
acc_z, = plt.plot(features['acceleration_z'], label='z')
plt.xlabel('Time [frames]')
plt.ylabel('Acceleration [pixel/s^2]')
plt.legend()


# Now we can train a Hidden markov Model with our features (centered corrdinates, head position, velocity and acceleration)
model1 = hmm.GaussianHMM(n_components = 9, covariance_type="full") # change the number of components you expect to find in your data
model1.fit(features)
pred1 = model1.predict(features)

transition1 = model1.transmat_
transition1
means1 = model1.means_
means1


# Note that we may want to exclude head movement at all and train our HMM with facial expression only
model2 = hmm.GaussianHMM(n_components = 9, covariance_type="full" )# change the number of components you expect to find in your data
model2.fit(coords)
pred2 = model2.predict(coords)

transition2 = model2.transmat_
transition2
means2 = model2.means_
means2


# To better see what just happend we can plot the time series segmentation by HMM predictions
def plot_prediction(data, predictions):
    """
    This function will plot the time series data and mark the transitions between predicted classes.
    
    """
    colors = {"0": "black", "1":"dimgray", "2":"darkgray", "3":"white", "4":"bisque", "5":"tan", "6":"orange", "7":"salmon", "8":"gold", "9":"rosybrown", "10":"beige", "11":"thistle", "12":"peachpuff", "13":"khaki", "14":"skyblue", "15":"lightblue", "16":"lightsteelblue", "17":"lavender", "18":"mediumaquamarine", "19":"cadetblue"}
    n = max(predictions)+1
    name =[x for x in globals() if globals()[x] is data][0]
    yloc = max(np.max(data))-(max(np.max(data)) - min(np.min(data)))/8
    locy = yloc - (max(np.max(data)) - min(np.min(data)))/8
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(data)
    start_pred = 0
    for i in range(len(predictions)):
        if i == len(predictions)-1:
            end_pred = i+1
            ax.axvspan(start_pred,end_pred, facecolor=colors["%d" %predictions[i]], alpha = 0.5)
            loc = start_pred + (end_pred - start_pred)/2
            ax.annotate('%d'%predictions[i], xy=(loc, locy), xytext=(loc+10, yloc),
            arrowprops=dict(arrowstyle="->", facecolor='black'))
        elif predictions[i] == predictions[i+1]:
            pass
        else:
            end_pred = i
            ax.axvspan(start_pred,end_pred, facecolor=colors["%d" %predictions[i]], alpha = 0.5)
            loc = start_pred + (end_pred - start_pred)/2
            ax.annotate('%d'%predictions[i], xy=(loc, locy), xytext=(loc+10, yloc),
            arrowprops=dict(arrowstyle="->", facecolor='black'))
            start_pred = end_pred
        
    plt.xlabel("Time [frames]")
    plt.ylabel("Feature value from %s data" %name)
    plt.title('Hidden Markov Model predictions with N = %d Components' %n)
    plt.show()
    return


plot_prediction(features, pred1)

plot_prediction(coords, pred2)



# Next we will spit the entire time series of our video into each predicted class and calculate the average pose during that segment
def split_data(data, prediction):
    """
    The split_data function will be used to split time series data into smaller 
    chunks by the prediction variable.
    
    """
    n = max(prediction)+1 #read out the number of predicted components
    data['pred'] = prediction
    grouped = data.groupby(data.pred)
    predictions = [grouped.get_group(i) for i in range(n)]
    pose = [predictions[i].mean() for i in range(n)]
    
    return predictions, pose

predictions1, pose1 = split_data(centered_coords, pred1)
predictions2, pose2 = split_data(centered_coords, pred2)


# Now we want to have a look at the average pose durig each predicted class. Because the facial landmarks alone would look a bit sad, we start by defining a facial skeleton to connect coordinates

def face_skeleton(pose):
    """
    The face_skeleton function defines a mesh skeleton by connecting the facial landmarks as defined below.
    This function is directly passed to plot_3Dpose. 

    """
    skeletons = []
    for n in range(len(pose)): # read out n_components from different poses
    
        lefteye = [pose[n]['lefteye1_x'], pose[n]['lefteye2_x']], [pose[n]['lefteye1_y'], pose[n]['lefteye2_y']], [pose[n]['lefteye1_z'], pose[n]['lefteye2_z']]
        righteye = [pose[n]['righteye1_x'], pose[n]['righteye2_x']], [pose[n]['righteye1_y'], pose[n]['righteye2_y']], [pose[n]['righteye1_z'], pose[n]['righteye2_z']]
        leyebrow = [pose[n]['leyebrow1_x'], pose[n]['leyebrow2_x'],pose[n]['leyebrow3_x']],[pose[n]['leyebrow1_y'], pose[n]['leyebrow2_y'],pose[n]['leyebrow3_y']],[pose[n]['leyebrow1_z'], pose[n]['leyebrow2_z'],pose[n]['leyebrow3_z']]
        reyebrow = [pose[n]['reyebrow1_x'], pose[n]['reyebrow2_x'],pose[n]['reyebrow3_x']],[pose[n]['reyebrow1_y'], pose[n]['reyebrow2_y'],pose[n]['reyebrow3_y']],[pose[n]['reyebrow1_z'], pose[n]['reyebrow2_z'],pose[n]['reyebrow3_z']]
        nose = [pose[n]['nose1_x'],pose[n]['nose3_x'],pose[n]['nose2_x'],pose[n]['nose4_x'],pose[n]['nose1_x']],[pose[n]['nose1_y'],pose[n]['nose3_y'],pose[n]['nose2_y'],pose[n]['nose4_y'],pose[n]['nose1_y']],[pose[n]['nose1_z'],pose[n]['nose3_z'],pose[n]['nose2_z'],pose[n]['nose4_z'],pose[n]['nose1_z']]
        lips = [pose[n]['uplip_x'],pose[n]['llip_x'],pose[n]['lowlip_x'],pose[n]['rlip_x'],pose[n]['uplip_x']],[pose[n]['uplip_y'],pose[n]['llip_y'],pose[n]['lowlip_y'],pose[n]['rlip_y'],pose[n]['uplip_y']],[pose[n]['uplip_z'],pose[n]['llip_z'],pose[n]['lowlip_z'],pose[n]['rlip_z'],pose[n]['uplip_z']]
        face = [pose[n]['rear_x'],pose[n]['chin_x'],pose[n]['lear_x']],[pose[n]['rear_y'],pose[n]['chin_y'],pose[n]['lear_y']],[pose[n]['rear_z'],pose[n]['chin_z'],pose[n]['lear_z']]
        
        skeleton = lefteye, righteye, leyebrow, reyebrow, nose, lips, face
        skeletons.append(skeleton)
    
    return skeletons


def plot_3Dpose(pose, elevation, azimuth):
    """
    This plot function takes the average pose coordinates of facial landmarks, creates a skeleton and visualizes the facial expression
    in a 3D coordinate system with predefined elevantion and azimuth angles.

    """
    skeletons = face_skeleton(pose)

    ncols = 3
    nrows = math.ceil(len(pose)/ncols)
    width = ncols*6
    height = nrows *5
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), subplot_kw=dict(projection='3d'))

    for ax, n in zip(axes.flat, range(len(pose))):
            x_points = pose[n][['_x' in s for s in pose[n].index]]
            y_points = pose[n][['_y' in s for s in pose[n].index]]
            z_points = pose[n][['_z' in s for s in pose[n].index]]
            ax.scatter3D(x_points,y_points, z_points)
            ax.view_init(elevation, azimuth)
            ax.set(xlabel='X axis', ylabel='Y axis', zlabel='Z axis')
            ax.set_title('Predicted Pose: %d' %(n+1))
            for i in range(len(skeletons[0])):
                x = skeletons[n][i][0]
                y = skeletons[n][i][1]
                z = skeletons[n][i][2]
                ax.plot(x,y,z, color='g') 
                
    plt.suptitle('Hidden Markov Model predictions with N = %d Components' %len(pose))
    plt.show()
    return

plot_3Dpose(pose1, 11, 280)    
plot_3Dpose(pose2, 11, 280)  


# Now that we know how each facial expression looks like, we could analyze some simple kinematics to describe what actaually happens in each segment, appart from the average pose
def plot_kinematics(predictions, pose):
    """
    This Function will create multiple subplots for every predicted pose and visualize simple kinematics as line plot and histogram. 
    
    """
    ncols = 3
    nrows = math.ceil(len(predictions)/ncols)
    width = ncols*6
    height = nrows *5
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))

    for ax, n in zip(axes.flat, range(len(pose))):
        ax.plot(predictions[n]['position_x'], color = 'g', label = 'pos_x')
        ax.plot(predictions[n]['position_y'], color = 'g', label = 'pos_y')
        ax.plot(predictions[n]['position_z'], color = 'g', label = 'pos_z')
        
        ax.plot(predictions[n]['velocity_x'], color = 'y', label = 'vel_x')
        ax.plot(predictions[n]['velocity_y'], color = 'y', label = 'vel_y')
        ax.plot(predictions[n]['velocity_z'], color = 'y', label = 'vel_z')
        
        ax.plot(predictions[n]['acceleration_x'], color = 'r', label = 'acc_x')
        ax.plot(predictions[n]['acceleration_y'], color = 'r', label = 'acc_y')
        ax.plot(predictions[n]['acceleration_z'], color = 'r', label = 'acc_z')
        
        ax.set(xlabel='Time (frames)', ylabel='Position, Velocity and Acceleration')
        ax.legend()
        ax.set_title('Kinematic Profile in Predicted Class: %d' %n)
                
    plt.suptitle('Hidden Markov Model predictions with N = %d Components' %len(pose))
    plt.show()
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    
    for ax, n in zip(axes.flat, range(len(pose))):
        ax.hist(predictions[n]['position_x'], color = 'g', label = 'x')
        ax.hist(predictions[n]['position_y'], color = 'y', label = 'y')
        ax.hist(predictions[n]['position_z'], color = 'r', label = 'z')
        
        ax.set(xlabel='x, y, z movement', ylabel='frequency')
        ax.legend()
        ax.set_title('Movement in Predicted Class: %d' %n)
                
    plt.suptitle('Hidden Markov Model predictions with N = %d Components' %len(pose))
    plt.show()
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    
    for ax, n in zip(axes.flat, range(len(pose))):
        ax.hist(predictions[n]['velocity_x'], color = 'g', label = 'x')
        ax.hist(predictions[n]['velocity_y'], color = 'y', label = 'y')
        ax.hist(predictions[n]['velocity_z'], color = 'r', label = 'z')
        
        ax.set(xlabel='x, y, z velocity', ylabel='frequency')
        ax.legend()
        ax.set_title('Velocity in Predicted Class: %d' %n)
                
    plt.suptitle('Hidden Markov Model predictions with N = %d Components' %len(pose))
    plt.show()
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    
    for ax, n in zip(axes.flat, range(len(pose))):
        ax.hist(predictions[n]['acceleration_x'], color = 'g', label = 'x')
        ax.hist(predictions[n]['acceleration_y'], color = 'y', label = 'y')
        ax.hist(predictions[n]['acceleration_z'], color = 'r', label = 'z')
        
        ax.set(xlabel='x, y, z acceleration', ylabel='frequency')
        ax.legend()
        ax.set_title('Acceleration in Predicted Class: %d' %n)
                
    plt.suptitle('Hidden Markov Model predictions with N = %d Components' %len(pose))
    plt.show()

    return


predictions1, pose1 = split_data(features, pred1)

plot_kinematics(predictions1, pose1)


predictions2, pose2 = split_data(features, pred2)

plot_kinematics(predictions2, pose2)


# And what now? 
# If this analysis was not comprehensive enough for you to get an impression of 
# what your facial expressions and head movements looked like during your video
# I would love to hear about it. I will gladly revise and expand this script with your input.