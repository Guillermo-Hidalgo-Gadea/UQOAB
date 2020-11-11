# test script for loading data, cleaning data, calculating new features
# and HMM examples


import numpy as np
from hmmlearn import hmm
import os
import pandas as pd
import matplotlib.pyplot as plt
import umap
import math


def split_data(data, prediction):
    
    n = max(prediction)+1 #read out n_components predicted
    
    data['pred'] = prediction
    grouped = data.groupby(data.pred)

    predictions = [grouped.get_group(i) for i in range(n)]

    pose = [predictions[i].mean() for i in range(n)]
    
    return predictions, pose


def face_skeleton(pose):
    
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
            ax.view_init(elevation, azimuth)  #11,280
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


def plot_prediction(data, predictions):
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


def plot_kinematics(predictions):
    
    ncols = 3
    nrows = math.ceil(len(predictions)/ncols)
    width = ncols*6
    height = nrows *5
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))

    for ax, n in zip(axes.flat, range(len(pose))):
        ax.plot(predictions[n]['position_x'], color = 'g')
        ax.plot(predictions[n]['position_y'], color = 'g')
        ax.plot(predictions[n]['position_z'], color = 'g')
        
        ax.plot(predictions[n]['velocity_x'], color = 'y')
        ax.plot(predictions[n]['velocity_y'], color = 'y')
        ax.plot(predictions[n]['velocity_z'], color = 'y')
        
        ax.plot(predictions[n]['acceleration_x'], color = 'r')
        ax.plot(predictions[n]['acceleration_y'], color = 'r')
        ax.plot(predictions[n]['acceleration_z'], color = 'r')
        
        ax.set(xlabel='Time (frames)', ylabel='Position, Velocity and Acceleration')
        ax.set_title('Kinematic Profile in Predicted Class: %d' %n)
                
    plt.suptitle('Hidden Markov Model predictions with N = %d Components' %len(pose))
    plt.show()
    
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    
    for ax, n in zip(axes.flat, range(len(pose))):
        ax.hist(predictions[n]['position_x'], color = 'g')
        ax.hist(predictions[n]['position_y'], color = 'y')
        ax.hist(predictions[n]['position_z'], color = 'r')
        
        ax.set(xlabel='x, y, z movement', ylabel='frequency')
        ax.set_title('Movement in Predicted Class: %d' %n)
                
    plt.suptitle('Hidden Markov Model predictions with N = %d Components' %len(pose))
    plt.show()
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    
    for ax, n in zip(axes.flat, range(len(pose))):
        ax.hist(predictions[n]['velocity_x'], color = 'g')
        ax.hist(predictions[n]['velocity_y'], color = 'y')
        ax.hist(predictions[n]['velocity_z'], color = 'r')
        
        ax.set(xlabel='x, y, z velocity', ylabel='frequency')
        ax.set_title('Velocity in Predicted Class: %d' %n)
                
    plt.suptitle('Hidden Markov Model predictions with N = %d Components' %len(pose))
    plt.show()
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    
    for ax, n in zip(axes.flat, range(len(pose))):
        ax.hist(predictions[n]['acceleration_x'], color = 'g')
        ax.hist(predictions[n]['acceleration_y'], color = 'y')
        ax.hist(predictions[n]['acceleration_z'], color = 'r')
        
        ax.set(xlabel='x, y, z acceleration', ylabel='frequency')
        ax.set_title('Acceleration in Predicted Class: %d' %n)
                
    plt.suptitle('Hidden Markov Model predictions with N = %d Components' %len(pose))
    plt.show()
        
    return


#### SET WORKING DIRECTORY AND SELECT DATA
path = '/Users/guillermo/OneDrive/Breadnut/Anipose/emoface2/session1/pose-3d'
os.chdir(path)
os.getcwd() 

data = pd.read_csv('emo.csv', header=0)


#### CLEAN AND REARRANGE DATA
data.info()
coords = data.loc[:,~data.columns.str.contains('score|error|ncams|fnum|center|M_')]

scores = data.loc[:, data.columns.str.contains('score')]
scores.describe()
scores.hist()
scores.boxplot()

errors = data.loc[:, data.columns.str.contains('error')]
errors.describe()
errors.hist()
errors.boxplot()


#### COORDINATE TRANSFORMATION with nasal bone as reference for facial expression
z_coords = coords.copy()
for i in range(z_coords.shape[1]):
    if '_x' in z_coords.columns[i]:
        z_coords.loc[:,z_coords.columns[i]] = z_coords.loc[:,z_coords.columns[i]].subtract(coords.loc[:,"nose1_x"].values)
    elif '_y' in z_coords.columns[i]:
        z_coords.loc[:,z_coords.columns[i]] = z_coords.loc[:,z_coords.columns[i]].subtract(coords.loc[:,"nose1_y"].values)
    elif '_z' in z_coords.columns[i]:
        z_coords.loc[:,z_coords.columns[i]] = z_coords.loc[:,z_coords.columns[i]].subtract(coords.loc[:,"nose1_z"].values)
    else:
        pass

#### Save egocentric coordinates 
emoface_egocentric = z_coords.to_numpy()
np.save('video-1-PE-seq', emoface_egocentric)

#### EXTRACT KINEMATIC FEATURES
features = z_coords.copy()

features['position_x'] = coords['nose1_x']
features['position_y'] = coords['nose1_y']
features['position_z'] = coords['nose1_z']

plt.plot(features['position_x'])
plt.plot(features['position_y'])
plt.plot(features['position_z'])


features['velocity_x'] = np.append([0],np.diff(features['position_x'], n=1)) #difference between adjacent points as velocity, add 0 to mantain length
features['velocity_y'] = np.append([0],np.diff(features['position_y'], n=1))
features['velocity_z'] = np.append([0],np.diff(features['position_z'], n=1))

plt.plot(features['velocity_x'])
plt.plot(features['velocity_y'])
plt.plot(features['velocity_z'])

features['acceleration_x'] = np.append([0],np.diff(features['velocity_x'], n=1)) #difference between adjacent speeds as acceleration, add 0 to mantain length
features['acceleration_y'] = np.append([0],np.diff(features['velocity_y'], n=1))
features['acceleration_z'] = np.append([0],np.diff(features['velocity_z'], n=1))

plt.plot(features['acceleration_x'])
plt.plot(features['acceleration_y'])
plt.plot(features['acceleration_z'])


#### UMAP DIMENSIONALITY REDUCTION
reducer = umap.UMAP()
umap_reduction = pd.DataFrame(reducer.fit_transform(features))
plt.plot(umap_reduction)


#### HIDDEN MARKOV MODEL trained on feature matrix
model = hmm.GaussianHMM(n_components = 20, covariance_type="full")
model.fit(features)

pred = model.predict(features)
transition = model.transmat_
transition
means = model.means_
means

np.savetxt('out_pred.csv', pred, delimiter=',')
np.savetxt('trans.csv', transition, delimiter=',')


#### HIDDEN MARKOV MODEL trained on umap reduction
model2 = hmm.GaussianHMM(n_components = 20, covariance_type="full")
model2.fit(umap_reduction)

pred2 = model2.predict(umap_reduction)
transition2 = model2.transmat_
transition2


#### HIDDEN MARKOV MODEL trained on umap reduction predicting raw data
pred3 = model2.predict(features)
pred4 = model2.predict(coords)


#### VISUALIZE MODEL PREDICTIONS FROM DATA

plot_prediction(features, pred)

plot_prediction(umap_reduction, pred2)

plot_prediction(features, pred2)

plot_prediction(umap_reduction, pred)



#### SPLIT DATA BY PREDICTION 
predictions, pose = split_data(z_coords, pred)
predictions2, pose2 = split_data(z_coords, pred2)


#### VISUALIZE AVERAGE POSE OF MODEL PREDICTIONS
plot_3Dpose(pose, 11, 280)    
plot_3Dpose(pose2, 11, 280)  

#skeletons = face_skeleton(pose) #this functin is passed directly to plot_3Dpose

#### ANALYZE KINEMATICS BY PREDICTIONS

predictions, pose = split_data(features, pred)

plot_kinematics(predictions)

predictions2, pose2 = split_data(features, pred2)

plot_kinematics(predictions2)

