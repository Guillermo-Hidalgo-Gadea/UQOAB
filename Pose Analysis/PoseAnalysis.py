import pandas as pd
import matplotlib.pyplot as plt
import cv2


# 1) Open data from csv filepath
filepath = "/Users/guillermo/Documents/GitHub/UQOAB/Pose Analysis/pose-3d.csv"
data = pd.read_csv(filepath, header=0)


# 2) Filter body part coordinates and ignore other variables
coordinates = data.loc[:,~data.columns.str.contains("score|error|ncams|fnum|center|M_")]


# 3) Create skeletons from body parts for each timeframe
skeletons = []
for t in range(len(coordinates)):

    lefteye = [coordinates['lefteye1_x'][t], coordinates['lefteye2_x'][t]], [coordinates['lefteye1_y'][t], coordinates['lefteye2_y'][t]], [coordinates['lefteye1_z'][t], coordinates['lefteye2_z'][t]]
    righteye = [coordinates['righteye1_x'][t], coordinates['righteye2_x'][t]], [coordinates['righteye1_y'][t], coordinates['righteye2_y'][t]], [coordinates['righteye1_z'][t], coordinates['righteye2_z'][t]]
    leyebrow = [coordinates['leyebrow1_x'][t], coordinates['leyebrow2_x'][t],coordinates['leyebrow3_x'][t]],[coordinates['leyebrow1_y'][t], coordinates['leyebrow2_y'][t],coordinates['leyebrow3_y'][t]],[coordinates['leyebrow1_z'][t], coordinates['leyebrow2_z'][t],coordinates['leyebrow3_z'][t]]
    reyebrow = [coordinates['reyebrow1_x'][t], coordinates['reyebrow2_x'][t],coordinates['reyebrow3_x'][t]],[coordinates['reyebrow1_y'][t], coordinates['reyebrow2_y'][t],coordinates['reyebrow3_y'][t]],[coordinates['reyebrow1_z'][t], coordinates['reyebrow2_z'][t],coordinates['reyebrow3_z'][t]]
    nose = [coordinates['nose1_x'][t],coordinates['nose3_x'][t],coordinates['nose2_x'][t],coordinates['nose4_x'][t],coordinates['nose1_x'][t]],[coordinates['nose1_y'][t],coordinates['nose3_y'][t],coordinates['nose2_y'][t],coordinates['nose4_y'][t],coordinates['nose1_y'][t]],[coordinates['nose1_z'][t],coordinates['nose3_z'][t],coordinates['nose2_z'][t],coordinates['nose4_z'][t],coordinates['nose1_z'][t]]
    lips = [coordinates['uplip_x'][t],coordinates['llip_x'][t],coordinates['lowlip_x'][t],coordinates['rlip_x'][t],coordinates['uplip_x'][t]],[coordinates['uplip_y'][t],coordinates['llip_y'][t],coordinates['lowlip_y'][t],coordinates['rlip_y'][t],coordinates['uplip_y'][t]],[coordinates['uplip_z'][t],coordinates['llip_z'][t],coordinates['lowlip_z'][t],coordinates['rlip_z'][t],coordinates['uplip_z'][t]]
    face = [coordinates['rear_x'][t],coordinates['chin_x'][t],coordinates['lear_x'][t]],[coordinates['rear_y'][t],coordinates['chin_y'][t],coordinates['lear_y'][t]],[coordinates['rear_z'][t],coordinates['chin_z'][t],coordinates['lear_z'][t]]

    # Create skeleton from bodyparts for given timeframe
    skeleton = lefteye, righteye, leyebrow, reyebrow, nose, lips, face

    # Summarize skeletons over all timeframes
    skeletons.append(skeleton)


# 4) Plot skeleton in 3D coordinates for each successive timeframe
img_array = [] # Initialize image array to save images from all timeframes

for timeframe in range(len(coordinates)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(11, -80) #(elevation, azimuth)
    ax.set_title("3D frame from %s data" %filepath.split("/")[-1])

    for bodypart in range(len(skeletons[0])):
        x = skeletons[timeframe][bodypart][0]
        y = skeletons[timeframe][bodypart][1]
        z = skeletons[timeframe][bodypart][2]
        ax.plot(x,y,z, color='k')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig("figure.png");
    plt.close()

    # save figue and append to img_array
    img = cv2.imread("figure.png")
    height, width, layers = img.shape
    img_array.append(img)


# 5) create video from moving skeleton
out = cv2.VideoWriter('3Dframe.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (width,height))

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()