import pandas as pd
import matplotlib.pyplot as plt
import cv2


# 1) Define functions
def create_skeleton(data):
    """
    This function creates skeletons from defined bodyparts for each timeframe.
    """
    skeletons = []
    for t in range(len(data)): # read out n_components from different poses

        lefteye = [data['lefteye1_x'][t], data['lefteye2_x'][t]], [data['lefteye1_y'][t], data['lefteye2_y'][t]], [data['lefteye1_z'][t], data['lefteye2_z'][t]]
        righteye = [data['righteye1_x'][t], data['righteye2_x'][t]], [data['righteye1_y'][t], data['righteye2_y'][t]], [data['righteye1_z'][t], data['righteye2_z'][t]]
        leyebrow = [data['leyebrow1_x'][t], data['leyebrow2_x'][t],data['leyebrow3_x'][t]],[data['leyebrow1_y'][t], data['leyebrow2_y'][t],data['leyebrow3_y'][t]],[data['leyebrow1_z'][t], data['leyebrow2_z'][t],data['leyebrow3_z'][t]]
        reyebrow = [data['reyebrow1_x'][t], data['reyebrow2_x'][t],data['reyebrow3_x'][t]],[data['reyebrow1_y'][t], data['reyebrow2_y'][t],data['reyebrow3_y'][t]],[data['reyebrow1_z'][t], data['reyebrow2_z'][t],data['reyebrow3_z'][t]]
        nose = [data['nose1_x'][t],data['nose3_x'][t],data['nose2_x'][t],data['nose4_x'][t],data['nose1_x'][t]],[data['nose1_y'][t],data['nose3_y'][t],data['nose2_y'][t],data['nose4_y'][t],data['nose1_y'][t]],[data['nose1_z'][t],data['nose3_z'][t],data['nose2_z'][t],data['nose4_z'][t],data['nose1_z'][t]]
        lips = [data['uplip_x'][t],data['llip_x'][t],data['lowlip_x'][t],data['rlip_x'][t],data['uplip_x'][t]],[data['uplip_y'][t],data['llip_y'][t],data['lowlip_y'][t],data['rlip_y'][t],data['uplip_y'][t]],[data['uplip_z'][t],data['llip_z'][t],data['lowlip_z'][t],data['rlip_z'][t],data['uplip_z'][t]]
        face = [data['rear_x'][t],data['chin_x'][t],data['lear_x'][t]],[data['rear_y'][t],data['chin_y'][t],data['lear_y'][t]],[data['rear_z'][t],data['chin_z'][t],data['lear_z'][t]]

        # Create skeleton from bodyparts for given timeframe
        skeleton = lefteye, righteye, leyebrow, reyebrow, nose, lips, face

        # Summarize skeletons over all timeframes
        skeletons.append(skeleton)

    return skeletons

def create_video_from_skeleton(data, elevation, azimuth):
    """
    This function takes the list of skeletons previously created, generates 3D plots and creates a video file.
    """
    img_array = [] # Initialize image array to save images from all timeframes


    for timeframe in range(len(data)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.view_init(elevation, azimuth)
        ax.set_title("3D frame from %s data" %filepath.split("/")[-1])
        for bodypart in range(len(data[0])):
            x = data[timeframe][bodypart][0]
            y = data[timeframe][bodypart][1]
            z = data[timeframe][bodypart][2]
            ax.plot(x,y,z, color='k')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.savefig("figure.png");
        plt.close()

        # Save figue in img_array
        img = cv2.imread("figure.png")
        height, width, layers = img.shape
        img_array.append(img)


    # Create video from moving skeleton
    out = cv2.VideoWriter('3Dframe.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (width,height))

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


# 2) Open data from csv filepath
filepath = "/Users/guillermo/Documents/GitHub/UQOAB/Pose Analysis/pose-3d.csv"
data = pd.read_csv(filepath, header=0)

# 3) Filter body part coordinates and ignore other variables
coordinates = data.loc[:,~data.columns.str.contains("score|error|ncams|fnum|center|M_")]

# 4) Create skeletons wtih function...
skeletons = create_skeleton(data = coordinates)

# 5) Create video file with function...
create_video_from_skeleton(data = skeletons, elevation = 10, azimuth = -90)