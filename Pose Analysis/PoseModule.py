import pandas as pd
import matplotlib.pyplot as plt
import cv2
import yaml


class Pose_3D:
    """
    This class initializes a Pose_3D object with attributes such as filepath, data etc. and methods such as create_skeleton and create_video_from_skeleton.
    """
    def __init__(self):
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file) # read from config.yaml

        self.filepath = config['filepath']
        self.elevation = config['elevation']
        self.azimuth = config['azimuth']
        self.data = pd.read_csv(self.filepath, header=0)
        self.coordinates = self.data.loc[:,~self.data.columns.str.contains("score|error|ncams|fnum|center|M_")]
        self.skeletons = []
        self.img_array = []

    def create_skeleton(self):
        """
        This function creates skeletons from defined bodyparts for each timeframe.
        """
        for t in range(len(self.coordinates)): # read out n_components from different poses

            lefteye = [self.coordinates['lefteye1_x'][t], self.coordinates['lefteye2_x'][t]], [self.coordinates['lefteye1_y'][t], self.coordinates['lefteye2_y'][t]], [self.coordinates['lefteye1_z'][t], self.coordinates['lefteye2_z'][t]]
            righteye = [self.coordinates['righteye1_x'][t], self.coordinates['righteye2_x'][t]], [self.coordinates['righteye1_y'][t], self.coordinates['righteye2_y'][t]], [self.coordinates['righteye1_z'][t], self.coordinates['righteye2_z'][t]]
            leyebrow = [self.coordinates['leyebrow1_x'][t], self.coordinates['leyebrow2_x'][t],self.coordinates['leyebrow3_x'][t]],[self.coordinates['leyebrow1_y'][t], self.coordinates['leyebrow2_y'][t],self.coordinates['leyebrow3_y'][t]],[self.coordinates['leyebrow1_z'][t], self.coordinates['leyebrow2_z'][t],self.coordinates['leyebrow3_z'][t]]
            reyebrow = [self.coordinates['reyebrow1_x'][t], self.coordinates['reyebrow2_x'][t],self.coordinates['reyebrow3_x'][t]],[self.coordinates['reyebrow1_y'][t], self.coordinates['reyebrow2_y'][t],self.coordinates['reyebrow3_y'][t]],[self.coordinates['reyebrow1_z'][t], self.coordinates['reyebrow2_z'][t],self.coordinates['reyebrow3_z'][t]]
            nose = [self.coordinates['nose1_x'][t],self.coordinates['nose3_x'][t],self.coordinates['nose2_x'][t],self.coordinates['nose4_x'][t],self.coordinates['nose1_x'][t]],[self.coordinates['nose1_y'][t],self.coordinates['nose3_y'][t],self.coordinates['nose2_y'][t],self.coordinates['nose4_y'][t],self.coordinates['nose1_y'][t]],[self.coordinates['nose1_z'][t],self.coordinates['nose3_z'][t],self.coordinates['nose2_z'][t],self.coordinates['nose4_z'][t],self.coordinates['nose1_z'][t]]
            lips = [self.coordinates['uplip_x'][t],self.coordinates['llip_x'][t],self.coordinates['lowlip_x'][t],self.coordinates['rlip_x'][t],self.coordinates['uplip_x'][t]],[self.coordinates['uplip_y'][t],self.coordinates['llip_y'][t],self.coordinates['lowlip_y'][t],self.coordinates['rlip_y'][t],self.coordinates['uplip_y'][t]],[self.coordinates['uplip_z'][t],self.coordinates['llip_z'][t],self.coordinates['lowlip_z'][t],self.coordinates['rlip_z'][t],self.coordinates['uplip_z'][t]]
            face = [self.coordinates['rear_x'][t],self.coordinates['chin_x'][t],self.coordinates['lear_x'][t]],[self.coordinates['rear_y'][t],self.coordinates['chin_y'][t],self.coordinates['lear_y'][t]],[self.coordinates['rear_z'][t],self.coordinates['chin_z'][t],self.coordinates['lear_z'][t]]

            # Create skeleton from bodyparts for given timeframe
            skeleton = lefteye, righteye, leyebrow, reyebrow, nose, lips, face

            # Summarize skeletons over all timeframes
            self.skeletons.append(skeleton)

        return self.skeletons

    def create_video_from_skeleton(self):
        """
        This function takes the list of skeletons previously created, generates 3D plots and creates a video file.
        """
        for timeframe in range(len(self.coordinates)):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.view_init(self.elevation, self.azimuth)
            ax.set_title("3D frame from %s data" %self.filepath.split("/")[-1])
            for bodypart in range(len(self.skeletons[0])):
                x = self.skeletons[timeframe][bodypart][0]
                y = self.skeletons[timeframe][bodypart][1]
                z = self.skeletons[timeframe][bodypart][2]
                ax.plot(x,y,z, color='k')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            plt.savefig("figure.png");
            plt.close()

            # Save figue in img_array
            img = cv2.imread("figure.png")
            height, width, layers = img.shape
            self.img_array.append(img)


        # Create video from moving skeleton
        out = cv2.VideoWriter('3Dframe.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (width,height))

        for i in range(len(self.img_array)):
            out.write(self.img_array[i])
        out.release()
