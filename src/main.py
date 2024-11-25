from typing import Tuple
import laspy
import os
import numpy as np
import time
import pyvista as pv
import CSF

class GWSeg:
    """
    GWSeg (Gehweg Segmentation) is a class that extract sidwalk geometries from LiDAR data.
    """

    def __init__(self):
        pass

    def read(self, path: str, classification=None) -> laspy.LasData:
        """
        Reads a point cloud from a file

        :param path: The path to the input file.
        :type path: str
        :param classification: The optional classification value to filter the point cloud, defaults to None.
        :type classification: int, optional
        :return: The point cloud as a NumPy array.
        :rtype: np.ndarray
        :raises ValueError: If the input file format is not supported.
        """
        start = time.time()
        extension = os.path.splitext(path)[1]
        try:
            if extension not in ['.laz', '.las']:
                raise ValueError(f'The input file format {extension} is not supported.\nThe file format should be [.las/.laz].')
        except ValueError as error:
            message = str(error)
            lines = message.split('\n')
            print(lines[-2])
            print(lines[-1])
            exit()

        print(f'Reading {path}...')
        las = laspy.read(path)
        if classification is not None:
            try:
                if hasattr(las, 'classification'):
                    print(f'- Reading points with classification value {classification}...')
                    las.points = las.points[las.raw_classification == classification]
                else:
                    raise ValueError('The input file does not contain classification values.')
            except ValueError as error:
                message = str(error)
                lines = message.split('\n')
                print(lines[-2])
                print(lines[-1])
                exit()

        end = time.time()
        print(f'File reading is completed in {end - start:.2f} seconds. The point cloud contains {las.points.array.size} points.')
        return las
    

    def view(self, pcd: np.ndarray, include_rgb: bool = True):
        """
        Visualizes the point cloud.

        :param points: The point cloud as a NumPy array.
        :type points: np.ndarray
        :param include_rgb: A flag to include RGB values in the visualization, defaults to True. If False, intensity values are used.
        :type include_rgb: bool, optional
        :raises ValueError: If the input point cloud is empty

        """
        print('Visualizing the point cloud...')
        rgb = include_rgb
        points = np.vstack([pcd.x, pcd.y, pcd.z]).T
        if include_rgb:
            scalars = np.vstack([pcd.red, pcd.green, pcd.blue]).T / 65535
        elif hasattr(pcd, 'intensity'):
            scalars = pcd.intensity
        else:
            rgb = False
        
        pv.plot(points, scalars=scalars, rgb=rgb)

        
    def csf(self, pcd: laspy.LasData, class_threshold: float = 0.5, cloth_resolution: float = 0.2, iterations: int = 500, slope_smooth: bool = False, csf_path: str = None) -> Tuple[laspy.LasData, laspy.LasData, laspy.LasData]:
        """
        Applies the CSF (Cloth Simulation Filter) algorithm to filter ground points in a point cloud.

        :param pcd: The input point cloud object.
        :type pcd: object
        :param class_threshold: The threshold value for classifying points as ground/non-ground, defaults to 0.5.
        :type class_threshold: float, optional
        :param cloth_resolution: The resolution value for cloth simulation, defaults to 0.2.
        :type cloth_resolution: float, optional
        :param iterations: The number of iterations for the CSF algorithm, defaults to 500.
        :type iterations: int, optional
        :param slope_smooth: A boolean indicating whether to enable slope smoothing, defaults to False.
        :type slope_smooth: bool, optional
        :param csf_path: The path to save the results, defaults to None.
        :type csf_path: str, optional
        :return: A tuple containing the filtered point cloud object, non-ground points, and ground points.
        :rtype: Tuple[object, object, object]
        """
        
        points = np.vstack([pcd.x, pcd.y, pcd.z]).T
        start = time.time()
        print(f'Applying CSF algorithm...')
        csf = CSF.CSF()
        csf.params.bSloopSmooth = slope_smooth
        csf.params.cloth_resolution = cloth_resolution
        csf.params.interations = iterations
        csf.params.class_threshold = class_threshold
        csf.setPointCloud(points)
        ground = CSF.VecInt()
        non_ground = CSF.VecInt()
        csf.do_filtering(ground, non_ground)
        end = time.time()

        pcd_non_ground = laspy.LasData(pcd.header)
        pcd_non_ground.points = pcd.points[np.array(non_ground)]

        pcd_ground = laspy.LasData(pcd.header)
        pcd_ground.points = pcd.points[np.array(ground)]

        if csf_path:
            print(f'Saving the filtered point cloud to {csf_path}...')
            pcd_non_ground.write(csf_path)
            print(f'The filtered point cloud is saved to {csf_path}.')
            pcd_ground.write(csf_path.replace('.las', '_ground.las'))

        print(f'CSF algorithm is completed in {end - start:.2f} seconds. The filtered non-ground cloud contains {len(non_ground)} points and the ground cloud contains {len(ground)} points.')
        
        return pcd, pcd_non_ground, pcd_ground


if __name__ == "__main__":
    # Create an instance of the SamLidar
    gwseg = GWSeg()
    # Read the point cloud
    pcd = gwseg.read("data/raw/filtered_8214_109554.laz")
    
    # Apply the Cloth Simulation Filter (CSF) algorithm for ground filtering using the csf method of the SamLidar instance.
    pcd, pcd_non_ground, pcd_ground = gwseg.csf(pcd)
    # Visualize the point cloud
    gwseg.view(pcd_ground)

