# Copyright (c) 2024 - Plan4Better GmbH 
# Author : Majk Shkurti (majk.shkurti@plan4better.de)
# This file is distributed under the MIT licence. See LICENSE file for complete text of the license.

from typing import Tuple
import laspy
import os
import numpy as np
import time
import pyvista as pv
import CSF
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
import open3d as o3d
import glob
from ml3d.datasets.parislille3d_50_classes import COLOR_MAP


# if ml3d._torch.backends.mps.is_available():
#     mps_device = ml3d._torch.device("mps")
#     x = ml3d._torch.ones(1, device=mps_device)
#     print (x)
# else:
#     print ("MPS device not found.")

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
        print('Applying CSF algorithm...')
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

    def prepare_for_inference(self, pcd: laspy.LasData) -> dict:
        """
        Prepares the point cloud for inference.

        :param pcd: The input point cloud object.
        :type pcd: object
        :return: A dictionary containing the point cloud data.
        :rtype: dict
        """

        point = np.vstack([pcd.x, pcd.y, pcd.z]).T
        data = {"point":point, 'feat': None, 'label':np.zeros((len(point),), dtype=np.int32)}

        return data



def laspy_to_o3d(las: laspy.lasdata) -> o3d.t.geometry.PointCloud:
    # get points
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # make blank cloud
    pcd_t = o3d.t.geometry.PointCloud()

    # add x,y,z to o3d cloud
    pcd_t.point.positions = o3d.core.Tensor(points)

    # if rgb present, convert to o3d colors
    all_dims = list(las.point_format.dimension_names)[3:]
    if "red" in all_dims and "green" in all_dims and "blue" in all_dims:
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
        pcd_t.point.colors = o3d.core.Tensor(colors)
        all_dims.remove("blue")
        all_dims.remove("green")
        all_dims.remove("red")

    # add other attributes
    attributes = all_dims

    for attr in attributes:
        # If only has single value, this is much faster
        if np.all(las[attr] == las[attr][0]):
            # Fill with the single value
            las_attr = np.full((len(las[attr]), 1), las[attr][0])
        else:
            # assumes only 1d.  Otherwise you need vstack which is slower
            las_attr = np.array(las[attr])[:, None]

        pcd_t.point[attr] = o3d.core.Tensor(las_attr)

    return pcd_t


# Class colors, RGB values as ints for easy reading
# COLOR_MAP = {
#     0: (0, 0, 0),
#     1: (245, 150, 100),
#     2: (245, 230, 100),
#     3: (150, 60, 30),
#     4: (180, 30, 80),
#     5: (255, 0., 0),
#     6: (30, 30, 255),
#     7: (200, 40, 255),
#     8: (90, 30, 150),
#     9: (255, 0, 255),
#     10: (255, 150, 255),
#     11: (75, 0, 75),
#     12: (75, 0., 175),
#     13: (0, 200, 255),
#     14: (50, 120, 255),
#     15: (0, 175, 0),
#     16: (0, 60, 135),
#     17: (80, 240, 150),
#     18: (150, 240, 255),
#     19: (0, 0, 255),
# }

# # ------ for custom data -------
# labels = {
#     ### SemanticKITTI labels
#     0: 'unlabeled',
#     1: 'car',
#     2: 'bicycle',
#     3: 'motorcycle',
#     4: 'truck',
#     5: 'other-vehicle',
#     6: 'person',
#     7: 'bicyclist',
#     8: 'motorcyclist',
#     9: 'road',
#     10: 'parking',
#     11: 'sidewalk',
#     12: 'other-ground',
#     13: 'building',
#     14: 'fence',
#     15: 'vegetation',
#     16: 'trunk',
#     17: 'terrain',
#     18: 'pole',
#     19: 'traffic-sign',
#     #### Toronto3D labels
#     # 0: 'Unclassified',
#     # 1: 'Ground',
#     # 2: 'Road_markings',
#     # 3: 'Natural',
#     # 4: 'Building',
#     # 5: 'Utility_line',
#     # 6: 'Pole',
#     # 7: 'Car',
#     # 8: 'Fence', 
#     #### ParisLille3D labels
#     # 0: 'unclassified',
#     # 1: 'ground',
#     # 2: 'building',
#     # 3: 'pole-road_sign-traffic_light',
#     # 4: 'bollard-small_pole',
#     # 5: 'trash_can',
#     # 6: 'barrier',
#     # 7: 'pedestrian',
#     # 8: 'car',
#     # 9: 'natural-vegetation'
# }

def load_custom_dataset(dataset_path):
	print("Loading custom dataset")
	pcd_paths = glob.glob(dataset_path+"/*.pcd")
	pcds = []
	for pcd_path in pcd_paths:
		pcds.append(o3d.io.read_point_cloud(pcd_path))
	return pcds


def prepare_point_cloud_for_inference(pcd):
    # # coords_offset = [410700.00, 5477700.00, 228.9439]
    # # Remove NaNs and infinity values
    # points = np.vstack([pcd.x, pcd.y, pcd.z]).T
    # points = np.float32(points)
    # # colors = (np.vstack([pcd.red, pcd.green, pcd.blue]).T / 256).astype(np.float32)
    # # feat = colors
    # # intensity = pcd.intensity.astype(np.float32)

    # labels = np.zeros(np.shape(points)[0], dtype=np.int32)
    data = {
        "point": np.asarray(pcd.points, dtype=np.float32),
        'feat': None,
        'label': np.zeros((len(pcd.points),), dtype=np.int32)
    }


    # data = {"point": points, 'feat': feat, 'intensity': intensity, 'label': labels}
    # data = {"point": points, 'feat': None, 'label': labels}

    return data, pcd

def custom_draw_geometry(pcd):
	vis = o3d.visualization.Visualizer()
	vis.create_window()
	vis.get_render_option().point_size = 2.0
	vis.get_render_option().background_color = np.asarray([1.0, 1.0, 1.0])
	vis.add_geometry(pcd)
	vis.run()
	vis.destroy_window()




if __name__ == "__main__":
    for label in COLOR_MAP:
        COLOR_MAP[label] = tuple(val/255 for val in COLOR_MAP[label])

    # cfg_file = "configs/randlanet_toronto3d.yml"
    cfg_file = "configs/randlanet_semantickitti.yml"
    # cfg_file = "src/Open3D-ML/ml3d/configs/randlanet_parislille3d.yml"
    # cfg_file = "src/Open3D-ML/ml3d/configs/randlanet_parislille3d_50c.yml"
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    model = ml3d.models.RandLANet(**cfg.model)
    
    # download the weights.ß
    ckpt_folder = "./logs/"
    os.makedirs(ckpt_folder, exist_ok=True)
    ckpt_path = ckpt_folder + "randlanet_semantickitti_202201071330utc.pth"
    randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth"
    # ckpt_path = ckpt_folder + "randlanet_toronto3d_202201071330utc.pth"
    # randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_toronto3d_202201071330utc.pth"
    # ckpt_path = ckpt_folder + "randlanet_parislille3d_202201071330utc.pth"
    # randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_parislille3d_202201071330utc.pth"
    # ckpt_path = ckpt_folder + "ckpt_00100.pth"
    # randlanet_url = ""
    if not os.path.exists(ckpt_path):
        cmd = "wget {} -O {}".format(randlanet_url, ckpt_path)
        os.system(cmd)

    # cfg.dataset['dataset_path'] = "data/raw/SemanticKITTI"
    # cfg.dataset['custom_dataset_path'] = "data/raw/pcds"
    # dataset = ml3d.datasets.ParisLille3D(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
    pipeline = ml3d.pipelines.SemanticSegmentation(model, **cfg.pipeline)
    # load the parameters.
    pipeline.load_ckpt(ckpt_path=ckpt_path)

    # test_split = dataset.get_split("test")
    # test_data = test_split.get_data(0)

    gwseg = GWSeg()
    # pcd = gwseg.read("data/raw/ajaccio_2.las")
    pcd = o3d.io.read_point_cloud("data/raw/filtered_8214_109554.ply")
    pcd.remove_non_finite_points()
    # pcd, pcd_non_ground, pcd_ground = gwseg.csf(pcd)
    # data = gwseg.prepare_for_inference(pcd_ground)
    # # Get one test point cloud from the custom dataset
    # pc_idx = 0 # change the index to get a different point cloud
    # data, pcd = prepare_point_cloud_for_inference(pcd)
    # feat = data["colors"].numpy().astype(np.float32)
    data = {
        "point" : np.asarray(pcd.points, dtype=np.float32),
        "feat": np.array(pcd.colors, dtype=np.float32),
        "label" : np.zeros((len(pcd.points),), dtype=np.int32)
    }

    # Run inference
    result = pipeline.run_inference(data)    


    # Colorize the point cloud with predicted labels
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(data["point"])
    colors = [COLOR_MAP[clr] for clr in list(result['predict_labels'])]
    o3d_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create visualization
    custom_draw_geometry(o3d_pcd)



    # test_split = dataset.get_split("test")
    # data = test_split.get_data(0)
    # run inference on a single example.
    # returns dict with 'predict_labels' and 'predict_scores'.
    # result = pipeline.run_inference(data)

    # Create a pcd to be visualized 
    # pcd = o3d.geometry.PointCloud()
    # xyz = data["point"] # Get the points
    # pcd.points = o3d.utility.Vector3dVector(xyz)

    # colors = [COLOR_MAP[clr] for clr in list(result['predict_labels'])] # Get the color associated to each predicted label
    # pcd.colors = o3d.utility.Vector3dVector(colors) # Add color data to the point cloud

    # # Create visualization
    # custom_draw_geometry(pcd)

    # pipeline.run_test()


    # # Create an instance of the SamLidar
    # gwseg = GWSeg()
    # # Read the point cloud
    # pcd = gwseg.read("data/raw/filtered_8214_109554.laz")
    
    # # Apply the Cloth Simulation Filter (CSF) algorithm for ground filtering using the csf method of the SamLidar instance.
    # pcd, pcd_non_ground, pcd_ground = gwseg.csf(pcd)
    # # Visualize the point cloud
    # gwseg.view(pcd_ground)

