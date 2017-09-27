import numpy as np
import os

import curvox.pc_vox_utils as pcvu
#above library having import errors so just remake some of the functions here
import pcl
import numpy as np





#in these Grasp Database directories, 3 prefixes for .pcd files:
#x is the partial view
#y is the expected completion
#pc is scaled version of x, so can just ignore it






def pcd_to_np(pcd_filename):
    """
    Read in PCD then return nx3 numpy array
    :type pcd_filename: str
    :rtype numpy.ndarray
    """

    pcd = pcl.load(pcd_filename)
    return pcl_to_np(pcd)


def pcl_to_np(pointcloud):
    """
    Convert PCL pointcloud to numpy nx3 numpy array
    :type pointcloud: pcl.PointCloud
    :rtype numpy.ndarray
    """

    pc_nx3 = pointcloud.to_array()
    return pc_nx3








#grasp_data_path = '/home/gregkocher/CLionProjects/ogn/data/grasp'
dir = "/home/gregkocher/CLionProjects/ogn/data/grasp_database/banjo_poisson_001/pointclouds"
pcd_filename = os.path.join(dir,'_0_0_0_x.pcd')

#Load pcd, convert to numpy array
pcd_as_np = pcd_to_np(pcd_filename)






#Now convert to voxels
#pcvu.get_voxel_resolution(pc, patch_size)
#pcvu. voxelize_points(points, pc_bbox_center, voxel_resolution, num_voxels_per_dim, pc_center_in_voxel_grid)
#pcvu.pc_to_binvox_for_shape_completion(points,patch_size)
patch_size=256#40
voxel_model=pcvu.pc_to_binvox_for_shape_completion(pcd_as_np,patch_size)
# print(voxel_model)
# print(type(voxel_model))
#is a <class 'binvox_rw.binvox_rw.Voxels'>



#Now convert from voxels to octree or whatever format the nets need
#Using the scripts provided by the OGN paper authors

#!!!!!Actually maybe not need convert?