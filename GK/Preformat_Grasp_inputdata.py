import numpy as np
import os

import curvox.pc_vox_utils as pcvu
#above library having import errors so just remake some of the functions here
import pcl

import binvox_rw
import subprocess


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





#'banjo_poisson_001/pointclouds/_0_0_0_y.pcd' is CUTOFF
#use 0 0 1 instead

#grasp_data_path = '/home/gregkocher/CLionProjects/ogn/data/grasp'


dir = "/home/gregkocher/CLionProjects/ogn/data/grasp_database/banjo_poisson_001/pointclouds"
#dir = "/home/gregkocher/CLionProjects/ogn/data/grasp_database/bowling_pin_poisson_000/pointclouds"
pcd_filename = os.path.join(dir,'_0_0_1_y.pcd')#y for full view, x for partial

#Load pcd, convert to numpy array
pcd_as_np = pcd_to_np(pcd_filename)



#Use curvox tools to get parameters:
patch_size=256#256#40#80
voxel_resolution = pcvu.get_voxel_resolution(pcd_as_np, patch_size)
print('voxel_resolution',voxel_resolution)
bbox_center = pcvu.get_bbox_center(pcd_as_np)#format [x,y,z]
print('bbox_center',bbox_center)



#Now convert to voxels
#pcvu. voxelize_points(points, pc_bbox_center, voxel_resolution, num_voxels_per_dim, pc_center_in_voxel_grid)
#binvox_rw.Voxels(data, dims, translate, scale, axis_order)
#pcvu.pc_to_binvox_for_shape_completion(points,patch_size)


#Saving this voxel_model as binvox gives pc like viewvox [small patches of voxels with gaps between]
voxel_model=pcvu.pc_to_binvox_for_shape_completion(pcd_as_np,patch_size)
# print(voxel_model)
# print(type(voxel_model))
#is a <class 'binvox_rw.binvox_rw.Voxels'>


#Vs. saving this method gives a solid version even at 256, but looks low resolution
# voxel_model = pcvu.voxelize_points(pcd_as_np, [0,0,0], 1., patch_size, [0,0,0])
# voxel_model = binvox_rw.Voxels(voxel_model, [patch_size,patch_size,patch_size], [0, 0, 0], 0, 'xzy')



#bbox_center = [0,0,0]
#voxel_resolution = 1.
# center = [patch_size/2.,patch_size/2.,patch_size/2.]
# voxel_model = pcvu.voxelize_points(pcd_as_np, bbox_center, voxel_resolution, patch_size, center)
# voxel_model = binvox_rw.Voxels(voxel_model, [patch_size,patch_size,patch_size], [0,0,0], 0, 'xzy')





#Just save out a binvox to look at it and make sure it looks right:
#voxel_model = binvox_rw.Voxels(occupancy_3d_array, [resolution, resolution, resolution], [0, 0, 0], 0, 'xzy')
test_binvox_path = "/home/gregkocher/Desktop/aaaaaaaaa.binvox"
with open(test_binvox_path, 'wb') as fp:
    binvox_rw.write(voxel_model, fp, fast=True)
viewvox_path = '/home/gregkocher/CLionProjects/viewvox/viewvox'
test_cmd = '{0} {1}'.format(viewvox_path, test_binvox_path)
test_process = subprocess.Popen(test_cmd, shell=True)







#Now convert from voxels to octree.
#Even if following OGN paper and doing input as 2D image, need the octrees as truth for loss functions.
#Using the scripts provided by the OGN paper authors:
#10/2/2017 actually the OGN repo does not have working code for this.
#Their .../tools/ogn_converter.cpp cannot compile because of various issues.
#Their python code does not go binvox->octree, only the reverse direction.
#So have to make our own...