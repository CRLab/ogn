import os
import subprocess
import numpy as np
from sklearn import metrics
import pandas as pd

import binvox_rw

import sys
sys.path.insert(0,"/home/gregkocher/CLionProjects/ogn/python/rendering")
import python_octree


import matplotlib.pyplot as plt








# def voxel_grid_to_octree(binvox_path):
#     """
#     Convert from .binvox file to .ot file
#     """










test_binvox_path = "/home/gregkocher/CLionProjects/ogn/examples/single_view_3d/shapenet_cars_128/output/172K_iters_output/0000.binvox"
with open(test_binvox_path,"rb") as f:
    ml = binvox_rw.read_as_3d_array(f)
    #print(ml.data)
    # print(ml.dims)
    # print(ml.scale)
    # print(ml.translate)
    print(ml.axis_order)

data = ml.data
# nonzero_count = data.sum()
# print(128**3, nonzero_count)
# nonzeros = np.where(data)

print data.shape

#ASSUMES CUBIC VOXEL GRIDS [NxNxN not LxWxH]
cube_len = data.shape[0]
max_level = int(np.ceil(np.log2(cube_len)))
print max_level

min_level = 0




x, y, z, side_len = python_octree.get_cube_params(1, 512)
print x,y,z,side_len

x, y, z, side_len = python_octree.get_cube_params(8, 512)
print x,y,z,side_len

x, y, z, side_len = python_octree.get_cube_params(90, 512)
print x,y,z,side_len


g=ggg


#1st have to pad 0_0_0_x.pcd and 0_0_0_y.pcd to be power of 2 side dims, and same dimensions all 3 axes
#...



def BuildOctree(ot,data,current_level,offset):
    """

    :param ot: dictionary of octree Morton order keys: binary occupancy values
    :param data: a 3d Boolean voxel grid numpy array
    :param current_level: int of octree level
    :param max_level: deepest level to go to when building the octree
    :param cube_len: length of each side of the octants [assumed NxNxN]
    :param offset: np array [x,y,z] of translation offset for this region, since needed for Z-order key

    :return: ot - dictionary of octree keys:values for all NONmixed octants for all levels
    """



    #Assuming NxNxN, just get 1st [0]
    cube_len = data.shape[0]


    # print 'current_level', current_level
    # print 'max_level', max_level
    # print 'offset', offset
    # print 'cube_len', cube_len


    #Check if the whole octant is empty [all voxels False]
    if np.alltrue(~data):
        #print current_level
        #print offset
        #!!!!!!!!!!!! not actually sure which key to use when not at highest level.
        #!!!!!!!!!!!! e.g. when an 8x8x8 region is all 0s', which if those 8**3 coordinates gets used????
        # x, y, z, side_len = python_octree.get_cube_params(key, resolution)
        Mortonkey = python_octree.compute_key(offset[0], offset[1], offset[2], current_level) #str(offset) + '_' + str(current_level)
        ot[Mortonkey] = 0
        return ot

    #Check if the whole octant is filled [all voxels True]
    if np.alltrue(data):
        #print current_level
        #print offset
        # !!!!!!!!!!!! not actually sure which key to use when not at highest level.
        # !!!!!!!!!!!! e.g. when an 8x8x8 region is all 0s', which if those 8**3 coordinates gets used????
        # x, y, z, side_len = python_octree.get_cube_params(key, resolution)
        Mortonkey = python_octree.compute_key(offset[0], offset[1], offset[2], current_level)
        ot[Mortonkey] = 1
        return ot


    #If not empty or filled, is mixed, so keep subdividing:



    #Check if already past max level:
    #if current_level <= max_level:
    #NO. Just always build out full tree until an octant is a 1x1x1 voxel
    cube_len = cube_len / 2
    octant1 = BuildOctree(ot, data[:cube_len,:cube_len,:cube_len], current_level + 1, offset+cube_len*np.array([0,0,0]))
    octant2 = BuildOctree(ot, data[cube_len:,:cube_len,:cube_len], current_level + 1, offset+cube_len*np.array([1,0,0]))
    octant3 = BuildOctree(ot, data[:cube_len,cube_len:,:cube_len], current_level + 1, offset+cube_len*np.array([0,1,0]))
    octant4 = BuildOctree(ot, data[:cube_len,:cube_len,cube_len:], current_level + 1, offset+cube_len*np.array([0,0,1]))
    octant5 = BuildOctree(ot, data[cube_len:,cube_len:,:cube_len], current_level + 1, offset+cube_len*np.array([1,1,0]))
    octant6 = BuildOctree(ot, data[cube_len:,:cube_len,cube_len:], current_level + 1, offset+cube_len*np.array([1,0,1]))
    octant7 = BuildOctree(ot, data[:cube_len,cube_len:,cube_len:], current_level + 1, offset+cube_len*np.array([0,1,1]))
    octant8 = BuildOctree(ot, data[cube_len:,cube_len:,cube_len:], current_level + 1, offset+cube_len*np.array([1,1,1]))

    return ot







ot = BuildOctree({},data,min_level,np.array([0,0,0]))
print ot
print len(ot)
#t = [j for j in ot.keys() if '127' in j]
#print t
#print len(t)






























"""
octant1 = data[:N/2,:N/2,:N/2]
octant2 = data[N/2:,:N/2,:N/2]

octant3 = data[:N/2,N/2:,:N/2]
octant4 = data[N/2:,:N/2,:N/2]

octant5 = data[:N/2,:N/2,:N/2]
octant6 = data[N/2:,:N/2,:N/2]
octant7 = data[:N/2,:N/2,:N/2]
octant8 = data[N/2:,:N/2,:N/2]
"""

"""
for i in [0,1]:
    for j in [0,1]:
        for k in [0, 1]:
            octant1 = data[:N/2,:N/2,:N/2]
"""







"""
#Recurisively go through each level, getting 8 octant slices of the NxNxN.
#If the octant is empty the sum is 0; add a key with value 0
#If the octant is "full" the sum is 1x(N/2)**3; add a key with value 1
#Otherwise it is mixed; do NOT add a key, go down to finer resolution
offset = [0,0,0]
power = 1
length = cube_len
for level in range(1,max_octree_level+1):
    length /= 2

    #Check if this 
    #if not filled/unfilled
    octant = data[]



    power += 1



#Generate list of all 8 slice indices we want to use:
"""




"""
doing the offset way
xx,yy,zz = np.meshgrid(np.arange(cube_len),np.arange(cube_len),np.arange(cube_len))
octant1 = BuildOctree(ot, data[slice1], current_level + 1, max_level, K, offset + K * np.array([0, 0, 0]))
octant2 = BuildOctree(ot, data[slice2], current_level + 1, max_level, K, offset + K * np.array([1, 0, 0]))
octant3 = BuildOctree(ot, data[slice3], current_level + 1, max_level, K, offset + K * np.array([0, 1, 0]))
octant4 = BuildOctree(ot, data[slice4], current_level + 1, max_level, K, offset + K * np.array([0, 0, 1]))
octant5 = BuildOctree(ot, data[slice5], current_level + 1, max_level, K, offset + K * np.array([1, 1, 0]))
octant6 = BuildOctree(ot, data[slice6], current_level + 1, max_level, K, offset + K * np.array([1, 0, 1]))
octant7 = BuildOctree(ot, data[slice7], current_level + 1, max_level, K, offset + K * np.array([0, 1, 1]))
octant8 = BuildOctree(ot, data[slice8], current_level + 1, max_level, K, offset + K * np.array([1, 1, 1]))
"""





"""octant1 = BuildOctree(ot, data[:cube_len, :cube_len, :cube_len], current_level + 1, max_level - 1,
                      offset + cube_len * np.array([0, 0, 0]))
octant2 = BuildOctree(ot, data[cube_len:, :cube_len, :cube_len], current_level + 1, max_level - 1,
                      offset + cube_len * np.array([1, 0, 0]))
octant3 = BuildOctree(ot, data[:cube_len, cube_len:, :cube_len], current_level + 1, max_level - 1,
                      offset + cube_len * np.array([0, 1, 0]))
octant4 = BuildOctree(ot, data[:cube_len, :cube_len, cube_len:], current_level + 1, max_level - 1,
                      offset + cube_len * np.array([0, 0, 1]))
octant5 = BuildOctree(ot, data[cube_len:, cube_len:, :cube_len], current_level + 1, max_level - 1,
                      offset + cube_len * np.array([1, 1, 0]))
octant6 = BuildOctree(ot, data[cube_len:, :cube_len, cube_len:], current_level + 1, max_level - 1,
                      offset + cube_len * np.array([1, 0, 1]))
octant7 = BuildOctree(ot, data[:cube_len, cube_len:, cube_len:], current_level + 1, max_level - 1,
                      offset + cube_len * np.array([0, 1, 1]))
octant8 = BuildOctree(ot, data[cube_len:, cube_len:, cube_len:], current_level + 1, max_level - 1,
                      offset + cube_len * np.array([1, 1, 1]))"""