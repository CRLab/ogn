#After running the test net, convert the output to a viewable format and also do basic numeric checks

import os
import subprocess
import numpy as np
from sklearn import metrics
import pandas as pd

import binvox_rw

import sys
sys.path.insert(0,"/home/gregkocher/CLionProjects/ogn/python/rendering")
import python_octree





def AnalyzeOutput_SingleFile(test_ot_path,truth_ot_path,view=True,ignore_truth=False):
    """
    Convert the .ot output file to a binvox file and optionally view the 3d model using viewvox.
    Also, load the ground truth .ot file to do basic numerical analysis.


    test_ot_path - path to output .ot file

    truth_ot_path - path to corresponding ground truth .ot file

    e.g.
    test_ot_path = '/home/gregkocher/CLionProjects/ogn/examples/shape_from_id/shapenet_cars_256/output/0065.ot'
    truth_ot_path = '/home/gregkocher/CLionProjects/ogn/data/shapenet_cars/256_l4/0065.ot'
    IOU, NMI = AnalyzeOutput_SingleFile(test_ot_path,truth_ot_path,view=True)
    """


    #Load the .ot file that is output from the OGN test net, use the authors' scripts to convert to numpy array
    #npy_out_path = test_ot_path.replace('.ot','_voxels.npy')
    ot, resolution = python_octree.import_ot(test_ot_path)
    output = python_octree.octree_to_voxel_grid(ot, resolution)
    # print(output)
    # print(output.sum())
    # print(output.size)
    #np.save(npy_out_path,output)


    #Now use binvox_rw python module to convert the numpy array to .binvox file
    occupancy_3d_array = output.astype(np.int)
    #binvox_rw.Voxels(occupancy_3d_array, data, dims, translate, scale, axis_order)
    voxel_model = binvox_rw.Voxels(occupancy_3d_array, [resolution, resolution, resolution], [0, 0, 0], 0, 'xzy')
    #Output as a binvox file
    test_binvox_path = test_ot_path[:-3] + '.binvox'
    with open(test_binvox_path,'wb') as fp:
        binvox_rw.write(voxel_model, fp, fast=True)
        #binvox_rw.write(voxel_model, fp, fast=False)




    IOU = None
    NMI = None
    if not ignore_truth:

        #Also load the ground truth .ot file for comparison
        truth_ot, truth_resolution = python_octree.import_ot(truth_ot_path)
        truth_output = python_octree.octree_to_voxel_grid(truth_ot, truth_resolution).astype(np.int)

        #Now use binvox_rw python module to convert the numpy array to .binvox file
        truth_occupancy_3d_array = truth_output.astype(np.int)
        #binvox_rw.Voxels(occupancy_3d_array, data, dims, translate, scale, axis_order)
        truth_voxel_model = binvox_rw.Voxels(truth_occupancy_3d_array, [truth_resolution, truth_resolution, truth_resolution], [0, 0, 0], 0, 'xzy')
        #Output as a binvox file, put it in the test output dir side by side with test output binvox
        truth_binvox_path = test_binvox_path.replace('.binvox','_truth.binvox')
        with open(truth_binvox_path,'wb') as fp:
            binvox_rw.write(truth_voxel_model, fp, fast=True)
            #binvox_rw.write(truth_voxel_model, fp, fast=False)



        #Now do some basic numerical comparisons of output to ground truth:
        #IOU, Jake's mesh Hausdorff....
        #intersection_over_union(gt, pr)
        IOU = python_octree.intersection_over_union(truth_occupancy_3d_array.flatten(), occupancy_3d_array.flatten())
        NMI = metrics.normalized_mutual_info_score(truth_occupancy_3d_array.flatten(), occupancy_3d_array.flatten())
        print('IOU', IOU)
        print('NMI', NMI)
        #print('AMI', metrics.adjusted_mutual_info_score(truth_occupancy_3d_array.flatten(), occupancy_3d_array.flatten()))






    #Optionally view the binvox files using viewvox
    if view:
        viewvox_path = '/home/gregkocher/CLionProjects/viewvox/viewvox'
        test_cmd = '{0} {1}'.format(viewvox_path,test_binvox_path)
        test_process = subprocess.Popen(test_cmd, shell=True)

        if not ignore_truth:
            truth_cmd = '{0} {1}'.format(viewvox_path, truth_binvox_path)
            truth_process = subprocess.Popen(truth_cmd, shell=True)




    return IOU, NMI












def AnalyzeOutput_WholeDirectory(out_dir_path,truth_dir_path):
    """
    For every file in directory, convert to binvox files and do basic numerical analysis.
    Just runs "AnalyzeOutput_SingleFile" function on each output file in directory and
    combines the analysis in single place.
    """

    out_filelist = os.listdir(out_dir_path)
    truth_filelist = os.listdir(truth_dir_path)
    out_filelist.sort()
    truth_filelist.sort()
    out_filelist = [os.path.join(out_dir_path,i) for i in out_filelist if i.endswith('.ot')]
    truth_filelist = [os.path.join(truth_dir_path, i) for i in truth_filelist if i.endswith('.ot')]

    IOU_scores = []
    NMI_scores = []
    outs = []
    truths = []
    for i, out_ot_path in enumerate(out_filelist):
        try:
            end = os.path.split(out_ot_path)[1]
            truth_ot_path = [k for k in truth_filelist if k.endswith(end)][0]
            IOU, NMI = AnalyzeOutput_SingleFile(out_ot_path, truth_ot_path, view=False)
            IOU_scores += [IOU]
            NMI_scores += [NMI]
            outs += [out_ot_path]
            truths += [truth_ot_path]
        except:
            continue

        print(i)
        print(out_ot_path)
        # print(truth_ot_path)
        # print(IOU_scores)
        # print(NMI_scores)
        # if i==2: break


    #Output csv of values for later
    # For each sample outfile, the paths, IOU, NMI, etc.
    df = pd.DataFrame(data={'GroundTruth':truths, 'Prediction':outs, 'IOU':IOU_scores, 'NMI':NMI_scores})
    df = df[['GroundTruth','Prediction','IOU','NMI']]
    df.to_csv(os.path.join(out_dir_path,'Metrics.csv'),index=False)






#Test this function
if __name__ == "__main__":
    # SHape from ID output:
    # number = '0002'
    # view=True#False
    # test_ot_path = '/home/gregkocher/CLionProjects/ogn/examples/shape_from_id/shapenet_cars_256/output/{}.ot'.format(number)
    # truth_ot_path = '/home/gregkocher/CLionProjects/ogn/data/shapenet_cars/256_l4/{}.ot'.format(number)
    # IOU, NMI = AnalyzeOutput_SingleFile(test_ot_path,truth_ot_path,view=view)

    # out_dir_path = '/home/gregkocher/CLionProjects/ogn/examples/shape_from_id/shapenet_cars_256/output/'
    # truth_dir_path = '/home/gregkocher/CLionProjects/ogn/data/shapenet_cars/256_l4/'
    # AnalyzeOutput_WholeDirectory(out_dir_path, truth_dir_path)





    #single view (3d from 2d) at 50K iterations
    number = '0050'
    view=True#False
    test_ot_path = '/home/gregkocher/CLionProjects/ogn/examples/single_view_3d/shapenet_cars_128/output/50K_iters_output/{}.ot'.format(number)
    truth_ot_path = None
    IOU, NMI = AnalyzeOutput_SingleFile(test_ot_path,truth_ot_path,view=view,ignore_truth=True)
    #single view (3d from 2d) at 172K iterations
    view=True#False
    test_ot_path = '/home/gregkocher/CLionProjects/ogn/examples/single_view_3d/shapenet_cars_128/output/172K_iters_output/{}.ot'.format(number)
    truth_ot_path = None
    IOU, NMI = AnalyzeOutput_SingleFile(test_ot_path,truth_ot_path,view=view,ignore_truth=True)