#Visualize loss(es) from Caffe Net as a function of iterations
#for training, validation, test sets

import os
import numpy as np
import matplotlib.pyplot as plt
import re
import time

import curvox.pc_vox_utils as pcvu
def VisualizeLoss(paths_to_caffe_logs,save=True):
    """
    Make some basic graphs to visualize the loss as a function of iteration
    """

    if len(paths_to_caffe_logs) == 1:
        path_to_combined = paths_to_caffe_logs[0]

    elif len(paths_to_caffe_logs) > 1:
        timestring = time.strftime("%Y_%m_%d__%H_%M_%S", time.gmtime())
        path_to_combined = os.path.join(os.path.split(paths_to_caffe_logs[0])[0], 'caffe.log_combined__{}'.format(timestring))
        with open(path_to_combined, 'a') as ff:
            for p in paths_to_caffe_logs:
                # Open the .log file that has the standard Caffe log info
                with open(p, 'r') as gg:
                    text = gg.read()
                ff.write(text)



    #Open the combined file:
    with open(path_to_combined,'r') as ff:
        fulltext = ff.read()
        #print(fulltext)
        #fulltext = pd.read_csv(paths_to_caffe_logs,delimiter='/n')

        #loss_pattern = "Iteration (\d+) (.*), loss = (.*)"

        pattern = r"Iteration (\d+) (.*), loss = ([\d]*.[\d]*)\n" + \
                                  r"(.*)Train net output #0: loss0 = ([\d]*.[\d]*)(.*)\n" + \
                                  r"(.*)Train net output #1: loss1 = ([\d]*.[\d]*)(.*)\n" + \
                                  r"(.*)Train net output #2: loss2 = ([\d]*.[\d]*)(.*)\n" + \
                                  r"(.*)Train net output #3: loss3 = ([\d]*.[\d]*)"


        iterations = []
        total_loss = []
        loss0 = []
        loss1 = []
        loss2 = []
        loss3 = []

        for r in re.findall(pattern, fulltext):
            iterations.append(int(r[0]))
            total_loss.append(float(r[2]))
            loss0.append(float(r[4]))
            loss1.append(float(r[7]))
            loss2.append(float(r[10]))
            loss3.append(float(r[13]))

        iterations = np.array(iterations)
        total_loss = np.array(total_loss)
        loss0 = np.array(loss0)
        loss1 = np.array(loss1)
        loss2 = np.array(loss2)
        loss3 = np.array(loss3)





        #Should also get the test loss... [sum of the 4 individual losses]





    #Graph it
    plt.figure()
    plt.title('Training Loss vs. iteration',fontsize=20)
    plt.plot(iterations,total_loss,marker='s',label='Total Loss',color='k')
    plt.plot(iterations, loss0, marker='o', markeredgecolor=None, label='Loss0',color='r')
    plt.plot(iterations, loss1, marker='+', markeredgecolor=None, label='Loss1',color='g')
    plt.plot(iterations, loss2, marker='v', markeredgecolor=None, label='Loss2',color='b')
    plt.plot(iterations, loss3, marker='^', markeredgecolor=None, label='Loss3',color='c')
    plt.xlabel('Iteration',fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(numpoints=1)
    plt.show()

    if save:
        figsavepath = os.path.join(os.path.split(path_to_combined)[0],'Loss.png')
        #figsavepath = 'Loss.png'#paths_to_caffe_logs + '__Loss.png'
        plt.savefig(figsavepath)


    return [iterations, total_loss, loss0, loss1, loss2, loss3]








if __name__=="__main__":
    #paths_to_caffe_logs = ["/home/gregkocher/CLionProjects/ogn/examples/single_view_3d/shapenet_cars_128/logs/caffe.long.gregkocher.log.INFO.20170921-194259.20051"]# - Copy"
    #paths_to_caffe_logs = ["/home/gregkocher/CLionProjects/ogn/examples/single_view_3d/shapenet_cars_128/logs/caffe.long.gregkocher.log.INFO.20170922-163952.6476"]

    save = True
    paths_to_caffe_logs = ["/home/gregkocher/CLionProjects/ogn/examples/single_view_3d/shapenet_cars_128/logs/caffe.long.gregkocher.log.INFO.20170921-194259.20051",\
                           "/home/gregkocher/CLionProjects/ogn/examples/single_view_3d/shapenet_cars_128/logs/caffe.long.gregkocher.log.INFO.20170922-163952.6476",\
                           "/home/gregkocher/CLionProjects/ogn/examples/single_view_3d/shapenet_cars_128/logs/caffe.long.gregkocher.log.INFO.20170925-181801.2688",\
                            "/home/gregkocher/CLionProjects/ogn/examples/single_view_3d/shapenet_cars_128/logs/caffe.long.gregkocher.log.INFO.20170926-184948.5945"]


    out = VisualizeLoss(paths_to_caffe_logs, save=save)
    #print(iterations)
    #print(losses)