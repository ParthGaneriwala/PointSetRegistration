import numpy as np
import transforms3d as t3d
from probreg import cpd
import copy
from probreg import callbacks
from output import utils
import open3d as o3
import logging
log = logging.getLogger('probreg')
log.setLevel(logging.DEBUG)

# source, target = utils.prepare_source_and_target_rigid_3d('data/cloud_0.pcd')
#
# cbs = [callbacks.Open3dVisualizerCallback(source, target)]
# tf_param, _, _ = cpd.registration_cpd(source, target,
#                                       callbacks=cbs)
#
# print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rot)),tf_param.scale, tf_param.t)


source, target = utils.prepare_source_and_target_rigid_3d('data/bunny.pcd')

vis = o3.visualization.Visualizer()
vis.create_window()
result = copy.deepcopy(source)
source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])
vis.add_geometry(source)
vis.add_geometry(target)
vis.add_geometry(result)
threshold = 0.05
icp_iteration = 2000
save_image = False

for i in range(icp_iteration):
    reg_p2p = o3.pipelines.registration.registration_icp(result, target, threshold,
                np.identity(4), o3.pipelines.registration.TransformationEstimationPointToPoint(),
                o3.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
    result.transform(reg_p2p.transformation)
    vis.update_geometry(source)
    vis.update_geometry(target)
    vis.update_geometry(result)
    vis.poll_events()
    vis.update_renderer()
    if save_image:
        vis.capture_screen_image("image_%04d.jpg" % i)
vis.run()
