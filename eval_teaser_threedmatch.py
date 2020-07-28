# use brute-force correspondences to register 3D Match training data
# enumerate all possible pairs of the training data
# compare two algorithms:
# TEASER
# RANSAC

import os
import numpy as np
import open3d as o3d
import teaserpp_python
import time
import scipy.io
from helpers import *

# hyper parameters
NOISE_BOUND = 0.1
VOXEL_SIZE = 0.5
ICP_TH = 0.05
TIM_INLIER_RATIO_TH = 0.8
RANSAC_MAXITERS = int(1e5)

# obtain all training pairs in the training data
threedmatch_path = '../../Datasets/threedmatch'
pairs_all, pairs_30, pairs_50, pairs_70 = \
    get3DMatchTrainPairs(threedmatch_path)

# choose the catagory of pairs to register
pairs_register = pairs_70[:500]

# start registering
# GT transformation (A and B are already registered)
R_gt = np.diag([1.0,1.0,1.0])
t_gt = np.zeros(3)
nrPairs = len(pairs_register)
print(f'Total number of pairs to register: {nrPairs}.')
# things to log:
# overlap rate
# R_err_ransac, t_err_ransac
# R_err_icp_ransac, t_err_icp_ransac
# R_err_teaser, t_err_teaser
# R_err_icp_teaser, t_err_icp_teaser
# teaser_tim_inlier_rate
log_results = np.zeros((nrPairs,1+2+2+2+2+1))
for pair_idx, pair in enumerate(pairs_register):
    A_path = pair[0]
    B_path = pair[1]
    overlap = float(pair[2])
    print(f'Register pair {pair_idx}: {A_path} and {B_path}, overlap: {overlap}.')

    # load the point clouds from .npz files
    cloudA = np.load(os.path.join(threedmatch_path,A_path))
    cloudB = np.load(os.path.join(threedmatch_path,B_path))
    A_xyz = cloudA['pcd']
    A_rgb = cloudA['color']
    B_xyz = cloudB['pcd']
    B_rgb = cloudB['color']
    print(f'Before downsample: # of points in A: {A_xyz.shape[0]}, # of points in B: {B_xyz.shape[0]}.')
    A_pcd = o3d.geometry.PointCloud()
    B_pcd = o3d.geometry.PointCloud()
    A_pcd.points = o3d.utility.Vector3dVector(A_xyz)
    A_pcd.colors = o3d.utility.Vector3dVector(A_rgb)
    B_pcd.points = o3d.utility.Vector3dVector(B_xyz)
    B_pcd.colors = o3d.utility.Vector3dVector(B_rgb)

    # downsample A and B
    A_pcd_ds = A_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    B_pcd_ds = B_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    A_ds_xyz = np.asarray(A_pcd_ds.points)
    B_ds_xyz = np.asarray(B_pcd_ds.points)
    print(f'After downsample with voxel size {VOXEL_SIZE}: # of points in A: {A_ds_xyz.shape[0]}, # of points in B: {B_ds_xyz.shape[0]}.')

    # create an all-to-all correspondence set
    A_ds_xyz = A_ds_xyz.T  # shape 3 by NA
    B_ds_xyz = B_ds_xyz.T  # shape 3 by NB
    NA = A_ds_xyz.shape[1]
    NB = B_ds_xyz.shape[1]
    N = NA * NB
    A_corr = np.repeat(A_ds_xyz,NB,axis=1)
    B_corr = np.tile(B_ds_xyz,(1,NA))
    assert (A_corr.shape[1] == N) and (B_corr.shape[1]), 'A_corr and B_corr wrong dimension'
    print(f'Created {N} all-to-all correspondences.')

    # use RANSAC to register
    ransac_T = ransac_registration(A_corr,B_corr,
                                   NOISE_BOUND,RANSAC_MAXITERS)
    R_ransac = ransac_T[:3,:3]
    t_ransac = ransac_T[:3,3]
    # compute pose error of RANSAC
    R_err_ransac = getRotationError(R_gt,R_ransac)
    t_err_ransac = getTranslationError(t_gt,t_ransac)
    print(f'RANSAC: R_err: {R_err_ransac}[deg], t_err: {t_err_ransac}[m]')

    # refine with ICP after RANSAC
    trans_init = np.identity(4)
    trans_init[:3,:3] = R_ransac
    trans_init[:3,3] = t_ransac
    icp_sol = o3d.registration.registration_icp(
            A_pcd, B_pcd, ICP_TH, trans_init,
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=500))
    R_icp_ransac = icp_sol.transformation[:3,:3]
    t_icp_ransac = icp_sol.transformation[:3,3]
    R_err_icp_ransac = getRotationError(R_gt,R_icp_ransac)
    t_err_icp_ransac = getTranslationError(t_gt,t_icp_ransac)
    print(f'ICP-RANSAC: R_err: {R_err_icp_ransac}[deg], t_err: {t_err_icp_ransac}[m]')

    # use TEASER to register
    # create a TEASER solver
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_tim_graph = teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    # solver_params.rotation_tim_graph = teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.COMPLETE
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(A_corr,B_corr)
    solution = solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation

    A_TIMs = solver.getMaxCliqueSrcTIMs()
    nrRotationInliers = np.sum(solver.getRotationInliersMask())
    teaser_tim_inlier_ratio = float(nrRotationInliers)/float(A_TIMs.shape[1])
    # compute pose error of TEASER
    R_err_teaser = getRotationError(R_gt,R_teaser)
    t_err_teaser = getTranslationError(t_gt,t_teaser)
    print(f'TEASER: R_err: {R_err_teaser}[deg], t_err: {t_err_teaser}[m], tim_inlier_ratio: {teaser_tim_inlier_ratio}.')

    # refine with ICP after TEASER
    trans_init = np.identity(4)
    trans_init[:3,:3] = R_teaser
    trans_init[:3,3] = t_teaser
    icp_sol = o3d.registration.registration_icp(
            A_pcd, B_pcd, ICP_TH, trans_init,
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=500))
    R_icp_teaser = icp_sol.transformation[:3,:3]
    t_icp_teaser = icp_sol.transformation[:3,3]
    R_err_icp_teaser = getRotationError(R_gt,R_icp_teaser)
    t_err_icp_teaser = getTranslationError(t_gt,t_icp_teaser)
    print(f'ICP-TEASER: R_err: {R_err_icp_teaser}[deg], t_err: {t_err_icp_teaser}[m]')

    # log results
    log_results[pair_idx,:] = np.asarray([overlap,
                                          R_err_ransac,t_err_ransac,
                                          R_err_icp_ransac,t_err_icp_ransac,
                                          R_err_teaser,t_err_teaser,
                                          R_err_icp_teaser,t_err_icp_teaser,
                                          teaser_tim_inlier_ratio])

np.savetxt('results/results_70.txt',log_results,fmt='%.5f',delimiter=',')