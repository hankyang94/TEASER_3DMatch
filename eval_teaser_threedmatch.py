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
ALL_ALL_CORR_LIMIT = 20000 # upper limit for all-to-all correspondences
NOISE_BOUND = 0.1
VOXEL_SIZE_INIT = 0.5
VOXEL_SIZE_STEP = 0.05
ICP_TH = 0.05
ICP_MAXITERS = 1000
TIM_INLIER_RATIO_TH = 0.8
RANSAC_MAXITERS_LOW = int(1e5)
RANSAC_MAXITERS_HIGH = int(1e6) # run RANSAC one million times

# obtain all training pairs in the training data
threedmatch_path = '../../Datasets/threedmatch'
pairs_all, pairs_30, pairs_50, pairs_70 = \
    get3DMatchTrainPairs(threedmatch_path)

# choose the catagory of pairs to register
pairs_register = pairs_70

# start registering
# GT transformation (A and B are already registered)
R_gt = np.diag([1.0,1.0,1.0])
t_gt = np.zeros(3)
nrPairs = len(pairs_register)
print(f'Total number of pairs to register: {nrPairs}.')
# things to log:
# overlap rate 1
# NNA, NNB, N 3
# R_err_ransac_low, t_err_ransac_low 2
# R_err_icp_ransac_low, t_err_icp_ransac_low 2
# R_err_ransac_high, t_err_ransac_high 2
# R_err_icp_ransac_high, t_err_icp_ransac_high 2
# R_err_teaser, t_err_teaser 2
# R_err_icp_teaser, t_err_icp_teaser 2
# teaser_tim_inlier_rate 1
# teaser_best_subopt 1
# teaser_nrMCInliers 1
# fitness_ransac_low, fitness_icp_ransac_low,  2
# fitness_ransac_high, fitness_icp_ransac_high,  2
# fitness_teaser, fitness_icp_teaser 2
nrCols = 1+3+2+2+2+2+2+2+1+1+1+2+2+2
log_results = np.zeros((nrPairs,nrCols))
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
    NNA = A_xyz.shape[0]
    NNB = B_xyz.shape[0]
    print(f'Before downsample: # of points in A: {NNA}, # of points in B: {NNB}.')
    A_pcd = o3d.geometry.PointCloud()
    B_pcd = o3d.geometry.PointCloud()
    A_pcd.points = o3d.utility.Vector3dVector(A_xyz)
    A_pcd.colors = o3d.utility.Vector3dVector(A_rgb)
    B_pcd.points = o3d.utility.Vector3dVector(B_xyz)
    B_pcd.colors = o3d.utility.Vector3dVector(B_rgb)


    # downsample A and B
    N = 2*ALL_ALL_CORR_LIMIT
    VOXEL_SIZE = VOXEL_SIZE_INIT - VOXEL_SIZE_STEP
    while N > ALL_ALL_CORR_LIMIT:
        VOXEL_SIZE = VOXEL_SIZE + VOXEL_SIZE_STEP
        A_pcd_ds = A_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        B_pcd_ds = B_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        A_ds_xyz = np.asarray(A_pcd_ds.points)
        B_ds_xyz = np.asarray(B_pcd_ds.points)
        NA = A_ds_xyz.shape[0]
        NB = B_ds_xyz.shape[0]
        N = NA * NB
        print(f'After downsample with voxel size {VOXEL_SIZE}: # of points in A: {NA}, # of points in B: {NB}, all-to-all corrs: {N}.')

    # create an all-to-all correspondence set
    A_ds_xyz = A_ds_xyz.T  # shape 3 by NA
    B_ds_xyz = B_ds_xyz.T  # shape 3 by NB
    A_corr = np.repeat(A_ds_xyz,NB,axis=1)
    B_corr = np.tile(B_ds_xyz,(1,NA))
    assert (A_corr.shape[1] == N) and (B_corr.shape[1]), 'A_corr and B_corr wrong dimension'
    print(f'Created {N} all-to-all correspondences.')


    # use RANSAC_LOW to register
    ransac_T = ransac_registration(A_corr,B_corr,
                                   NOISE_BOUND,RANSAC_MAXITERS_LOW)
    R_ransac_low = ransac_T[:3,:3]
    t_ransac_low = ransac_T[:3,3]
    fitness_ransac_low = computeFitnessScore(A_pcd_ds,B_pcd_ds,NOISE_BOUND,ransac_T)
    # compute pose error of RANSAC
    R_err_ransac_low = getRotationError(R_gt,R_ransac_low)
    t_err_ransac_low = getTranslationError(t_gt,t_ransac_low)
    print(f'RANSAC-LOW: R_err: {R_err_ransac_low}[deg], t_err: {t_err_ransac_low}[m], fitness: {fitness_ransac_low}.')

    # refine with ICP after RANSAC-LOW
    trans_init = np.identity(4)
    trans_init[:3,:3] = R_ransac_low
    trans_init[:3,3] = t_ransac_low
    icp_sol = o3d.registration.registration_icp(
            A_pcd, B_pcd, ICP_TH, trans_init,
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=ICP_MAXITERS))
    R_icp_ransac_low = icp_sol.transformation[:3,:3]
    t_icp_ransac_low = icp_sol.transformation[:3,3]
    R_err_icp_ransac_low = getRotationError(R_gt,R_icp_ransac_low)
    t_err_icp_ransac_low = getTranslationError(t_gt,t_icp_ransac_low)
    fitness_icp_ransac_low = computeFitnessScore(A_pcd,B_pcd,ICP_TH,icp_sol.transformation)
    print(f'ICP-RANSAC-LOW: R_err: {R_err_icp_ransac_low}[deg], t_err: {t_err_icp_ransac_low}[m], fitness: {fitness_icp_ransac_low}.')


    # use RANSAC_HIGH to register
    ransac_T = ransac_registration(A_corr,B_corr,
                                   NOISE_BOUND,RANSAC_MAXITERS_HIGH)
    R_ransac_high = ransac_T[:3,:3]
    t_ransac_high = ransac_T[:3,3]
    fitness_ransac_high = computeFitnessScore(A_pcd_ds,B_pcd_ds,NOISE_BOUND,ransac_T)
    # compute pose error of RANSAC
    R_err_ransac_high = getRotationError(R_gt,R_ransac_high)
    t_err_ransac_high = getTranslationError(t_gt,t_ransac_high)
    print(f'RANSAC-HIGH: R_err: {R_err_ransac_high}[deg], t_err: {t_err_ransac_high}[m], fitness: {fitness_ransac_high}.')

    # refine with ICP after RANSAC
    trans_init = np.identity(4)
    trans_init[:3,:3] = R_ransac_high
    trans_init[:3,3] = t_ransac_high
    icp_sol = o3d.registration.registration_icp(
            A_pcd, B_pcd, ICP_TH, trans_init,
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=ICP_MAXITERS))
    fitness_icp_ransac_high = computeFitnessScore(A_pcd,B_pcd,ICP_TH,icp_sol.transformation)
    R_icp_ransac_high = icp_sol.transformation[:3,:3]
    t_icp_ransac_high = icp_sol.transformation[:3,3]
    R_err_icp_ransac_high = getRotationError(R_gt,R_icp_ransac_high)
    t_err_icp_ransac_high = getTranslationError(t_gt,t_icp_ransac_high)
    print(f'ICP-RANSAC-HIGH: R_err: {R_err_icp_ransac_high}[deg], t_err: {t_err_icp_ransac_high}[m], fitness: {fitness_icp_ransac_high}.')

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
    teaser_T = Rt2T(R_teaser,t_teaser)
    # obtain number of inliers survived maximum clique
    teaser_nrMCInliers = len(solver.getInlierMaxClique())

    fitness_teaser = computeFitnessScore(A_pcd_ds,B_pcd_ds,NOISE_BOUND,teaser_T)
    # certify TEASER's result (rotation part)
    A_TIMs = solver.getMaxCliqueSrcTIMs()
    B_TIMs = solver.getMaxCliqueDstTIMs()
    theta_TIMs_raw = solver.getRotationInliersMask()
    nrRotationInliers = np.sum(theta_TIMs_raw)
    theta_TIMs = getBinaryTheta(theta_TIMs_raw)
    certifier_params = teaserpp_python.DRSCertifier.Params()
    certifier_params.cbar2 = 1.0
    certifier_params.noise_bound = 2*NOISE_BOUND
    certifier_params.sub_optimality = 1e-3
    certifier_params.max_iterations = 1e3
    certifier_params.gamma_tau = 2.0
    certifier = teaserpp_python.DRSCertifier(certifier_params)
    certificate = certifier.certify(R_teaser,A_TIMs,B_TIMs,theta_TIMs)
    teaser_best_subopt = certificate.best_suboptimality

    # compute pose error of TEASER
    teaser_tim_inlier_ratio = float(nrRotationInliers)/float(A_TIMs.shape[1])
    R_err_teaser = getRotationError(R_gt,R_teaser)
    t_err_teaser = getTranslationError(t_gt,t_teaser)
    print(f'TEASER: R_err: {R_err_teaser}[deg], t_err: {t_err_teaser}[m], fitness: {fitness_teaser}, tim_inlier_ratio: {teaser_tim_inlier_ratio}, best_subopt: {teaser_best_subopt}.')

    # refine with ICP after TEASER
    trans_init = np.identity(4)
    trans_init[:3,:3] = R_teaser
    trans_init[:3,3] = t_teaser
    icp_sol = o3d.registration.registration_icp(
            A_pcd, B_pcd, ICP_TH, trans_init,
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(max_iteration=ICP_MAXITERS))
    fitness_icp_teaser = computeFitnessScore(A_pcd,B_pcd,ICP_TH,icp_sol.transformation)
    R_icp_teaser = icp_sol.transformation[:3,:3]
    t_icp_teaser = icp_sol.transformation[:3,3]
    R_err_icp_teaser = getRotationError(R_gt,R_icp_teaser)
    t_err_icp_teaser = getTranslationError(t_gt,t_icp_teaser)
    print(f'ICP-TEASER: R_err: {R_err_icp_teaser}[deg], t_err: {t_err_icp_teaser}[m], fitness: {fitness_icp_teaser}.')

    # log results
    log_results[pair_idx,:] = np.asarray([overlap,
                                          NNA,NNB,N,
                                          R_err_ransac_low,t_err_ransac_low,
                                          R_err_icp_ransac_low,t_err_icp_ransac_low,
                                          R_err_ransac_high,t_err_ransac_high,
                                          R_err_icp_ransac_high,t_err_icp_ransac_high,
                                          R_err_teaser,t_err_teaser,
                                          R_err_icp_teaser,t_err_icp_teaser,
                                          teaser_tim_inlier_ratio,
                                          teaser_best_subopt,
                                          teaser_nrMCInliers,
                                          fitness_ransac_low,
                                          fitness_icp_ransac_low,
                                          fitness_ransac_high,
                                          fitness_icp_ransac_high,
                                          fitness_teaser,
                                          fitness_icp_teaser])

np.savetxt('results/results_70_full.txt',log_results,fmt='%.5f',delimiter=',')