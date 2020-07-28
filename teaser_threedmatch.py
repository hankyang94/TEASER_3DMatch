# use brute-force correspondences to register 3D Match training data
import os
import numpy as np
import open3d as o3d
import teaserpp_python
import time
import scipy.io
from helpers import *

NOISE_BOUND = 0.1
VOXEL_SIZE = 0.5
ICP_TH = 0.05
RANSAC_MAXITERS = int(1e5)

threedmatch_path = '../../Datasets/threedmatch'
# 7-scenes-chess@seq-01_003.npz 7-scenes-chess@seq-01_004.npz
A_path = '7-scenes-chess@seq-01_000.npz'
B_path = '7-scenes-chess@seq-01_002.npz'
# B_path = '7-scenes-chess@seq-01_003.npz'

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

# o3d.visualization.draw_geometries([A_pcd, B_pcd])
# o3d.visualization.draw_geometries([B_pcd])

# downsample A and B
A_pcd_ds = A_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
B_pcd_ds = B_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

A_ds_xyz = np.asarray(A_pcd_ds.points)
B_ds_xyz = np.asarray(B_pcd_ds.points)

print(f'After downsample with voxel size {VOXEL_SIZE}: # of points in A: {A_ds_xyz.shape[0]}, # of points in B: {B_ds_xyz.shape[0]}.')
# o3d.visualization.draw_geometries([A_pcd_ds,B_pcd_ds])

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

# GT transformation (A and B are already registered)
R_gt = np.diag([1.0,1.0,1.0])
t_gt = np.zeros(3)

# # register with RANSAC
# ransac_T = ransac_registration(A_corr,B_corr,
#                                NOISE_BOUND,RANSAC_MAXITERS)
# R_ransac = ransac_T[:3,:3]
# t_ransac = ransac_T[:3,3]
# # compute pose error of RANSAC
# R_err_ransac = getRotationError(R_gt,R_ransac)
# t_err_ransac = getTranslationError(t_gt,t_ransac)
# print(f'RANSAC: R_err: {R_err_ransac}[deg], t_err: {t_err_ransac}[m]')


# register with TEASER
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
# disable MaxClique
# solver_params.inlier_selection_mode = teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.NONE
# KCORE heuristics
# solver_params.inlier_selection_mode = teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.KCORE_HEU

solver = teaserpp_python.RobustRegistrationSolver(solver_params)
start = time.time()
solver.solve(A_corr,B_corr)
end = time.time()
solver_time = end - start
nrMCInliers = len(solver.getInlierMaxClique())
solution = solver.getSolution()
R_est = solution.rotation
t_est = solution.translation
A_TIMs = solver.getMaxCliqueSrcTIMs()
B_TIMs = solver.getMaxCliqueDstTIMs()
theta_TIMs_raw = solver.getRotationInliersMask()
nrRotationInliers = np.sum(theta_TIMs_raw)
theta_TIMs = getBinaryTheta(theta_TIMs_raw)
tim_inlier_ratio = float(nrRotationInliers)/float(theta_TIMs.shape[0])
print(f'TEASER: MCInliers: {nrMCInliers}, TIMS: {theta_TIMs.shape[0]}, TIMInliers: {nrRotationInliers}, solverTime: {solver_time}[s].')

# scipy.io.savemat('test_false.mat', dict(A_TIMs=A_TIMs, B_TIMs=B_TIMs, theta_TIMs=theta_TIMs, NOISE_BOUND=NOISE_BOUND, R_est=R_est))

# create a TEASER certifier
certifier_params = teaserpp_python.DRSCertifier.Params()
certifier_params.cbar2 = 1.0
certifier_params.noise_bound = 2*NOISE_BOUND
certifier_params.sub_optimality = 1e-3
certifier_params.max_iterations = 1e3
certifier_params.gamma_tau = 2.0
certifier = teaserpp_python.DRSCertifier(certifier_params)

# use the certifier to certify GNC-TLS
start = time.time()
certificate = certifier.certify(R_est,A_TIMs,B_TIMs,theta_TIMs)
end = time.time()
certifier_time = end - start
best_subopt = certificate.best_suboptimality
print(f'TEASER Certifier: best subopt: {best_subopt*100}%, iterations: {len(certificate.suboptimality_traj)}.')

# compute pose error of TEASER
R_err = getRotationError(R_gt,R_est)
t_err = getTranslationError(t_gt,t_est)
print(f'TEASER: R_err: {R_err}[deg], t_err: {t_err}[m], tim_inlier_ratio: {tim_inlier_ratio}.')

# refine with ICP
trans_init = np.identity(4)
trans_init[:3,:3] = R_est
trans_init[:3,3] = t_est
icp_sol = o3d.registration.registration_icp(
        A_pcd, B_pcd, ICP_TH, trans_init,
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=500))
R_icp = icp_sol.transformation[:3,:3]
t_icp = icp_sol.transformation[:3,3]
R_err_icp = getRotationError(R_gt,R_icp)
t_err_icp = getTranslationError(t_gt,t_icp)
print(f'ICP: R_err: {R_err_icp}[deg], t_err: {t_err_icp}[m]')


