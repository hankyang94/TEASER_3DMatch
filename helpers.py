# helper functions for
import numpy as np
import os
import glob
import open3d as o3d

def getRotationError(R,R_est):
    error = np.abs( np.arccos( (np.trace(R.T @ R_est) - 1)/2.0 ) )
    return error/np.pi * 180 # return rotation error in degrees

def getTranslationError(t,t_est):
    return np.linalg.norm(t - t_est)

def getBinaryTheta(theta):
    N = theta.shape[0]
    output = -1.0 * np.ones(N)
    output[theta] = 1.0 
    return output

def get3DMatchTrainPairs(path):
    dir = os.getcwd()
    os.chdir(path)
    txtFiles = glob.glob("*.txt")
    pairs_all = []
    pairs_30 = []
    pairs_50 = []
    pairs_70 = []
    print(f'Found {len(txtFiles)} .txt files in {path}.')
    for file in txtFiles:
        if file.endswith("0.30.txt"):
            pairs_30.append(file)
        elif file.endswith("0.50.txt"):
            pairs_50.append(file)
        elif file.endswith("0.70.txt"):
            pairs_70.append(file)
        else:
            pairs_all.append(file)

    pairs_all.sort()
    pairs_30.sort()
    pairs_50.sort()
    pairs_70.sort()

    pairs_all_info = []
    for fileName in pairs_all:
        file = os.path.join(path,fileName)
        with open(file) as f:
            lines = [line.rstrip().split() for line in f]
        pairs_all_info += lines

    pairs_30_info = []
    for fileName in pairs_30:
        file = os.path.join(path,fileName)
        with open(file) as f:
            lines = [line.rstrip().split() for line in f]
        pairs_30_info += lines

    pairs_50_info = []
    for fileName in pairs_50:
        file = os.path.join(path,fileName)
        with open(file) as f:
            lines = [line.rstrip().split() for line in f]
        pairs_50_info += lines

    pairs_70_info = []
    for fileName in pairs_70:
        file = os.path.join(path,fileName)
        with open(file) as f:
            lines = [line.rstrip().split() for line in f]
        pairs_70_info += lines

    os.chdir(dir)
    
    return pairs_all_info, pairs_30_info, pairs_50_info, pairs_70_info

def make_o3d_pointcloud(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def ransac_registration(xyz0,xyz1,distance_threshold,iterations):
    '''
    xyz0 and xyz1: numpy array of size 3 by N
    '''
    pcd0 = make_o3d_pointcloud(xyz0.T)
    pcd1 = make_o3d_pointcloud(xyz1.T)
    nrCorrs = xyz0.shape[1]
    idx0 = np.arange(nrCorrs)
    idx1 = idx0
    corres = np.stack((idx0,idx1), axis=1)
    corres = o3d.utility.Vector2iVector(corres)

    result = o3d.registration.registration_ransac_based_on_correspondence(
        pcd0, pcd1, corres, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 3,
        o3d.registration.RANSACConvergenceCriteria(iterations, 40000000))
    return result.transformation

def computeFitnessScore(pcd0,pcd1,distance_threshold,transformation):
    evaluation = o3d.registration.evaluate_registration(pcd0, pcd1, distance_threshold, transformation)
    return evaluation.fitness

def Rt2T(R,t):
    T = np.identity(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def weightedProcrustes(xyzA,xyzB,weights):
    '''
    weighted procrustes
    xyzA and xyzB are numpy arrays of dimension (3,N)
    weights is a numpy array of dimension (N)
    '''
    N = xyzA.shape[1]
    centerA = np.matmul(xyzA,weights) / np.sum(weights)
    centerB = np.matmul(xyzB,weights) / np.sum(weights)

    xyzA_ref = (xyzA - centerA[:,None]) * np.sqrt(weights)
    xyzB_ref = (xyzB - centerB[:,None]) * np.sqrt(weights)

    # compute rotation from SVD (Wahba problem)
    M = np.zeros([3,3])
    for i in range(N):
        ai = xyzA_ref[:,i]
        bi = xyzB_ref[:,i]
        M = M + np.outer(bi,ai)
    U,S,Vh = np.linalg.svd(M)
    R = U @ np.diag([1,1,np.linalg.det(U)*np.linalg.det(Vh)]) @ Vh

    # recover translation
    t = centerB - R @ centerA
    return R, t

if __name__ == "__main__":
    threedmatch_path = '../../Datasets/threedmatch'
    pairs_all_info, pairs_30_info, pairs_50_info, pairs_70_info = \
        get3DMatchTrainPairs(threedmatch_path)

    print(f'done')
