# plot the results for TEASER and RANSAC using all-to-all correspondences
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib

# things to log:
# 0-overlap rate
# 1-R_err_ransac_low, 2-t_err_ransac_low
# 3-R_err_icp_ransac_low, 4-t_err_icp_ransac_low
# 5-R_err_ransac_high, 6-t_err_ransac_high
# 7-R_err_icp_ransac_high, 8-t_err_icp_ransac_high
# 9-R_err_teaser, 10-t_err_teaser
# 11-R_err_icp_teaser, 12-t_err_icp_teaser
# 13-teaser_tim_inlier_rate

TIM_INLIER_RATIO_TH = 0.80
ROTATION_SUCCESS_TH = 5.0
TRANSLATION_SUCCESS_TH = 0.1

data = np.loadtxt('results/results_70.txt',delimiter=',')

R_err_ransac_low = data[:,1]
t_err_ransac_low = data[:,2]
R_err_icp_ransac_low = data[:,3]
t_err_icp_ransac_low = data[:,4]
R_err_ransac_high = data[:,5]
t_err_ransac_high = data[:,6]
R_err_icp_ransac_high = data[:,7]
t_err_icp_ransac_high = data[:,8]
R_err_teaser = data[:,9]
t_err_teaser = data[:,10]
R_err_icp_teaser = data[:,11]
t_err_icp_teaser = data[:,12]
teaser_tim_inlier_ratio = data[:,-1]
R_err_teaser_filter = R_err_teaser[teaser_tim_inlier_ratio>TIM_INLIER_RATIO_TH]
R_err_icp_teaser_filter = R_err_icp_teaser[teaser_tim_inlier_ratio>TIM_INLIER_RATIO_TH]

nrTests = data.shape[0]
nrSuccess_ransac_low = np.sum((R_err_icp_ransac_low < ROTATION_SUCCESS_TH) * (t_err_icp_ransac_low < TRANSLATION_SUCCESS_TH))
nrSuccess_ransac_high = np.sum((R_err_icp_ransac_high < ROTATION_SUCCESS_TH) * (t_err_icp_ransac_high < TRANSLATION_SUCCESS_TH))
nrSuccess_teaser = np.sum((R_err_icp_teaser < ROTATION_SUCCESS_TH) * (t_err_icp_teaser < TRANSLATION_SUCCESS_TH))

print(f'Total tests: {nrTests}')
print(f'RANSAC 100K sucess: {nrSuccess_ransac_low}')
print(f'RANSAC 1000K success: {nrSuccess_ransac_high}')
print(f'TEASER success: {nrSuccess_teaser}')
print(f'filtered tests: {R_err_icp_teaser_filter.shape[0]}. filter ratio: {TIM_INLIER_RATIO_TH}.')

fig, ax = plt.subplots()
NUM_BINS = 500
# ax.hist(R_err_ransac_low, NUM_BINS, density=True, histtype='step',
#         cumulative=True, label='RANSAC 100K')
ax.hist(R_err_icp_ransac_low, NUM_BINS, density=True, histtype='step',
        cumulative=True, label='RANSAC 100K+ICP', linewidth=2.0, color='red')
# ax.hist(R_err_ransac_high, NUM_BINS, density=True, histtype='step',
#         cumulative=True, label='RANSAC 1000K')
ax.hist(R_err_icp_ransac_high, NUM_BINS, density=True, histtype='step',
        cumulative=True, label='RANSAC 1000K+ICP',linewidth=2.0, color='cyan')
# ax.hist(R_err_teaser, NUM_BINS, density=True, histtype='step',
#         cumulative=True, label='TEASER')
ax.hist(R_err_icp_teaser, NUM_BINS, density=True, histtype='step',
        cumulative=True, label='TEASER+ICP',linewidth=2.0, color='magenta')

ax.hist(R_err_icp_teaser_filter, NUM_BINS, density=True, histtype='step',
        cumulative=True, label='TEASER Filter+ICP',linewidth=2.0, color='blue')
ax.legend(loc='best')
ax.set_xlabel('Rotation Error [deg]')
ax.set_ylabel('Cumulative Distribution')

plt.xscale('log')

# ax.boxplot([R_err_ransac_low,
#             R_err_icp_ransac_low,
#             R_err_ransac_high,
#             R_err_icp_ransac_low,
#             R_err_teaser,
#             R_err_icp_teaser,
#             R_err_teaser_filter,
#             R_err_icp_teaser_filter])

plt.show()

print(f'done')
