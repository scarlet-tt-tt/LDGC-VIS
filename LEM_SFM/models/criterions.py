from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as func
import LEM_SFM.models.geometry as geo

def EPE3D_loss(input_flow, target_flow, invalid=None):
    """
    :param the estimated optical / scene flow
    :param the ground truth / target optical / scene flow
    :param the invalid mask, the mask has value 1 for all areas that are invalid
    """
    epe_map = torch.norm(target_flow-input_flow,p=2,dim=1)
    B = epe_map.shape[0]

    invalid_flow = (target_flow != target_flow) # check Nan same as torch.isnan

    mask = (invalid_flow[:,0,:,:] | invalid_flow[:,1,:,:] | invalid_flow[:,2,:,:]) 
    if invalid is not None:
        mask = mask | (invalid.view(mask.shape) > 0)

    epes = []
    for idx in range(B):
        epe_sample = epe_map[idx][~mask[idx].data]
        if len(epe_sample) == 0:
            epes.append(torch.zeros(()).type_as(input_flow))
        else:
            epes.append(epe_sample.mean()) 

    return torch.stack(epes)

def RPE(R, t):
    """ Calcualte the relative pose error 
    (a batch version of the RPE error defined in TUM RGBD SLAM TUM dataset)
    :param relative rotation
    :param relative translation
    """
    angle_error = geo.batch_mat2angle(R)
    trans_error = torch.norm(t, p=2, dim=1) 
    return angle_error, trans_error

def compute_RPE_loss(R_est, t_est, R_gt, t_gt):
    """
    :param estimated rotation matrix Bx3x3
    :param estimated translation vector Bx3
    :param ground truth rotation matrix Bx3x3
    :param ground truth translation vector Bx3
    """ 
    # print('R_est, t_est',R_est[0], t_est[0])
    # print('R_gt, t_gt',R_gt[0], t_gt[0])
    dR, dt = geo.batch_Rt_between(R_est, t_est, R_gt, t_gt)
    # print('dR, dt',dR[0], dt[0])
    angle_error, trans_error = RPE(dR, dt)
    # print('angle_error, trans_error',angle_error[0], trans_error[0])
    return angle_error, trans_error

def compute_RT_EPE_loss(R_est, t_est, R_gt, t_gt, depth0_gt, K, invalid=None): 
    """ Compute the epe point error of rotation & translation
    :param estimated rotation matrix Bx3x3
    :param estimated translation vector Bx3
    :param ground truth rotation matrix Bx3x3
    :param ground truth translation vector Bx3
    :param reference depth image, 
    :param camera intrinsic 
    """
    
    loss = 0
    if R_est.dim() > 3: # training time [batch, num_poses, rot_row, rot_col]
        rH, rW = 120, 160 # we train the algorithm using a downsized input, (since the size of the input is not super important at training time)

        B,C,H,W = depth0_gt.shape
        rdepth = func.interpolate(depth0_gt, size=(rH, rW), mode='bilinear')
        rinvalid = func.interpolate(invalid.float(), size=(rH,rW), mode='bilinear')
        rK = K.clone()
        rK[:,0] *= float(rW) / W
        rK[:,1] *= float(rH) / H
        rK[:,2] *= float(rW) / W
        rK[:,3] *= float(rH) / H
        xyz = geo.batch_inverse_project(rdepth, rK)
        flow_gt = geo.batch_transform_xyz(xyz, R_gt, t_gt, get_Jacobian=False)

        for idx in range(R_est.shape[1]):
            flow_est= geo.batch_transform_xyz(xyz, R_est[:,idx], t_est[:,idx], get_Jacobian=False)
            loss += EPE3D_loss(flow_est, flow_gt.detach(), rinvalid) #* (1<<idx) scaling does not help that much
    else:
        # print(depth0_gt.device)
        # print(K.device)
        xyz = geo.batch_inverse_project(depth0_gt, K)
        flow_gt = geo.batch_transform_xyz(xyz, R_gt, t_gt, get_Jacobian=False)

        flow_est= geo.batch_transform_xyz(xyz, R_est, t_est, get_Jacobian=False)
        loss = EPE3D_loss(flow_est, flow_gt, invalid)

    return loss

def  compute_depth_loss():
    
    return 0


# class MyPoseLoss(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.flow_loss = compute_RT_EPE_loss

#     def forward(self,R_infer, t_infer_s, R_gt, t_gt, depth0_gt, K, invalid_mask_0):
#         epes3d = self.flow_loss( R_infer, t_infer_s, R_gt, t_gt, 1.0/depth0_gt, K, invalid_mask_0).mean() * 1e2

#         return epes3d
    
# class MyDepthLoss(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def forward(self,depth_list0_s,depth_list1_s,depth0_gt,depth1_gt,invalid_mask_0,invalid_mask_1):
#         from relative_depth.lossx import InvariantLoss
#         criterion = InvariantLoss()
#         a = 1
#         depth_loss_init_0= criterion(depth_list0_s[0], depth0_gt,mask = ~invalid_mask_0)*a
#         depth_loss_del1_0= criterion(depth_list0_s[1], depth0_gt,mask = ~invalid_mask_0)*a
#         depth_loss_del2_0= criterion(depth_list0_s[2], depth0_gt,mask = ~invalid_mask_0)*a
#         depth_loss_del3_0= criterion(depth_list0_s[3], depth0_gt,mask = ~invalid_mask_0)*a

#         depth_loss_init_1= criterion(depth_list1_s[0], depth1_gt,mask = ~invalid_mask_0)*a
#         depth_loss_del1_1= criterion(depth_list1_s[1], depth1_gt,mask = ~invalid_mask_0)*a
#         depth_loss_del2_1= criterion(depth_list1_s[2], depth1_gt,mask = ~invalid_mask_0)*a
#         depth_loss_del3_1= criterion(depth_list1_s[3], depth1_gt,mask = ~invalid_mask_0)*a
#         alpha =[1,1,1]
#         loss_depth_0 = depth_loss_del1_0*alpha[0]+depth_loss_del2_0*alpha[1]+depth_loss_del3_0*alpha[2]
#         loss_depth_1 = depth_loss_del1_1*alpha[0]+depth_loss_del2_1*alpha[1]+depth_loss_del3_1*alpha[2]
#         DepthLoss = loss_depth_0+loss_depth_1
        
#         return DepthLoss