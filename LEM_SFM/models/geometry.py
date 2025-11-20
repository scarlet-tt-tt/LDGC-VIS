from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch import sin, cos, atan2, acos

import numpy as np

_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def meshgrid(H, W, B=None, is_cuda=False):
    """ torch version of numpy meshgrid function

    :input
    :param height
    :param width
    :param batch size
    :param initialize a cuda tensor if true
    -------
    :return 
    :param meshgrid in column
    :param meshgrid in row
    """
    u = torch.arange(0, W)
    v = torch.arange(0, H)

    if is_cuda:
        u, v = u.cuda(), v.cuda()

    u = u.repeat(H, 1).view(1,H,W)
    v = v.repeat(W, 1).t_().view(1,H,W)

    if B is not None:
        u, v = u.repeat(B,1,1,1), v.repeat(B,1,1,1)
    return u, v

def generate_xy_grid(B, H, W, K):
    """ Generate a batch of image grid from image space to world space 
        px = (u - cx) / fx
        py = (y - cy) / fy

        function tested in 'test_geometry.py'

    :input
    :param batch size
    :param height
    :param width
    :param camera intrinsic array [fx,fy,cx,cy] 
    ---------
    :return 
    :param 
    :param 
    """

    """从图像空间到世界空间生成一批图像网格
    px=（u-cx）/fx
    py=（y-cy）/fy

    在“test_geometry.py”中测试的功能

    ：输入
    ：param批量大小
    ：参数高度
    ：参数宽度
    ：参数相机固有阵列[fx，fy，cx，cy]
    ---------
    ：返回
    ：param
    ：param
    """
    # print('=> K2 : ', K)
    fx, fy, cx, cy = K.split(1,dim=1)
    # fx=K[0]
    # fy=K[1]
    # cx=K[2]
    # cy=K[3]
    # fy= torch.tensor(K[1])
    # cx= torch.tensor(K[2])
    # cy= torch.tensor(K[3])
    # print(K)

    uv_grid = meshgrid(H, W, B)
    u_grid, v_grid = [uv.type_as(cx) for uv in uv_grid]
    # # print(u_grid.view(B,-1))
    # # print(cx)
    # temp1 = (u_grid.view(B,-1) - cx)
    # # print(temp1)
    # # print(fx)
    # temp2 = temp1/fx
    # # print(temp2)
    px = ((u_grid.view(B,-1) - cx) / fx).view(B,1,H,W)
    py = ((v_grid.view(B,-1) - cy) / fy).view(B,1,H,W)
    return px, py

def batch_inverse_Rt(R, t):
    """ The inverse of the R, t: [R' | -R't] 

        function tested in 'test_geometry.py'

    :input 
    :param rotation Bx3x3
    :param translation Bx3
    ----------
    :return 
    :param rotation inverse Bx3x3
    :param translation inverse Bx3
    """
    R_t = R.transpose(1,2)
    t_inv = -torch.bmm(R_t, t.contiguous().view(-1, 3, 1))

    return R_t, t_inv.view(-1,3)

def batch_Rt_compose(d_R, d_t, R0, t0):
    """ Compose operator of R, t: [d_R*R | d_R*t + d_t] 
        We use left-mulitplication rule here. 

        function tested in 'test_geometry.py'
    
    :input
    :param rotation incremental Bx3x3
    :param translation incremental Bx3
    :param initial rotation Bx3x3
    :param initial translation Bx3
    ----------
    :return 
    :param composed rotation Bx3x3
    :param composed translation Bx3
    """
    R1 = d_R.bmm(R0)
    t1 = d_R.bmm(t0.view(-1,3,1)) + d_t.view(-1,3,1)
    return R1, t1.view(-1,3)

def batch_Rt_between(R0, t0, R1, t1): 
    """ Between operator of R, t, transform of T_0=[R0, t0] to T_1=[R1, t1]
        which is T_1 \compose T^{-1}_0 

        function tested in 'test_geometry.py'
    
    :input 
    :param rotation of source Bx3x3
    :param translation of source Bx3
    :param rotation of target Bx3x3
    :param translation of target Bx3
    ----------
    :return 
    :param incremental rotation Bx3x3
    :param incremnetal translation Bx3
    """
    R0t = R0.transpose(1,2)
    # R0t= np.transpose(R0, (0, 2, 1))
    dR = R1.bmm(R0t)
    dt = t1.view(-1,3) - dR.bmm(t0.view(-1,3,1)).view(-1,3)
    return dR, dt

def batch_skew(w):
    """ Generate a batch of skew-symmetric matrices. 

        function tested in 'test_geometry.py'

    :input
    :param skew symmetric matrix entry Bx3
    ---------
    :return 
    :param the skew-symmetric matrix Bx3x3
    """
    B, D = w.size()
    assert(D == 3)
    o = torch.zeros(B).type_as(w)
    w0, w1, w2 = w[:, 0], w[:, 1], w[:, 2]
    return torch.stack((o, -w2, w1, w2, o, -w0, -w1, w0, o), 1).view(B, 3, 3)

def batch_twist2Mat(twist):
    """ The exponential map from so3 to SO3

        Calculate the rotation matrix using Rodrigues' Rotation Formula
        http://electroncastle.com/wp/?p=39 
        or Ethan Eade's lie group note:
        http://ethaneade.com/lie.pdf equation (13)-(15) 

        @todo: may rename the interface to batch_so3expmap(twist)

        functioned tested with cv2.Rodrigues implementation in 'test_geometry.py'

    :input
    :param twist/axis angle Bx3 \in \so3 space 
    ----------
    :return 
    :param Rotation matrix Bx3x3 \in \SO3 space
    """
    B = twist.size()[0]
    theta = twist.norm(p=2, dim=1).view(B, 1)
    w_so3 = twist / theta.expand(B, 3)
    W = batch_skew(w_so3)
    return torch.eye(3).repeat(B,1,1).type_as(W) \
        + W*sin(theta.view(B,1,1)) \
        + W.bmm(W)*(1-cos(theta).view(B,1,1))

def batch_mat2angle(R):
    """ Calcuate the axis angles (twist) from a batch of rotation matrices

        Ethan Eade's lie group note:
        http://ethaneade.com/lie.pdf equation (17)

        function tested in 'test_geometry.py'

    :input
    :param Rotation matrix Bx3x3 \in \SO3 space
    --------
    :return 
    :param the axis angle B
    """
    R1 = [torch.trace(R[i]) for i in range(R.size()[0])]
    R_trace = torch.stack(R1)
    # clamp if the angle is too large (break small angle assumption)
    # @todo: not sure whether it is absoluately necessary in training. 
    angle = acos( ((R_trace - 1)/2).clamp(-1,1))
    return angle

def batch_mat2twist(R):
    """ The log map from SO3 to so3

        Calculate the twist vector from Rotation matrix 

        Ethan Eade's lie group note:
        http://ethaneade.com/lie.pdf equation (18)

        @todo: may rename the interface to batch_so3logmap(R)

        function tested in 'test_geometry.py'

        @note: it currently does not consider extreme small values. 
        If you use it as training loss, you may run into problems

    :input
    :param Rotation matrix Bx3x3 \in \SO3 space 
    --------
    :param the twist vector Bx3 \in \so3 space
    """
    B = R.size()[0]
 
    R1 = [torch.trace(R[i]) for i in range(R.size()[0])]
    tr = torch.stack(R1)
    theta = acos( ((tr - 1)/2).clamp(-1,1) )

    r11,r12,r13,r21,r22,r23,r31,r32,r33 = torch.split(R.view(B,-1),1,dim=1)
    res = torch.cat([r32-r23, r13-r31, r21-r12],dim=1)  

    magnitude = (0.5*theta/sin(theta))

    return magnitude.view(B,1) * res

def batch_warp_inverse_depth(p_x, p_y, p_invD, pose, K):
    """ Compute the warping grid w.r.t. the SE3 transform given the inverse depth

    :input
    :param p_x the x coordinate map
    :param p_y the y coordinate map
    :param p_invD the inverse depth
    :param pose the 3D transform in SE3
    :param K the intrinsics
    --------
    :return 
    :param projected u coordinate in image space Bx1xHxW
    :param projected v coordinate in image space Bx1xHxW
    :param projected inverse depth Bx1XHxW
    """
    assert not torch.isnan(p_x).any(), "p_x contains NaN values!"
    assert not torch.isinf(p_x).any(), "p_x contains Inf values!"
    assert not torch.isnan(p_invD).any(), "p_x contains Inf values!"
    # print(p_invD)
    # input()
    # print(p_x)
    # input()
    # print(p_y)
    # input()

    [R, t] = pose
    R = torch.clamp(R, min=-10, max=10)
    t = torch.clamp(t, min=-10, max=10)
    
    B, _, H, W = p_x.shape

    I = torch.ones((B,1,H,W)).type_as(p_invD)
    x_y_1 = torch.cat((p_x, p_y, I), dim=1)

    warped = torch.bmm(R, x_y_1.view(B,3,H*W)) + \
        t.view(B,3,1).expand(B,3,H*W) * p_invD.view(B, 1, H*W).expand(B,3,H*W)
    # print('warped',warped)
    # input()


    x_, y_, s_ = torch.split(warped, 1, dim=1)
    fx, fy, cx, cy = torch.split(K, 1, dim=1)

    # print(x_)

    assert not torch.isnan(x_).any(), "x_ contains NaN values!"
    assert not torch.isnan(y_).any(), "y_ contains NaN values!"
    assert not torch.isnan(s_).any(), "s_ contains NaN values!"
    assert not torch.isinf(x_).any(), "x_ contains Inf values!"
    assert not torch.isinf(y_).any(), "y_ contains Inf values!"
    assert not torch.isinf(s_).any(), "s_ contains Inf values!"

    eps = 1e-6
    s_ = torch.clamp(s_, min=eps)
    # print(f"s_.min(): {s_.min()}, s_.max(): {s_.max()}")
    # print(f"y_.min(): {y_.min()}, y_.max(): {y_.max()}")
    u_ = (x_ / s_+eps).view(B,-1) * fx + cx
    v_ = (y_ / s_+eps).view(B,-1) * fy + cy

    inv_z_ = p_invD / s_.view(B,1,H,W)

    return u_.view(B,1,H,W), v_.view(B,1,H,W), inv_z_

def batch_warp_affine(pu, pv, affine):
    # A = affine[:,:,:2]
    # t = affine[:,:, 2]
    B,_,H,W = pu.shape
    ones = torch.ones(pu.shape).type_as(pu)
    uv = torch.cat((pu, pv, ones), dim=1)
    uv = torch.bmm(affine, uv.view(B,3,-1)) #+ t.view(B,2,1)
    return uv[:,0].view(B,1,H,W), uv[:,1].view(B,1,H,W)

def check_occ(inv_z_buffer, inv_z_ref, u, v, thres=1e-1):
    """ z-buffering check of occlusion 
    :param inverse depth of target frame
    :param inverse depth of reference frame
    """
    B, _, H, W = inv_z_buffer.shape

    inv_z_warped = warp_features(inv_z_ref, u, v)
    inlier = (inv_z_buffer > inv_z_warped - thres)

    inviews = inlier & (u > 0) & (u < W) & \
        (v > 0) & (v < H)
    return ~inviews

def warp_features(F, u, v):
    """
    Warp the feature map (F) w.r.t. the grid (u, v)
    """
    B, C, H, W = F.shape

    u_norm = u / ((W-1)/2) - 1
    v_norm = v / ((H-1)/2) - 1
    # print("*********custom_grid_sample***********")
    uv_grid = torch.cat((u_norm.view(B,H,W,1), v_norm.view(B,H,W,1)), dim=3)

    # F_warped = custom_grid_sample(F, uv_grid)
    # print("grid_sample after",F_warped.shape)
    F_warped = nn.functional.grid_sample(F, uv_grid,
        mode='bilinear', padding_mode='border',align_corners=True)
    # print("grid_sample",F_warped.shape)
    # print("*********after custom_grid_sample***********")

    return F_warped

# def custom_grid_sample(input, grid):
    # # 确保输入是PyTorch张量
    # if not isinstance(input, torch.Tensor) or not isinstance(grid, torch.Tensor):
    #     raise TypeError("input和grid必须是PyTorch张量")
    
    # N, C, H_in, W_in = input.shape
    # N, H_out, W_out, _ = grid.shape
    
    # # 初始化输出张量（使用PyTorch的zeros，匹配输入的dtype和device）
    # output = torch.zeros((N, C, H_out, W_out), dtype=input.dtype, device=input.device)
    
    # # 预计算坐标转换系数
    # w_scale = (W_in - 1) / 2.0
    # h_scale = (H_in - 1) / 2.0
    
    # # 为边界处理创建标量张量（确保与输入在同一设备）
    # zero = torch.tensor(0, dtype=torch.int32, device=input.device)
    # w_max = torch.tensor(W_in - 1, dtype=torch.int32, device=input.device)
    # h_max = torch.tensor(H_in - 1, dtype=torch.int32, device=input.device)
    
    # # 遍历批次和通道
    # for i in range(N):
    #     for j in range(C):
    #         # 取出当前批次和通道的输入数据
    #         input_slice = input[i, j]  # 形状: (H_in, W_in)
            
    #         # 取出当前批次的坐标网格
    #         x_grid = grid[i, ..., 0]  # 形状: (H_out, W_out)
    #         y_grid = grid[i, ..., 1]  # 形状: (H_out, W_out)
            
    #         # 坐标转换（基础算术运算）
    #         x_coords = (x_grid + 1.0) * w_scale
    #         y_coords = (y_grid + 1.0) * h_scale
            
    #         # 计算整数坐标（用类型转换替代floor）
    #         x0 = x_coords.to(dtype=torch.int32)
    #         x1 = x0 + 1
    #         y0 = y_coords.to(dtype=torch.int32)
    #         y1 = y0 + 1
            
    #         # 边界处理（使用clamp替代max+min组合，避免维度问题）
    #         x0 = torch.clamp(x0, zero, w_max)
    #         x1 = torch.clamp(x1, zero, w_max)
    #         y0 = torch.clamp(y0, zero, h_max)
    #         y1 = torch.clamp(y1, zero, h_max)
            
    #         # 计算插值权重
    #         wx = x_coords - x0.to(dtype=input.dtype)
    #         wy = y_coords - y0.to(dtype=input.dtype)
    #         wx1 = 1.0 - wx
    #         wy1 = 1.0 - wy
            
    #         # 获取四个角落的像素值（基础索引操作）
    #         top_left = input_slice[y0, x0]
    #         top_right = input_slice[y0, x1]
    #         bottom_left = input_slice[y1, x0]
    #         bottom_right = input_slice[y1, x1]
            
    #         # 双线性插值计算
    #         output[i, j] = (top_left * wx1 * wy1 +
    #                        top_right * wx * wy1 +
    #                        bottom_left * wx1 * wy +
    #                        bottom_right * wx * wy)
    
    # return output
    
# def custom_grid_sample(input, grid):
#     N, C, H_in, W_in = input.shape
#     N, H_out, W_out, _ = grid.shape
#     output = np.random.random((N, C, H_out, W_out))
#     for i in range(N):
#         for j in range(C):
#             for k in range(H_out):
#                 for l in range(W_out):
#                     x, y = grid[i][k][l][0], grid[i][k][l][1]
#                     param = [0.0, 0.0]
#                     param[0] = (W_in - 1) * (x + 1) / 2
#                     param[1] = (H_in - 1) * (y + 1) / 2
#                     x1 = int(param[0] + 1)
#                     x0 = x1 - 1
#                     y1 = int(param[1] + 1)
#                     y0 = y1 - 1
#                     param[0] = abs(param[0] - x0)
#                     param[1] = abs(param[1] - y0)
#                     left_top_value, left_bottom_value, right_top_value, right_bottom_value = 0, 0, 0, 0
#                     if 0 <= x0 < W_in and 0 <= y0 < H_in:
#                         left_top_value = input[i][j][y0][x0]
#                     if 0 <= x1 < W_in and 0 <= y0 < H_in:
#                         right_top_value = input[i][j][y0][x1]
#                     if 0 <= x0 < W_in and 0 <= y1 < H_in:
#                         left_bottom_value = input[i][j][y1][x0]
#                     if 0 <= x1 < W_in and 0 <= y1 < H_in:
#                         right_bottom_value = input[i][j][y1][x1]
#                     left_top = left_top_value * (1 - param[0]) * (1 - param[1])
#                     left_bottom = left_bottom_value * (1 - param[0]) * param[1]
#                     right_top = right_top_value * param[0] * (1 - param[1])
#                     right_bottom = right_bottom_value * param[0] * param[1]
#                     result = left_bottom + left_top + right_bottom + right_top
#                     output[i][j][k][l] = result
#     return output

# def custom_grid_sample(F, uv_grid, align_corners=True):
#     """
#     移除 Floor 算子的 RKNN 兼容 grid_sample（用 trunc 替代）
#     Args:
#         F: 输入特征图，shape=(N, C, H, W)
#         uv_grid: 采样网格，shape=(N, H_out, W_out, 2)（x/y 为归一化坐标 [-1,1]）
#         align_corners: 与原 grid_sample 保持一致
#     Returns:
#         F_warped: 采样后特征图，shape=(N, C, H_out, W_out)
#     """
#     N, C, H, W = F.shape
#     N_grid, H_out, W_out, _ = uv_grid.shape
#     assert N == N_grid, "批量数不匹配：F与uv_grid的N需一致"
#     max_flat_idx = H * W - 1  # 展平后的最大合法索引

#     # 步骤1：归一化坐标 → 像素坐标
#     if align_corners:
#         x = (uv_grid[..., 0] + 1) * (W - 1) / 2
#         y = (uv_grid[..., 1] + 1) * (H - 1) / 2
#     else:
#         x = (uv_grid[..., 0] + 1) * W / 2
#         y = (uv_grid[..., 1] + 1) * H / 2

#     # 步骤2：边界处理（彻底钳位）
#     x = torch.clamp(x, 0.0, float(W - 1))
#     y = torch.clamp(y, 0.0, float(H - 1))

#     # -------------------------- 关键修改：用 trunc 替代 floor --------------------------
#     # i0 = torch.floor(y).long()  # 原代码（含 Floor 算子，RKNN 不支持）
#     # j0 = torch.floor(x).long()
#     # i0 = torch.trunc(y).long()  # 替代方案1：trunc（截断小数，非负场景与 floor 一致）
#     # j0 = torch.trunc(x).long()
#     # （备选）替代方案2：floor_divide（向下整除1.0，与 floor 一致）
#     i0 = torch.floor_divide(y, 1.0).long()
#     j0 = torch.floor_divide(x, 1.0).long()
#     # -----------------------------------------------------------------------------------

#     # 步骤3：处理 i1/j1（避免越界）
#     i1 = torch.clamp(i0 + 1, 0, H - 1)
#     j1 = torch.clamp(j0 + 1, 0, W - 1)

#     # 步骤4：双线性插值权重计算
#     w_y0 = 1.0 - (y - i0.float())
#     w_y1 = y - i0.float()
#     w_x0 = 1.0 - (x - j0.float())
#     w_x1 = x - j0.float()
#     w_y0 = w_y0.unsqueeze(1)
#     w_y1 = w_y1.unsqueeze(1)
#     w_x0 = w_x0.unsqueeze(1)
#     w_x1 = w_x1.unsqueeze(1)

#     # 步骤5：计算展平索引（断言防越界）
#     idx_i0j0 = i0 * W + j0
#     idx_i0j1 = i0 * W + j1
#     idx_i1j0 = i1 * W + j0
#     idx_i1j1 = i1 * W + j1
#     assert torch.all(idx_i0j0 >= 0) and torch.all(idx_i0j0 <= max_flat_idx), "i0j0索引越界"
#     assert torch.all(idx_i0j1 >= 0) and torch.all(idx_i0j1 <= max_flat_idx), "i0j1索引越界"
#     assert torch.all(idx_i1j0 >= 0) and torch.all(idx_i1j0 <= max_flat_idx), "i1j0索引越界"
#     assert torch.all(idx_i1j1 >= 0) and torch.all(idx_i1j1 <= max_flat_idx), "i1j1索引越界"

#     # 步骤6：gather 提取像素值
#     F_flat = F.view(N, C, H * W)
#     idx_i0j0 = idx_i0j0.unsqueeze(1).view(N, 1, H_out * W_out)
#     idx_i0j1 = idx_i0j1.unsqueeze(1).view(N, 1, H_out * W_out)
#     idx_i1j0 = idx_i1j0.unsqueeze(1).view(N, 1, H_out * W_out)
#     idx_i1j1 = idx_i1j1.unsqueeze(1).view(N, 1, H_out * W_out)
#     F_i0j0 = torch.gather(F_flat, dim=2, index=idx_i0j0.expand(-1, C, -1))
#     F_i0j1 = torch.gather(F_flat, dim=2, index=idx_i0j1.expand(-1, C, -1))
#     F_i1j0 = torch.gather(F_flat, dim=2, index=idx_i1j0.expand(-1, C, -1))
#     F_i1j1 = torch.gather(F_flat, dim=2, index=idx_i1j1.expand(-1, C, -1))

#     # 步骤7：加权求和
#     F_i0j0 = F_i0j0.view(N, C, H_out, W_out)
#     F_i0j1 = F_i0j1.view(N, C, H_out, W_out)
#     F_i1j0 = F_i1j0.view(N, C, H_out, W_out)
#     F_i1j1 = F_i1j1.view(N, C, H_out, W_out)
#     F_warped = (w_y0 * (w_x0 * F_i0j0 + w_x1 * F_i0j1)) + (w_y1 * (w_x0 * F_i1j0 + w_x1 * F_i1j1))

#     return F_warped




# def custom_grid_sample(input, grid):
#     """
#     A PyTorch implementation of the custom grid sample function.
#     Args:
#         input (torch.Tensor): Input tensor of shape (N, C, H_in, W_in).
#         grid (torch.Tensor): Grid tensor of shape (N, H_out, W_out, 2).
#     Returns:
#         torch.Tensor: Output tensor of shape (N, C, H_out, W_out).
#     """
#     # 获取输入和网格的形状
#     N, C, H_in, W_in = input.shape
#     _, H_out, W_out, _ = grid.shape

#     # 初始化输出张量
#     output = torch.zeros((N, C, H_out, W_out), device=input.device)

#     # 计算网格对应的输入坐标
#     x = (grid[..., 0] + 1) * (W_in - 1) / 2
#     y = (grid[..., 1] + 1) * (H_in - 1) / 2

#     # 计算四个邻近点的索引
#     x0 = torch.floor(x).long()
#     x1 = torch.clamp(x0 + 1, 0, W_in - 1)
#     x0 = torch.clamp(x0, 0, W_in - 1)

#     y0 = torch.floor(y).long()
#     y1 = torch.clamp(y0 + 1, 0, H_in - 1)
#     y0 = torch.clamp(y0, 0, H_in - 1)

#     # 计算插值权重
#     dx = x - x0
#     dy = y - y0
#     w_tl = (1 - dx) * (1 - dy)  # top-left
#     w_tr = dx * (1 - dy)        # top-right
#     w_bl = (1 - dx) * dy        # bottom-left
#     w_br = dx * dy              # bottom-right

#     # 遍历通道并计算插值
#     for c in range(C):
#         input_c = input[:, c, :, :]  # 提取当前通道的输入
#         # 获取四个邻近点的值
#         v_tl = input_c.gather(1, y0.unsqueeze(1).expand(-1, 1, H_out, W_out)).gather(2, x0.unsqueeze(1).expand(-1, 1, H_out, W_out))
#         v_tr = input_c.gather(1, y0.unsqueeze(1).expand(-1, 1, H_out, W_out)).gather(2, x1.unsqueeze(1).expand(-1, 1, H_out, W_out))
#         v_bl = input_c.gather(1, y1.unsqueeze(1).expand(-1, 1, H_out, W_out)).gather(2, x0.unsqueeze(1).expand(-1, 1, H_out, W_out))
#         v_br = input_c.gather(1, y1.unsqueeze(1).expand(-1, 1, H_out, W_out)).gather(2, x1.unsqueeze(1).expand(-1, 1, H_out, W_out))

#         # 计算输出
#         output[:, c, :, :] = (
#             w_tl * v_tl +
#             w_tr * v_tr +
#             w_bl * v_bl +
#             w_br * v_br
#         )

#     return output

# def custom_grid_sample(input, grid):
#     N, C, H_in, W_in = input.shape
#     N, H_out, W_out, _ = grid.shape
#     output = np.random.random((N, C, H_out, W_out))
#     for i in range(N):
#         for j in range(C):
#             for k in range(H_out):
#                 for l in range(W_out):
#                     x, y = grid[i][k][l][0], grid[i][k][l][1]
#                     param = [0.0, 0.0]
#                     param[0] = (W_in - 1) * (x + 1) / 2
#                     param[1] = (H_in - 1) * (y + 1) / 2
#                     x1 = int(param[0] + 1)
#                     x0 = x1 - 1
#                     y1 = int(param[1] + 1)
#                     y0 = y1 - 1
#                     param[0] = abs(param[0] - x0)
#                     param[1] = abs(param[1] - y0)
#                     left_top_value, left_bottom_value, right_top_value, right_bottom_value = 0, 0, 0, 0
#                     if 0 <= x0 < W_in and 0 <= y0 < H_in:
#                         left_top_value = input[i][j][y0][x0]
#                     if 0 <= x1 < W_in and 0 <= y0 < H_in:
#                         right_top_value = input[i][j][y0][x1]
#                     if 0 <= x0 < W_in and 0 <= y1 < H_in:
#                         left_bottom_value = input[i][j][y1][x0]
#                     if 0 <= x1 < W_in and 0 <= y1 < H_in:
#                         right_bottom_value = input[i][j][y1][x1]
#                     left_top = left_top_value * (1 - param[0]) * (1 - param[1])
#                     left_bottom = left_bottom_value * (1 - param[0]) * param[1]
#                     right_top = right_top_value * param[0] * (1 - param[1])
#                     right_bottom = right_bottom_value * param[0] * param[1]
#                     result = left_bottom + left_top + right_bottom + right_top
#                     output[i][j][k][l] = result
#     return output


# def custom_grid_sample(F, uv_grid, mode='bilinear', padding_mode='border'):
#     # 获取输入张量的形状
#     N, C, H, W = F.shape
#     _, H_out, W_out, _ = uv_grid.shape

#     # 将 uv_grid 的范围从 [-1, 1] 转换为图像坐标
#     x = ((uv_grid[..., 0] + 1) * 0.5) * (W - 1)
#     y = ((uv_grid[..., 1] + 1) * 0.5) * (H - 1)

#     # 双线性插值
#     x0 = torch.floor(x).long()
#     x1 = x0 + 1
#     y0 = torch.floor(y).long()
#     y1 = y0 + 1

#     # 边界处理
#     x0 = torch.clamp(x0, 0, W - 1)
#     x1 = torch.clamp(x1, 0, W - 1)
#     y0 = torch.clamp(y0, 0, H - 1)
#     y1 = torch.clamp(y1, 0, H - 1)

#     # 计算插值权重
#     wa = (x1.float() - x) * (y1.float() - y)
#     wb = (x1.float() - x) * (y - y0.float())
#     wc = (x - x0.float()) * (y1.float() - y)
#     wd = (x - x0.float()) * (y - y0.float())

#     # 根据权重计算采样值
#     batch_indices = torch.arange(N, device=F.device)[:, None, None]
#     Ia = F[batch_indices, :, y0, x0]
#     Ib = F[batch_indices, :, y1, x0]
#     Ic = F[batch_indices, :, y0, x1]
#     Id = F[batch_indices, :, y1, x1]

#     # 扩展权重维度以匹配通道数
#     wa = wa.unsqueeze(1)  # [N, 1, H_out, W_out]
#     wb = wb.unsqueeze(1)
#     wc = wc.unsqueeze(1)
#     wd = wd.unsqueeze(1)

#     # 合并结果
#     out = (wa * Ia + wb * Ib + wc * Ic + wd * Id)

#     return out

def batch_transform_xyz(xyz_tensor, R, t, get_Jacobian=True):
    '''
    transform the point cloud w.r.t. the transformation matrix
    :param xyz_tensor: B * 3 * H * W
    :param R: rotation matrix B * 3 * 3
    :param t: translation vector B * 3
    '''
    B, C, H, W = xyz_tensor.size()
    t_tensor = t.contiguous().view(B,3,1).repeat(1,1,H*W)
    p_tensor = xyz_tensor.contiguous().view(B, C, H*W)
    # the transformation process is simply:
    # p' = t + R*p
    xyz_t_tensor = torch.baddbmm(t_tensor, R, p_tensor)

    if get_Jacobian:
        # return both the transformed tensor and its Jacobian matrix
        J_r = R.bmm(batch_skew_symmetric_matrix(-1*p_tensor.permute(0,2,1)))
        J_t = -1 * torch.eye(3).view(1,3,3).expand(B,3,3)
        J = torch.cat((J_r, J_t), 1)
        return xyz_t_tensor.view(B, C, H, W), J
    else:
        return xyz_t_tensor.view(B, C, H, W)

def flow_from_rigid_transform(depth, extrinsic, intrinsic):
    """
    Get the optical flow induced by rigid transform [R,t] and depth
    """
    [R, t] = extrinsic
    [fx, fy, cx, cy] = intrinsic

def batch_project(xyz_tensor, K):
    """ Project a point cloud into pixels (u,v) given intrinsic K
    [u';v';w] = [K][x;y;z]
    u = u' / w; v = v' / w

    :param the xyz points 
    :param calibration is a torch array composed of [fx, fy, cx, cy]
    -------
    :return u, v grid tensor in image coordinate
    (tested through inverse project)
    """
    B, _, H, W = xyz_tensor.size()
    batch_K = K.expand(H, W, B, 4).permute(2,3,0,1)

    x, y, z = torch.split(xyz_tensor, 1, dim=1)
    fx, fy, cx, cy = torch.split(batch_K, 1, dim=1)

    u = fx*x / z + cx
    v = fy*y / z + cy
    return torch.cat((u,v), dim=1)

def batch_inverse_project(depth, K):
    """ Inverse project pixels (u,v) to a point cloud given intrinsic 
    :param depth dim B*H*W
    :param calibration is torch array composed of [fx, fy, cx, cy]
    :param color (optional) dim B*3*H*W
    -------
    :return xyz tensor (batch of point cloud)
    (tested through projection)
    """
    if depth.dim() == 3:
        B, H, W = depth.size()
    else: 
        B, _, H, W = depth.size()

    x, y = generate_xy_grid(B,H,W,K)
    z = depth.view(B,1,H,W)
    return torch.cat((x*z, y*z, z), dim=1)

def batch_euler2mat(ai, aj, ak, axes='sxyz'):
    """ A torch implementation euler2mat from transform3d:
    https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    :param ai : First rotation angle (according to `axes`).
    :param aj : Second rotation angle (according to `axes`).
    :param ak : Third rotation angle (according to `axes`).
    :param axes : Axis specification; one of 24 axis sequences as string or encoded tuple - e.g. ``sxyz`` (the default).
    -------
    :return rotation matrix, array-like shape (B, 3, 3)

    Tested w.r.t. transforms3d.euler module
    """
    B = ai.size()[0]

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]
    order = [i, j, k]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = sin(ai), sin(aj), sin(ak)
    ci, cj, ck = cos(ai), cos(aj), cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    # M = torch.zeros(B, 3, 3).cuda()
    if repetition:
        c_i = [cj, sj*si, sj*ci]
        c_j = [sj*sk, -cj*ss+cc, -cj*cs-sc]
        c_k = [-sj*ck, cj*sc+cs, cj*cc-ss]
    else:
        c_i = [cj*ck, sj*sc-cs, sj*cc+ss]
        c_j = [cj*sk, sj*ss+cc, sj*cs-sc]
        c_k = [-sj, cj*si, cj*ci]

    def permute(X): # sort X w.r.t. the axis indices
        return [ x for (y, x) in sorted(zip(order, X)) ]

    c_i = permute(c_i)
    c_j = permute(c_j)
    c_k = permute(c_k)

    r =[torch.stack(c_i, 1),
        torch.stack(c_j, 1),
        torch.stack(c_k, 1)]
    r = permute(r)

    return torch.stack(r, 1)

def batch_mat2euler(M, axes='sxyz'): 
    """ A torch implementation euler2mat from transform3d:
    https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    :param array-like shape (3, 3) or (4, 4). Rotation matrix or affine.
    :param  Axis specification; one of 24 axis sequences as string or encoded tuple - e.g. ``sxyz`` (the default).
    --------
    :returns 
    :param ai : First rotation angle (according to `axes`).
    :param aj : Second rotation angle (according to `axes`).
    :param ak : Third rotation angle (according to `axes`).
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if repetition:
        sy = torch.sqrt(M[:, i, j]**2 + M[:, i, k]**2)
        # A lazy way to cope with batch data. Can be more efficient
        mask = ~(sy > 1e-8) 
        ax = atan2( M[:, i, j],  M[:, i, k])
        ay = atan2( sy,          M[:, i, i])
        az = atan2( M[:, j, i], -M[:, k, i])
        if mask.sum() > 0:
            ax[mask] = atan2(-M[:, j, k][mask], M[:, j, j][mask])
            ay[mask] = atan2( sy[mask],         M[:, i, i][mask])
            az[mask] = 0.0
    else:
        cy = torch.sqrt(M[:, i, i]**2 + M[:, j, i]**2)
        mask = ~(cy > 1e-8)
        ax = atan2( M[:, k, j],  M[:, k, k])
        ay = atan2(-M[:, k, i],  cy)
        az = atan2( M[:, j, i],  M[:, i, i])
        if mask.sum() > 0:
            ax[mask] = atan2(-M[:, j, k][mask],  M[:, j, j][mask])
            ay[mask] = atan2(-M[:, k, i][mask],  cy[mask])
            az[mask] = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az
