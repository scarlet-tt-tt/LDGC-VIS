"""
The algorithm backbone, primarily the three contributions proposed in our paper

@author: Zhaoyang Lv
@date: March, 2019
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as func

import LEM_SFM.models.geometry as geometry
from LEM_SFM.models.submodules import convLayer as conv
from LEM_SFM.models.submodules import fcLayer, initialize_weights



# class DepthUpdate(nn.Module):
#     def __init__(self, 
#                  max_iter,
#                  solver_depth_func,
#                  timers      = None):
#         super(DepthUpdate,self).__init__()
#         self.max_iterations = max_iter
#         self.solver_depth_func = solver_depth_func
#         self.timers         = timers
#     def forward(self,depth,):
#         for idx in range(self.max_iterations):

#             if self.timers: self.timers.tic('update depth')
#             depth = self.solver_depth_func(depth)
#             if self.timers: self.timers.toc('update depth')

#             # if self.timers: self.timers.tic('compute warping residuals')
#             # residuals, occ = compute_warped_residual(pose, invD0, invD1, \
#             #     x0, x1, px, py, K)
#             # if self.timers: self.timers.toc('compute warping residuals')

#         return depth



# class DirectDepthSolverNet(nn.Module):


#     def __init__(self):
#         super(DirectDepthSolverNet, self).__init__()


#     def forward(self, depth):

#         depth_update = depth

#         return depth_update
class cal_delta_depth(nn.Module):
    def __init__(self, func0,func1):
        super(cal_delta_depth, self).__init__()
        self.func0 = func0
        self.func1 = func1
    def forward(self,residuals0,residuals1, x0, x1, d0, d1):

        cat0 = torch.cat((residuals0, x0, x1, d0, d1), dim=1)
        cat0 = torch.nn.functional.interpolate(cat0,scale_factor=2)
        out0 =self.func0(cat0)

        cat1 = torch.cat((residuals1,  x1,x0,  d1,d0), dim=1)
        cat1 = torch.nn.functional.interpolate(cat1,scale_factor=2)
        out1 =self.func1(cat1)

        return out0,out0
    




class TrustRegionBase(nn.Module):
    """ 
    This is the the base function of the trust-region based inverse compositional algorithm. 
    """
    def __init__(self,
        max_iter    = 3,
        mEst_func   = None,
        solver_func = None,
        timers      = None):
        """
        :param max_iter, maximum number of iterations
        :param mEst_func, the M-estimator function / network 
        :param solver_func, the trust-region function / network
        :param timers, if yes, counting time for each step
        """
        super(TrustRegionBase, self).__init__()

        self.max_iterations = max_iter
        self.mEstimator     = mEst_func
        self.directSolver   = solver_func
        self.timers         = timers

    def forward(self, pose, x0, x1, invD0, invD1, K, wPrior=None,feature0_infer=None):
        """
        :param pose, the initial pose
            (extrinsic of the target frame w.r.t. the referenc frame)
        :param x0, the template features
        :param x1, the image features
        :param invD0, the template inverse depth
        :param invD1, the image inverse depth
        :param K, the intrinsic parameters, [fx, fy, cx, cy]
        :param wPrior (optional), provide an initial weight as input to the convolutional m-estimator
        """
        B, C, H, W = x0.shape
        px, py = geometry.generate_xy_grid(B,H,W,K)

        # 13
        if self.timers: self.timers.tic('pre-compute Jacobians')
        J_F_p = self.precompute_Jacobian(invD0, x0, px, py, K)
        if self.timers: self.timers.toc('pre-compute Jacobians')

        # 12
        if self.timers: self.timers.tic('compute warping residuals')
        residuals, occ = compute_warped_residual(pose, invD0, invD1, \
            x0, x1, px, py, K)
        residuals = feature0_infer*residuals+residuals
        if self.timers: self.timers.toc('compute warping residuals')

        # 16
        if self.timers: self.timers.tic('robust estimator')
        weights = self.mEstimator(residuals, x0, x1, wPrior)
        wJ = weights.view(B,-1,1) * J_F_p
        if self.timers: self.timers.toc('robust estimator')

        if self.timers: self.timers.tic('pre-compute JtWJ')
        JtWJ = torch.bmm(torch.transpose(J_F_p, 1, 2) , wJ)
        if self.timers: self.timers.toc('pre-compute JtWJ')

        # 循环优化
        for idx in range(self.max_iterations):
            if self.timers: self.timers.tic('solve x=A^{-1}b')
            # print('循环优化',residuals.shape)
            pose = self.directSolver(JtWJ,
                torch.transpose(J_F_p,1,2), weights, residuals,
                pose, invD0, invD1, x0, x1, K)
            if self.timers: self.timers.toc('solve x=A^{-1}b')
    
            if self.timers: self.timers.tic('compute warping residuals')
            residuals, occ = compute_warped_residual(pose, invD0, invD1, \
                x0, x1, px, py, K)
            residuals = feature0_infer*residuals+residuals
            if self.timers: self.timers.toc('compute warping residuals')


        return pose, weights 


    def inverse_pose(self,R, t):
        # 计算旋转矩阵的逆（正交矩阵的逆等于其转置）
        R_inv = R.transpose(-2, -1)
        # 计算逆平移向量
        t_inv = -torch.matmul(R_inv, t.unsqueeze(-1)).squeeze(-1)
        # 构建 4x4 齐次逆变换矩阵
        batch_size = R.shape[0]
        T_inv = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        T_inv[:, :3, :3] = R_inv
        T_inv[:, :3, 3] = t_inv
        return [R_inv,t_inv]
    
    def precompute_Jacobian(self, invD, x, px, py, K):
        """ Pre-compute the image Jacobian on the reference frame
        refer to equation (13) in the paper
        
        :param invD, template depth
        :param x, template feature
        :param px, normalized image coordinate in cols (x)
        :param py, normalized image coordinate in rows (y)
        :param K, the intrinsic parameters, [fx, fy, cx, cy]

        ------------
        :return precomputed image Jacobian on template
        """
        """在参考系上预先计算图像雅可比矩阵
        参考论文中的方程式（13）
                
        ：参数invD，模板深度
        ：参数x，模板特征
        ：param-px，cols（x）中的归一化图像坐标
        ：参数py，行（y）中的归一化图像坐标
        ：参数K，固有参数[fx，fy，cx，cy]

        ------------
        ：返回模板上预先计算的图像雅可比矩阵
        """
        Jf_x, Jf_y = feature_gradient(x)
        Jx_p, Jy_p = compute_jacobian_warping(invD, K, px, py)
        J_F_p = compute_jacobian_dIdp(Jf_x, Jf_y, Jx_p, Jy_p)
        return J_F_p


class TrustRegionBase_i(nn.Module):
    """ 
    This is the the base function of the trust-region based inverse compositional algorithm. 
    """
    def __init__(self,
        max_iter    = 3,
        mEst_func   = None,
        solver_func = None,
        timers      = None):
        """
        :param max_iter, maximum number of iterations
        :param mEst_func, the M-estimator function / network 
        :param solver_func, the trust-region function / network
        :param timers, if yes, counting time for each step
        """
        super(TrustRegionBase_i, self).__init__()

        self.max_iterations = max_iter
        self.mEstimator     = mEst_func
        self.directSolver   = solver_func
        self.timers         = timers

    def forward(self, pose, x0, x1, invD0, invD1, K, wPrior=None,feature0_infer=None):
        """
        :param pose, the initial pose
            (extrinsic of the target frame w.r.t. the referenc frame)
        :param x0, the template features
        :param x1, the image features
        :param invD0, the template inverse depth
        :param invD1, the image inverse depth
        :param K, the intrinsic parameters, [fx, fy, cx, cy]
        :param wPrior (optional), provide an initial weight as input to the convolutional m-estimator
        """
        B, C, H, W = x0.shape
        px, py = geometry.generate_xy_grid(B,H,W,K)

        # 13
        if self.timers: self.timers.tic('pre-compute Jacobians')
        J_F_p = self.precompute_Jacobian(invD0, x0, px, py, K)
        if self.timers: self.timers.toc('pre-compute Jacobians')

        # 12
        if self.timers: self.timers.tic('compute warping residuals')
        residuals, occ = compute_warped_residual(pose, invD0, invD1, \
            x0, x1, px, py, K)
        
        residuals = residuals-feature0_infer*residuals
        if self.timers: self.timers.toc('compute warping residuals')

        # 16
        if self.timers: self.timers.tic('robust estimator')
        weights = self.mEstimator(residuals, x0, x1, wPrior)
        wJ = weights.view(B,-1,1) * J_F_p
        if self.timers: self.timers.toc('robust estimator')

        if self.timers: self.timers.tic('pre-compute JtWJ')
        JtWJ = torch.bmm(torch.transpose(J_F_p, 1, 2) , wJ)
        if self.timers: self.timers.toc('pre-compute JtWJ')

        # 循环优化
        for idx in range(self.max_iterations):
            if self.timers: self.timers.tic('solve x=A^{-1}b')
            # print('循环优化',residuals.shape)
            pose = self.directSolver(JtWJ,
                torch.transpose(J_F_p,1,2), weights, residuals,
                pose, invD0, invD1, x0, x1, K)
            if self.timers: self.timers.toc('solve x=A^{-1}b')
    
            if self.timers: self.timers.tic('compute warping residuals')
            residuals, occ = compute_warped_residual(pose, invD0, invD1, \
                x0, x1, px, py, K)
            residuals = residuals-feature0_infer*residuals
            if self.timers: self.timers.toc('compute warping residuals')


        return pose, weights 


    def inverse_pose(self,R, t):
        # 计算旋转矩阵的逆（正交矩阵的逆等于其转置）
        R_inv = R.transpose(-2, -1)
        # 计算逆平移向量
        t_inv = -torch.matmul(R_inv, t.unsqueeze(-1)).squeeze(-1)
        # 构建 4x4 齐次逆变换矩阵
        batch_size = R.shape[0]
        T_inv = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        T_inv[:, :3, :3] = R_inv
        T_inv[:, :3, 3] = t_inv
        return [R_inv,t_inv]
    
    def precompute_Jacobian(self, invD, x, px, py, K):
        """ Pre-compute the image Jacobian on the reference frame
        refer to equation (13) in the paper
        
        :param invD, template depth
        :param x, template feature
        :param px, normalized image coordinate in cols (x)
        :param py, normalized image coordinate in rows (y)
        :param K, the intrinsic parameters, [fx, fy, cx, cy]

        ------------
        :return precomputed image Jacobian on template
        """
        """在参考系上预先计算图像雅可比矩阵
        参考论文中的方程式（13）
                
        ：参数invD，模板深度
        ：参数x，模板特征
        ：param-px，cols（x）中的归一化图像坐标
        ：参数py，行（y）中的归一化图像坐标
        ：参数K，固有参数[fx，fy，cx，cy]

        ------------
        ：返回模板上预先计算的图像雅可比矩阵
        """
        Jf_x, Jf_y = feature_gradient(x)
        Jx_p, Jy_p = compute_jacobian_warping(invD, K, px, py)
        J_F_p = compute_jacobian_dIdp(Jf_x, Jf_y, Jx_p, Jy_p)
        return J_F_p


class ImagePyramids(nn.Module):
    """ Construct the pyramids in the image / depth space
    """
    """在图像/深度空间中构建金字塔
    """
    def __init__(self, scales, pool='avg'):
        super(ImagePyramids, self).__init__()
        if pool == 'avg':
            self.multiscales = [nn.AvgPool2d(1<<i, 1<<i) for i in scales]
        elif pool == 'max':
            self.multiscales = [nn.MaxPool2d(1<<i, 1<<i) for i in scales]
        else:
            raise NotImplementedError()

    def forward(self, x):
        x_out = [f(x) for f in self.multiscales]
        return x_out

class FeaturePyramid(nn.Module):
    """ 
    The proposed feature-encoder (A).
    It also supports to extract features using one-view only.
    """
    """ 
    所提出的特征编码器（A）。
    它还支持仅使用一个视图提取特征。
    """
    def __init__(self, D):
        super(FeaturePyramid, self).__init__()
        self.net0 = nn.Sequential(
            conv(True, D,  16, 3), 
            conv(True, 16, 32, 3, dilation=2),
            conv(True, 32, 32, 3, dilation=2))
        self.net1 = nn.Sequential(
            conv(True, 32, 32, 3),
            conv(True, 32, 64, 3, dilation=2),
            conv(True, 64, 64, 3, dilation=2))
        self.net2 = nn.Sequential(
            conv(True, 64, 64, 3),
            conv(True, 64, 96, 3, dilation=2),
            conv(True, 96, 96, 3, dilation=2))
        self.net3 = nn.Sequential(
            conv(True, 96, 96, 3),
            conv(True, 96, 128, 3, dilation=2),
            conv(True, 128,128, 3, dilation=2))

        initialize_weights(self.net0)
        initialize_weights(self.net1)
        initialize_weights(self.net2)
        initialize_weights(self.net3)
        self.downsample = torch.nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x0 = self.net0(x)
        x0s= self.downsample(x0)
        x1 = self.net1(x0s)
        x1s= self.downsample(x1)
        x2 = self.net2(x1s)
        x2s= self.downsample(x2)
        x3 = self.net3(x2s)
        return x0, x1, x2, x3

class FeaturePyramid_new(nn.Module):
    """ 
    The proposed feature-encoder (A).
    It also supports to extract features using one-view only.
    """
    """ 
    所提出的特征编码器（A）。
    它还支持仅使用一个视图提取特征。
    """
    def __init__(self, D):
        super(FeaturePyramid_new, self).__init__()
        self.net0 = nn.Sequential(
            conv(True, D,  16, 3), 
            conv(True, 16, 32, 3, dilation=2),
            conv(True, 32, 32, 3, dilation=2))
        self.net1 = nn.Sequential(
            conv(True, 32, 32, 3),
            conv(True, 32, 64, 3, dilation=2),
            conv(True, 64, 64, 3, dilation=2))
        self.net2 = nn.Sequential(
            conv(True, 64, 64, 3),
            conv(True, 64, 96, 3, dilation=2),
            conv(True, 96, 96, 3, dilation=2))
        self.net3 = nn.Sequential(
            conv(True, 96, 96, 3),
            conv(True, 96, 128, 3, dilation=2),
            conv(True, 128,128, 3, dilation=2))

        initialize_weights(self.net0)
        initialize_weights(self.net1)
        initialize_weights(self.net2)
        initialize_weights(self.net3)
        self.downsample = torch.nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x0 = self.net0(x)
        x0s= self.downsample(x0)

        x1 = self.net1(x0s)
        x1s= self.downsample(x1)

        x2 = self.net2(x1s)
        x2s= self.downsample(x2)

        x3 = self.net3(x2s)
        x = [x0, x1, x2, x3]


        x_before =[]
        for i in range(len(x)):
            # new_x.append(self.__Nto1(x[i]*feature_list[i]))
            x_before.append(self.__Nto1(x[i]))

        return x_before
    
    def __Nto1(self, x):
        """ Take the average of multi-dimension feature into one dimensional,
            which boostrap the optimization speed
        """
        C = x.shape[1]
        return x.sum(dim=1, keepdim=True) / C
        # return x0, x1, x2, x3


class weightNet(nn.Module):
    def __init__(self):
        super(weightNet, self).__init__()
        self.net = nn.Sequential(
                    nn.Conv2d(5, 64, kernel_size=3, stride=1, bias=True,padding=1),nn.ReLU(),
                    nn.Conv2d(64,32,3,1,1),nn.ReLU(),
                    nn.Conv2d(32,32,3,1,1),nn.ReLU(),
                    nn.Conv2d(32,32,3,1,1),nn.ReLU(),
                    nn.Conv2d(32,8,3,1,1),nn.ReLU(),
                    nn.Conv2d(8,1,3,1,1),nn.ReLU())
        initialize_weights(self.net)
    def forward(self,x):
        return self.net(x)
    
    
class DeepRobustEstimator(nn.Module):
    """ The M-estimator 

    When use estimator_type = 'MultiScale2w', it is the proposed convolutional M-estimator
    """

    """M估计量
    当使用estimator_type='MultiScale2w'时，它是所提出的卷积M-估计器
    """

    def __init__(self):
        super(DeepRobustEstimator, self).__init__()
        self.D = 4
        self.net = nn.Sequential(
            conv(True, self.D, 16, 3, dilation=1),
            conv(True, 16, 32, 3, dilation=2),
            conv(True, 32, 64, 3, dilation=4),
            conv(True, 64, 1,  3, dilation=1),
            nn.Sigmoid() )
        initialize_weights(self.net)


    def forward(self, residual, x0, x1, ws=None):
        """
        :param residual, the residual map
        :param x0, the feature map of the template
        :param x1, the feature map of the image
        :param ws, the initial weighted residual
        """

        B, C, H, W = residual.shape
        wl = func.interpolate(ws, (H,W), mode='bilinear', align_corners=True)
        context = torch.cat((residual.abs(), x0, x1, wl), dim=1)
        w = self.net(context)

        return w

    def __weight_Huber(self, x, alpha = 0.02):
        """ weight function of Huber loss:
        refer to P. 24 w(x) at
        https://members.loria.fr/moberger/Enseignement/Master2/Documents/ZhangIVC-97-01.pdf

        Note this current implementation is not differentiable.
        """
        abs_x = torch.abs(x)
        linear_mask = abs_x > alpha
        w = torch.ones(x.shape).type_as(x)

        if linear_mask.sum().item() > 0: 
            w[linear_mask] = alpha / abs_x[linear_mask]
        return w

    def __constant_weight(self, x):
        """ mimic the standard least-square when weighting function is constant
        """
        return torch.ones(x.shape).type_as(x)

class DirectSolverNet(nn.Module):

    # the enum types for direct solver
    SOLVER_NO_DAMPING       = 0
    SOLVER_RESIDUAL_VOLUME  = 1

    def __init__(self, solver_type, samples=10):
        super(DirectSolverNet, self).__init__()

        if solver_type == 'Direct-Nodamping':
            self.net = None
            self.type = self.SOLVER_NO_DAMPING
        elif solver_type == 'Direct-ResVol':
            # flattened JtJ and JtR (number of samples, currently fixed at 10)
            samples=16
            self.samples = samples
            self.net = deep_damping_regressor(D=6*6+6*samples)
            self.type = self.SOLVER_RESIDUAL_VOLUME
            initialize_weights(self.net)
        else: 
            raise NotImplementedError()

    def forward(self, JtJ, Jt, weights, R, pose0, invD0, invD1, x0, x1, K):
        """
        :param JtJ, the approximated Hessian JtJ
        :param Jt, the trasposed Jacobian
        :param weights, the weight matrix
        :param R, the residual
        :param pose0, the initial estimated pose
        :param invD0, the template inverse depth map
        :param invD1, the image inverse depth map
        :param x0, the template feature map
        :param x1, the image feature map
        :param K, the intrinsic parameters

        -----------
        :return updated pose
        """
        """
        ：参数JtJ，近似的Hessian JtJ
        ：参数Jt，变换雅可比矩阵
        ：参数权重，权重矩阵
        ：参数R，残差
        ：param pose0，初始估计姿态
        ：param invD0，模板逆深度图
        ：param invD1，图像逆深度图
        ：参数x0，模板特征图
        ：param x1，图像特征图
        ：param K，内部参数

        -----------
        ：返回更新的姿势
        """
        B = JtJ.shape[0]

        wR = (weights * R).view(B, -1, 1)
        JtR = torch.bmm(Jt, wR)

        if self.type == self.SOLVER_NO_DAMPING:
            # Add a small diagonal damping. Without it, the training becomes quite unstable
            # Do not see a clear difference by removing the damping in inference though
            #添加一个小的对角线阻尼。没有它，训练变得相当不稳定
            #尽管消除了推理中的阻尼，但看不到明显的区别
            diag_mask = torch.eye(6).view(1,6,6).type_as(JtJ)
            diagJtJ = diag_mask * JtJ
            traceJtJ = torch.sum(diagJtJ, (2,1))
            epsilon = (traceJtJ * 1e-6).view(B,1,1) * diag_mask
            Hessian = JtJ + epsilon
            pose_update = inverse_update_pose(Hessian, JtR, pose0)

        elif self.type == self.SOLVER_RESIDUAL_VOLUME:
            Hessian = self.__regularize_residual_volume(JtJ, Jt, JtR, weights,
                pose0, invD0, invD1, x0, x1, K, sample_range=self.samples)
            pose_update = inverse_update_pose(Hessian, JtR, pose0)
        else:
            raise NotImplementedError()

        return pose_update

    def __regularize_residual_volume(self, JtJ, Jt, JtR, weights, pose,
        invD0, invD1, x0, x1, K, sample_range):
        """ regularize the approximate with residual volume

        :param JtJ, the approximated Hessian JtJ
        :param Jt, the trasposed Jacobian
        :param JtR, the Right-hand size residual
        :param weights, the weight matrix
        :param pose, the initial estimated pose
        :param invD0, the template inverse depth map
        :param invD1, the image inverse depth map
        :param K, the intrinsic parameters
        :param x0, the template feature map
        :param x1, the image feature map
        :param sample_range, the numerb of samples

        ---------------
        :return the damped Hessian matrix
        """
        """用剩余体积对近似值进行正则化
        ：参数JtJ，近似的Hessian JtJ
        ：参数Jt，变换雅可比矩阵
        ：参数JtR，右手尺寸残差
        ：参数权重，权重矩阵
        ：param姿势，初始估计姿势
        ：param invD0，模板逆深度图
        ：param invD1，图像逆深度图
        ：param K，内部参数
        ：参数x0，模板特征图
        ：param x1，图像特征图
        ：param sample_range，样本数量
        ---------------
        ：返回阻尼Hessian矩阵
        """
        # the following current support only single scale
        JtR_volumes = []

        B, C, H, W = x0.shape
        px, py = geometry.generate_xy_grid(B, H, W, K)

        diag_mask = torch.eye(6).view(1,6,6).type_as(JtJ)
        diagJtJ = diag_mask * JtJ
        traceJtJ = torch.sum(diagJtJ, (2,1))
        epsilon = (traceJtJ * 1e-6).view(B,1,1) * diag_mask
        n = sample_range
        # n=22

        lambdas = torch.logspace(-5, 10, n).type_as(JtJ)
        # print('lambdas',lambdas)

        # 在对数空间生成n个λ值（lambdas），覆盖1e-5到1e5范围，用于探索不同强度的正则化效果。
        # 对每个λ值进行循环
        for s in range(n):
            # the epsilon is to prevent the matrix to be too ill-conditioned
            D = lambdas[s] * diagJtJ + epsilon
            Hessian = JtJ + D
            pose_s = inverse_update_pose(Hessian, JtR, pose)

            res_s,_= compute_warped_residual(pose_s, invD0, invD1, x0, x1, px, py, K)
            JtR_s = torch.bmm(Jt, (weights * res_s).view(B,-1,1))
            JtR_volumes.append(JtR_s)

        JtR_flat = torch.cat(tuple(JtR_volumes), dim=2).view(B,-1)
        JtJ_flat = JtJ.view(B,-1)
        damp_est = self.net(torch.cat((JtR_flat, JtJ_flat), dim=1))

        # scarlet
        # print('damp_est',damp_est[0])
        R = diag_mask * damp_est.view(B,6,1) + epsilon # also lift-up

        return JtJ + R

def deep_damping_regressor(D):
    """ Output a damping vector at each dimension
    """
    net = nn.Sequential(
        fcLayer(in_planes=D,   out_planes=128, bias=True),
        fcLayer(in_planes=128, out_planes=256, bias=True),
        fcLayer(in_planes=256, out_planes=6, bias=True)
    ) # the last ReLU makes sure every predicted value is positive
    return net

def feature_gradient(img, normalize_gradient=True):
    """ Calculate the gradient on the feature space using Sobel operator
    :param the input image 
    -----------
    :return the gradient of the image in x, y direction
    """
    B, C, H, W = img.shape
    # to filter the image equally in each channel
    wx = torch.FloatTensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]).view(1,1,3,3).type_as(img)
    wy = torch.FloatTensor([[-1,-2,-1],[ 0, 0, 0],[ 1, 2, 1]]).view(1,1,3,3).type_as(img)

    img_reshaped = img.view(-1, 1, H, W)
    img_pad = func.pad(img_reshaped, (1,1,1,1), mode='replicate')
    img_dx = func.conv2d(img_pad, wx, stride=1, padding=0)
    img_dy = func.conv2d(img_pad, wy, stride=1, padding=0)

    if normalize_gradient:
        mag = torch.sqrt((img_dx ** 2) + (img_dy ** 2)+ 1e-8)
        img_dx = img_dx / mag 
        img_dy = img_dy / mag

    return img_dx.view(B,C,H,W), img_dy.view(B,C,H,W)

def compute_jacobian_dIdp(Jf_x, Jf_y, Jx_p, Jy_p):
    """ chained gradient of image w.r.t. the pose
    :param the Jacobian of the feature map in x direction
    :param the Jacobian of the feature map in y direction
    :param the Jacobian of the x map to manifold p
    :param the Jacobian of the y map to manifold p
    ------------
    :return the image jacobian in x, y, direction, Bx2x6 each
    """
    """
        图像相对于姿态的链式梯度
    ：param x方向上特征图的雅可比矩阵
    ：param y方向上特征图的雅可比矩阵
    ：param x映射到流形p的雅可比矩阵
    ：param y映射到流形p的雅可比矩阵
    ------------
    ：返回x、y方向上的图像雅可比矩阵，每个方向为Bx2x6
    """
    B, C, H, W = Jf_x.shape

    # precompute J_F_p, JtWJ
    Jf_p = Jf_x.view(B,C,-1,1) * Jx_p.view(B,1,-1,6) + \
        Jf_y.view(B,C,-1,1) * Jy_p.view(B,1,-1,6)
    
    return Jf_p.view(B,-1,6)

def compute_jacobian_warping(p_invdepth, K, px, py):
    """ Compute the Jacobian matrix of the warped (x,y) w.r.t. the inverse depth
    (linearized at origin)
    :param p_invdepth the input inverse depth
    :param the intrinsic calibration
    :param the pixel x map
    :param the pixel y map
     ------------
    :return the warping jacobian in x, y direction
    """
    """ 计算扭曲的（x，y）相对于逆深度的雅可比矩阵
    （在原点处线性化）
    ：param p_invdest输入逆深度
    ：param固有校准
    ：param像素x映射
    ：param像素y映射
    ------------
    ：返回x、y方向上的扭曲雅可比矩阵
    """

    B, C, H, W = p_invdepth.size()
    assert(C == 1)

    x = px.view(B, -1, 1)
    y = py.view(B, -1, 1)
    invd = p_invdepth.view(B, -1, 1)

    xy = x * y
    O = torch.zeros((B, H*W, 1)).type_as(p_invdepth)

    # This is cascaded Jacobian functions of the warping function
    # Refer to the supplementary materials for math documentation
    dx_dp = torch.cat((-xy,     1+x**2, -y, invd, O, -invd*x), dim=2)
    dy_dp = torch.cat((-1-y**2, xy,     x, O, invd, -invd*y), dim=2)

    # fx=torch.tensor( K[0])
    # fy= torch.tensor(K[1])
    # cx= torch.tensor(K[2])
    # cy= torch.tensor(K[3])
    # fx=K[0]
    # fy=K[1]
    # cx=K[2]
    # cy=K[3]
    fx, fy, cx, cy = K.split(1,dim=1)

    return dx_dp*fx.view(B,1,1), dy_dp*fy.view(B,1,1)

def compute_warped_residual(pose, invD0, invD1, x0, x1, px, py, K, obj_mask=None):
    """ Compute the residual error of warped target image w.r.t. the reference feature map.
    refer to equation (12) in the paper

    :param the forward warping pose from the reference camera to the target frame.
        Note that warping from the target frame to the reference frame is the inverse of this operation.
    :param the reference inverse depth
    :param the target inverse depth
    :param the reference feature image
    :param the target feature image
    :param the pixel x map
    :param the pixel y map
    :param the intrinsic calibration
    -----------
    :return the residual (of reference image), and occlusion information
    """
    u_warped, v_warped, inv_z_warped = geometry.batch_warp_inverse_depth(
        px, py, invD0, pose, K)
    x1_1to0 = geometry.warp_features(x1, u_warped, v_warped)
    occ = geometry.check_occ(inv_z_warped, invD1, u_warped, v_warped)

    residuals = x1_1to0 - x0 # equation (12)

    B, C, H, W = x0.shape
    if obj_mask is not None:
        # determine whether the object is in-view
        occ = occ & (obj_mask.view(B,1,H,W) < 1)
    residuals[occ.expand(B,C,H,W)] = 1e-3
    return residuals, occ

def inverse_update_pose(H, Rhs, pose):
    """ Ues left-multiplication for the pose update 
    in the inverse compositional form
    refer to equation (10) in the paper 

    :param the (approximated) Hessian matrix
    :param Right-hand side vector
    :param the initial pose (forward transform inverse of xi)
    ---------
    :return the forward updated pose (inverse of xi)
    """
    inv_H = invH(H)
    xi = torch.bmm(inv_H, Rhs)
    # simplifed inverse compositional for SE3
    d_R = geometry.batch_twist2Mat(-xi[:, :3].view(-1,3))
    d_t = -torch.bmm(d_R, xi[:, 3:])

    R, t = pose
    pose = geometry.batch_Rt_compose(R, t, d_R, d_t) 
    return pose

# def invH(H):
#     """ Generate (H+damp)^{-1}, with predicted damping values
#     :param approximate Hessian matrix JtWJ
#     -----------
#     :return the inverse of Hessian
#     """
#     # GPU is much slower for matrix inverse when the size is small (compare to CPU)
#     # works (50x faster) than inversing the dense matrix in GPU
#     if H.is_cuda:
#         # invH = bpinv((H).cpu()).cuda()
#         # invH = torch.inverse(H)
#         invH = torch.inverse(H.cpu()).cuda()
#         # invH = torch.inverse(H.cpu()).cuda()
#     else:
#         invH = torch.inverse(H)
#     return invH

def invH( H, reg_param=1e-6):
    # 获取矩阵的维度信息
    batch_size, n, _ = H.shape
    # 创建与H同设备的单位矩阵
    I = torch.eye(n, device=H.device).expand(batch_size, n, n)
    # 对矩阵进行正则化处理
    H_reg = H + reg_param * I
    if H.is_cuda:
        # invH = bpinv((H).cpu()).cuda()
        # invH = torch.inverse(H)
        # invH = torch.inverse(H.cpu()).cuda()
        # 尝试求逆
        try:
            invH = torch.inverse(H_reg.cpu()).cuda()
        except:
            # 如果求逆失败，使用伪逆
            invH = torch.linalg.pinv(H_reg.cpu()).cuda()
    else:
            # 尝试求逆
        try:
            invH = torch.inverse(H_reg)
        except:
            # 如果求逆失败，使用伪逆
            invH = torch.linalg.pinv(H_reg)

    return invH


class DampingNet(nn.Module):
    def __init__(self, conf, input_size,const_size=6,num_consts=6):
        super().__init__()
        self.num_consts = num_consts
        self.const_size = const_size

        self.input_sizes = input_size

        # 定义 num_consts 组可选的 const 参数，每组尺寸为 [const_size]
        self.consts = nn.Parameter(torch.randn(num_consts, const_size))

        # # 定义一个简单的神经网络 lambda_net，用于生成 lambda 权重
        # # 定义 lambda_net，包含 2 层卷积层和线性层
        # # 定义 lambda_net，包含卷积层、池化层和线性层
        # # print(input_size)
        self.lambda_net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # 第 1 层池化，下采样 2 倍

            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 第 1 层卷积
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 第 1 层池化，下采样 2 倍
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 第 2 层卷积
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 第 2 层池化，下采样 2 倍
            nn.Flatten(),  # 展平为线性层的输入
            nn.Linear(self._get_flattened_size(self.input_sizes ), 64),  # 线性层 1
            nn.ReLU(),
            nn.Linear(64, num_consts),  # 线性层 2，输出 num_consts 个权重
            nn.Softmax(dim=-1)  # 对权重进行 softmax，确保和为 1
        )
    def _get_flattened_size(self, input_size):
        """
        根据输入尺寸计算展平后的特征图大小。
        input_size: 输入尺寸 (height, width)
        """
        height, width = input_size[0],input_size[1]
        # print(height,width)
        height = height // 2  # 池化
        width = width // 2
        # 第 1 层卷积 + 第 1 层池化
        height = (height + 2 * 1 - 3) // 1 + 1  # 卷积
        width = (width + 2 * 1 - 3) // 1 + 1
        height = height // 2  # 池化
        width = width // 2

        # 第 2 层卷积 + 第 2 层池化
        height = (height + 2 * 1 - 3) // 1 + 1  # 卷积
        width = (width + 2 * 1 - 3) // 1 + 1
        height = height // 2  # 池化
        width = width // 2
        # print(height,width)

        # 最终展平大小
        return height * width * 16
    
    def forward(self,residuals):

        """
        residuals: 残差，形状为 (batch, 30, 40)等三种
        """
        # print(residuals.shape)
        # from models.LeastSquareTracking import plot_one_tensor
        # plot_one_tensor(residuals[0],'residuals[0]',2)
        # batch_size,_, height, width = residuals.shape

        # 使用 lambda_net 根据残差生成 lambda 权重
        # print('residuals.shape',residuals.shape)
        # print('self.input_sizes',self.input_sizes)
        # print('==========')
        lambdas_w = self.lambda_net(residuals)  # 形状为 (batch, num_consts, 256, 320)
        # print(lambdas_w.shape)

        # 对 consts 加权，生成每个样本的最终参数
        # consts: (num_consts, const_size) -> (1, num_consts, const_size)
        

        
        # weighted_consts: (batch, const_size)
        lambdas_w = lambdas_w.unsqueeze(0)  # 扩展 batch 维度
        # print(lambdas_w.shape)
        # print(self.consts.shape)

        weighted_consts = torch.matmul(lambdas_w, self.consts)
        # print(weighted_consts.shape)
        # print('==========')
        min_, max_ = -6,-5
        lambda_ = 10.**(min_ + weighted_consts.sigmoid()*(max_ - min_))

        return lambda_
    
class LearnedSolverNet(nn.Module):

    def __init__(self,input_size):
        super().__init__()
        default_conf = dict(
            damping=dict(
                type='constant',
                log_range=[-6, 5],
            ),
            feature_dim=None,

            # deprecated entries
            lambda_=0.,
            learned_damping=True,
        )
        conf = default_conf

        self.dampingnet = DampingNet(conf["damping"],input_size=input_size)
        
        assert conf['learned_damping']
        
    def forward(self, JtJ, Jt, weights, R, pose0, invD0, invD1, x0, x1, K):
        # print('forward',R.shape)
        B = JtJ.shape[0]

        wR = (weights * R).view(B, -1, 1)
        JtR = torch.bmm(Jt, wR)

        Hessian = self.__regularize_residual_volume_new(JtJ, Jt, JtR, weights,
            pose0, invD0, invD1, x0, x1, K,R)
        pose_update = inverse_update_pose(Hessian, JtR, pose0)

        return pose_update

    def __regularize_residual_volume_new(self, JtJ, Jt, JtR, weights, pose,
        invD0, invD1, x0, x1, K,R):

        # the following current support only single scale
        JtR_volumes = []

        B, C, H, W = x0.shape
        # px, py = geometry.generate_xy_grid(B, H, W, K)
        # print('__regularize_residual_volume_new',R.shape)
        lambda_ = self.dampingnet(residuals=R)

        
        diag_mask = torch.eye(6).view(1,6,6).type_as(JtJ)
        diagJtJ = diag_mask * JtJ
        traceJtJ = torch.sum(diagJtJ, (2,1))
        epsilon = (traceJtJ * 1e-6).view(B,1,1) * diag_mask

        R = diag_mask * lambda_.unsqueeze(2)+ epsilon # also lift-up

        return JtJ + R.squeeze(0)

    # def __regularize_residual_volume_new(self, JtJ, Jt, JtR, weights, pose,
    #         invD0, invD1, x0, x1, K,R):

    #         # the following current support only single scale
    #         JtR_volumes = []

    #         B, C, H, W = x0.shape
    #         # px, py = geometry.generate_xy_grid(B, H, W, K)
    #         # print('__regularize_residual_volume_new',R.shape)
    #         lambda_ = self.dampingnet(residuals=R)

            
    #         # 替换diag_mask的实现方式：使用diag_embed创建对角矩阵
    #         diag_mask = torch.diag_embed(torch.ones(6, device=JtJ.device)).view(1,6,6).type_as(JtJ)
            
    #         # 另一种实现方式：直接构造对角矩阵
    #         # diag_mask = torch.zeros(6, 6, device=JtJ.device).type_as(JtJ)
    #         # diag_mask[range(6), range(6)] = 1.0
    #         # diag_mask = diag_mask.view(1,6,6)
            
    #         diagJtJ = diag_mask * JtJ
    #         traceJtJ = torch.sum(diagJtJ, (2,1))
    #         epsilon = (traceJtJ * 1e-6).view(B,1,1) * diag_mask

    #         R = diag_mask * lambda_.unsqueeze(2)+ epsilon # also lift-up

    #         return JtJ + R.squeeze(0)


    # def __regularize_residual_volume_new(self, JtJ, Jt, JtR, weights, pose,
    #                                     invD0, invD1, x0, x1, K, R):
    #     JtR_volumes = []
    #     B, C, H, W = x0.shape
    #     px, py = geometry.generate_xy_grid(B, H, W, K)
        
    #     # 1. 强制lambda_为固定形状(B,6)，避免dampingnet引入控制流
    #     # 若dampingnet内部有条件分支，建议临时替换为固定值测试：
    #     # lambda_ = torch.ones(B, 6, device=JtJ.device, dtype=JtJ.dtype) * 0.1  # 测试用固定值
    #     lambda_ = self.dampingnet(residuals=R)
    #     # 确保lambda_形状绝对固定，无动态维度
    #     assert lambda_.shape == (B, 6), f"lambda_形状异常: {lambda_.shape}"

    #     # 2. 预生成对角掩码（直接作为常量，避免动态生成）
    #     # 提前在模型初始化时定义diag_mask为缓冲区（buffer），此处直接使用
    #     # 模型初始化时添加：self.register_buffer('diag_mask', torch.eye(6).reshape(1,6,6))
    #     diag_mask = self.diag_mask  # 形状(1,6,6)，与JtJ同设备/类型

    #     # 3. 简化diagJtJ计算（仅基础乘法）
    #     diagJtJ = diag_mask * JtJ  # 形状(B,6,6)

    #     # 4. 简化迹计算（显式展开维度，避免多维sum的歧义）
    #     traceJtJ = (diagJtJ[:,0,0] + diagJtJ[:,1,1] + diagJtJ[:,2,2] + 
    #                 diagJtJ[:,3,3] + diagJtJ[:,4,4] + diagJtJ[:,5,5])  # 形状(B,)

    #     # 5. 简化epsilon计算（避免广播歧义）
    #     epsilon_scalar = traceJtJ * 1e-6  # 形状(B,)
    #     # 显式扩展为(B,6,6)，与diag_mask对应位置相乘
    #     epsilon = diag_mask * epsilon_scalar.view(B, 1, 1)  # 形状(B,6,6)

    #     # 6. 简化R的计算（完全消除unsqueeze/reshape，用广播规则直接匹配）
    #     # lambda_形状(B,6) → 与diag_mask(B,6,6)相乘时，自动广播为(B,6,1)
    #     term_diag_lambda = diag_mask * lambda_.unsqueeze(2)  # 形状(B,6,6)
    #     R = term_diag_lambda + epsilon  # 形状(B,6,6)，Add节点输出

    #     # 7. 避免squeeze，直接用索引或reshape固定输出形状
    #     # 若JtJ是(B,6,6)，则R无需squeeze；若JtJ是(6,6)，则用R[0]替代squeeze(0)
    #     # 根据实际JtJ形状选择：
    #     # return JtJ + R  # 若JtJ是(B,6,6)
    #     return JtJ + R[0]  # 若JtJ是(6,6)，用索引替代squeeze，避免Squeeze节点

