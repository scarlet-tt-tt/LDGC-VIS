import torch
import torch.nn as nn
import numpy as np

from LEM_SFM.models.submodules import convLayer as conv
from LEM_SFM.models.submodules import color_normalize

from LEM_SFM.models.algorithms import TrustRegionBase as TrustRegion
from LEM_SFM.models.algorithms import TrustRegionBase_i
from LEM_SFM.models.algorithms import ImagePyramids, DirectSolverNet, FeaturePyramid, DeepRobustEstimator,LearnedSolverNet

is_plot = True

def plot_one_tensor(tensor,name,plot_type):
    import matplotlib.pyplot as plt
    if plot_type ==1:
        # [1,3,240,320]
        tensor_2d = tensor.squeeze(0).cpu().detach().numpy()
        plt.imshow(tensor_2d.transpose(1,2,0), cmap='viridis')
        plt.axis('off') 
        plt.savefig(name)
    if plot_type ==2:
        # [1,1,240,320]
        tensor_2d = tensor.squeeze(0).squeeze(0).cpu().detach().numpy()
        plt.imshow(tensor_2d, cmap='viridis')
        plt.axis('off') 
        plt.colorbar()
        plt.savefig(name)
        plt.clf()

class LeastSquareTracking(nn.Module):

    # all enum types
    NONE                = -1
    RGB                 = 0

    CONV_RGBD           = 1
    CONV_RGBD2          = 2

    def __init__(self, encoder_name,
        max_iter_per_pyr,
        mEst_type,
        solver_type,
        tr_samples = 10,
        no_weight_sharing = False,
        timers = None):
        """
        :param the backbone network used for regression.
        :param the maximum number of iterations at each pyramid levels
        :param the type of weighting functions.
        :param the type of solver. 
        :param number of samples in trust-region solver
        :param True if we do not want to share weight at different pyramid levels
        :param (optional) time to benchmark time consumed at each step
        """

        """
        encoder_name：param用于回归的主干网络。
        max_iter_per_pyr：param每个金字塔级别的最大迭代次数
        mEst_type：param表示加权函数的类型。
        solver_type：param求解器的类型。
        tr_samples = 10：信任区域求解器中的样本数参数
        no_weight_sharing = False：param如果我们不想在不同的金字塔级别共享权重，则为True
        timers = None：param（可选）每个步骤消耗的基准时间
        """

        super(LeastSquareTracking, self).__init__()

        self.construct_image_pyramids = ImagePyramids([0,1,2,3], pool='avg')
        self.construct_depth_pyramids = ImagePyramids([0,1,2,3], pool='max')

        self.timers = timers

        """ =============================================================== """
        """             Initialize the Deep Feature Extractor               """
        """ =============================================================== """

        if encoder_name == 'RGB':
            print('The network will use raw image as measurements.')
            self.encoder = None
            self.encoder_type = self.RGB
            context_dim = 1
        elif encoder_name == 'ConvRGBD':
            print('Use a network with RGB-D information \
            to extract the features')
            context_dim = 4
            self.encoder = FeaturePyramid(D=context_dim)
            self.encoder_type = self.CONV_RGBD
        elif encoder_name == 'ConvRGBD2':
            print('Use two stream network with two frame input')
            context_dim = 8
            self.encoder = FeaturePyramid(D=context_dim)
            self.encoder_type = self.CONV_RGBD2
        else:
            raise NotImplementedError()

        """ =============================================================== """
        """             Initialize the Robust Estimator                     """
        """ =============================================================== """

        if no_weight_sharing:
            self.mEst_func0 = DeepRobustEstimator(mEst_type)
            self.mEst_func1 = DeepRobustEstimator(mEst_type)
            self.mEst_func2 = DeepRobustEstimator(mEst_type)
            self.mEst_func3 = DeepRobustEstimator(mEst_type)
            mEst_funcs = [self.mEst_func0, self.mEst_func1, self.mEst_func2,
            self.mEst_func3]
        else:
            self.mEst_func = DeepRobustEstimator(mEst_type)
            mEst_funcs = [self.mEst_func, self.mEst_func, self.mEst_func,
            self.mEst_func]

        """ =============================================================== """
        """             Initialize the Trust-Region Damping                 """
        """ =============================================================== """

        # if no_weight_sharing:
        #     # for residual volume, the input K is not assigned correctly
        #     self.solver_func0 = DirectSolverNet(solver_type, samples=tr_samples)
        #     self.solver_func1 = DirectSolverNet(solver_type, samples=tr_samples)
        #     self.solver_func2 = DirectSolverNet(solver_type, samples=tr_samples)
        #     self.solver_func3 = DirectSolverNet(solver_type, samples=tr_samples)
        #     solver_funcs = [self.solver_func0, self.solver_func1,
        #     self.solver_func2, self.solver_func3]
        # else:
        #     self.solver_func = DirectSolverNet(solver_type, samples=tr_samples)
        #     solver_funcs = [self.solver_func, self.solver_func,
        #         self.solver_func, self.solver_func]
        
        self.solver_func3 = LearnedSolverNet((30,40))
        self.solver_func2 = LearnedSolverNet((60,80))
        self.solver_func1 = LearnedSolverNet((120,160))
        self.solver_func0 = LearnedSolverNet((240,320))
        solver_funcs = [self.solver_func0, self.solver_func1,
            self.solver_func2, self.solver_func3]
        """ =============================================================== """
        """             Initialize the Trust-Region Method                  """
        """ =============================================================== """

        self.tr_update0 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[0],
            solver_func = solver_funcs[0],
            timers      = timers)
        self.tr_update1 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[1],
            solver_func = solver_funcs[1],
            timers      = timers)
        self.tr_update2 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[2],
            solver_func = solver_funcs[2],
            timers      = timers)
        self.tr_update3 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[3],
            solver_func = solver_funcs[3],
            timers      = timers)
        self.downsample = torch.nn.AvgPool2d(kernel_size=2)


# img0, img1, inv_depth0_infer, inv_depth1_infer ,feature0_infer,feature1_infer, K
    def forward(self, img0, img1, inv_depth0_infer, inv_depth1_infer,feature0_infer,feature1_infer, K, init_only=False):

        # img0, img1, inv_depth0_infer, inv_depth1_infer , K
        """
        :input
        :param the reference image
        :param the target image
        :param the inverse depth of the reference image
        :param the inverse depth of the target image
        :param the pin-hole camera instrinsic (in vector) [fx, fy, cx, cy] 
        :param the initial pose [Rotation, translation]
        --------
        :return 
        :param estimated transform 
        # """
        # if is_plot ==True:
        #     print(img0.shape)
        #     plot_one_tensor(img0[0,:,:,:],name='img0',type=1)
        #     print(depth0.shape)
        #     plot_one_tensor(depth0[0,:,:,:],name='depth0',type=2)
        # print(img0.shape)
        # plot_one_tensor(img0[0,:,:,:],name='img0',type=1)
        # print(depth0.shape)
        # plot_one_tensor(depth0[0,:,:,:],name='depth0',type=2)
        if self.timers: self.timers.tic('extract features')

        # pre-processing all the data, all the invalid inputs depth are set to 0
        invD0 = torch.clamp(inv_depth0_infer, 0, 10)
        invD1 = torch.clamp(inv_depth1_infer, 0, 10)
        invD0[invD0 == invD0.min()] = 0
        invD1[invD1 == invD1.min()] = 0
        invD0[invD0 == invD0.max()] = 0
        invD1[invD1 == invD1.max()] = 0

        # plot_one_tensor(invD0[0,:,:,:],name='invD0',type=2)

        # inv_w0_list = []
        # inv_w1_list = []
        # # inv_depth0_list.append(invD0)
        # # inv_depth1_list.append(invD1)

        # init_depth,更新1 1.0/d0[2]，更新2 1.0/d0[1]，更新3 1.0/d0[0]
        I0 = color_normalize(img0)
        I1 = color_normalize(img1)

        x0= self.__encode_features(I0, inv_depth0_infer)
        x1= self.__encode_features(I1, inv_depth1_infer)

        # # print(len(x0))
        # # print(feature0_infer.shape)
        # x0 = x0*feature0_infer
        # x1 = x1*feature1_infer

        d0 = self.construct_depth_pyramids(invD0)
        d1 = self.construct_depth_pyramids(invD1)

        # # # print(x0[0].shape)
        # # import tensorflow as tf
        # plot_one_tensor(torch.abs(x0[0]),'x0[0]',2)
        # plot_one_tensor(torch.abs(x1[0]),'x1[0]',2)

        d0_w = []
        for x in d0:
            d0_w.append(torch.zeros_like(x))
        d1_w = []
        for x in d1:
            d1_w.append(torch.zeros_like(x))

        if self.timers: self.timers.toc('extract features')

        feature_list = []

        feature_list.append(torch.sigmoid(feature1_infer))
        feature_list.append(self.downsample (feature_list[0]))
        feature_list.append(self.downsample (feature_list[1]))
        feature_list.append(self.downsample (feature_list[2]))

        poses_to_train = [[],[]] # '[rotation, translation]'
        B = invD0.shape[0]
        R0 = torch.eye(3,dtype=torch.float).expand(B,3,3).type_as(img0)
        t0 = torch.zeros(B,3,1,dtype=torch.float).type_as(img0)
        poseI = [R0, t0]

        # the prior of the mask
        prior_W = torch.ones(d0[3].shape).type_as(d0[3])

        # 更新pose
        if self.timers: self.timers.tic('trust-region update')
        K3 = K/8.0

        output3 = self.tr_update3(poseI, x0[3], x1[3], d0[3], d1[3], K3, prior_W,feature_list[3])

        pose3, mEst_W3= output3[0], output3[1]
        poses_to_train[0].append(pose3[0])
        poses_to_train[1].append(pose3[1])


        # 更新pose
        K2 = K/4.0
        output2 = self.tr_update2(pose3, x0[2], x1[2], d0[2], d1[2], K2, mEst_W3,feature_list[2])
        pose2, mEst_W2 = output2[0], output2[1]
        poses_to_train[0].append(pose2[0])
        poses_to_train[1].append(pose2[1])

        # 更新pose
        K1 = K/2.0
        output1 = self.tr_update1(pose2, x0[1], x1[1], d0[1], d1[1], K1, mEst_W2,feature_list[1])
        pose1, mEst_W1= output1[0], output1[1]
        poses_to_train[0].append(pose1[0])
        poses_to_train[1].append(pose1[1])


        # 更新pose
        output0 = self.tr_update0(pose1, x0[0], x1[0], d0[0], d1[0], K, mEst_W1,feature_list[0])
        pose0= output0[0]
        poses_to_train[0].append(pose0[0])
        poses_to_train[1].append(pose0[1])



        # return pose0,inv_depth0_infer, inv_depth1_infer
        return pose0,inv_depth0_infer, x0[0]


    
    def __encode_features(self, img0, invD0, img1, invD1):
        """ get the encoded features
        """
        if self.encoder_type == self.RGB:
            # In the RGB case, we will only use the intensity image
            I = self.__color3to1(img0)
            x = self.construct_image_pyramids(I)
        elif self.encoder_type == self.CONV_RGBD:
            m = torch.cat((img0, invD0), dim=1)
            x = self.encoder.forward(m)
        elif self.encoder_type in [self.CONV_RGBD2]:
            # print('img0 shape:', img0.shape)
            # print('invD0 shape:', invD0.shape)
            # print('img1 shape:', img1.shape)
            # print('invD1 shape:', invD1.shape)
            m = torch.cat((img0, invD0, img1, invD1), dim=1)
            x = self.encoder.forward(m)
            # feature_list = []

            # feature_list.append(torch.sigmoid(feature0))
            # feature_list.append(self.downsample (feature_list[0]))
            # feature_list.append(self.downsample (feature_list[1]))
            # feature_list.append(self.downsample (feature_list[2]))
            
            # print(feature0.shape)
            # print(x.shape)
            # x = x*feature0
        else:
            raise NotImplementedError()

        # new_x = []
        # for a in x:
        #     new_x.append(self.__Nto1(a))
        # x = new_x
        # new_x = []
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

    def __color3to1(self, img):
        """ Return a gray-scale image
        """
        B, _, H, W = img.shape
        return (img[:,0] * 0.299 + img[:, 1] * 0.587 + img[:, 2] * 0.114).view(B,1,H,W)
    

class PoseNet(nn.Module):

    def __init__(self, max_iter_per_pyr, timers = None):
        """
        # encoder_name：param用于回归的主干网络。
        max_iter_per_pyr：param每个金字塔级别的最大迭代次数
        # mEst_type：param表示加权函数的类型。
        # solver_type：param求解器的类型。
        # tr_samples = 10：信任区域求解器中的样本数参数
        # no_weight_sharing = False：param如果我们不想在不同的金字塔级别共享权重，则为True
        timers = None：param（可选）每个步骤消耗的基准时间
        """

        super(PoseNet, self).__init__()

        self.construct_feature_weight_pyramids = ImagePyramids([0,1,2,3], pool='avg')
        self.construct_depth_pyramids = ImagePyramids([0,1,2,3], pool='max')
        self.downsample = torch.nn.AvgPool2d(kernel_size=2)

        self.timers = timers
        """ =============================================================== """
        """             Initialize the Robust Estimator                     """
        """ =============================================================== """
        self.mEst_func = DeepRobustEstimator()
        mEst_funcs = [self.mEst_func, self.mEst_func, self.mEst_func,
        self.mEst_func]

        """ =============================================================== """
        """             Initialize the Trust-Region Damping                 """
        """ =============================================================== """

        self.solver_func3 = LearnedSolverNet((30,40))
        self.solver_func2 = LearnedSolverNet((60,80))
        self.solver_func1 = LearnedSolverNet((120,160))
        self.solver_func0 = LearnedSolverNet((240,320))
        solver_funcs = [self.solver_func0, self.solver_func1,
            self.solver_func2, self.solver_func3]
        
        """ =============================================================== """
        """             Initialize the Trust-Region Method                  """
        """ =============================================================== """

        self.tr_update0 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[0],
            solver_func = solver_funcs[0],
            timers      = timers)
        self.tr_update1 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[1],
            solver_func = solver_funcs[1],
            timers      = timers)
        self.tr_update2 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[2],
            solver_func = solver_funcs[2],
            timers      = timers)
        self.tr_update3 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[3],
            solver_func = solver_funcs[3],
            timers      = timers)
        
    def forward(self, pose0,img0, img1, inv_depth0_infer, inv_depth1_infer,feature0_infer,feature1_infer, weight1_infer,K,temp_K):

        # img0, img1, depth0_gt, depth1_gt ,feature0_infer,feature1_infer,weight1_infer, K
        """
        :input
        :param the reference image
        :param the target image
        :param the inverse depth of the reference image
        :param the inverse depth of the target image
        :param the pin-hole camera instrinsic (in vector) [fx, fy, cx, cy] 
        :param the initial pose [Rotation, translation]
        --------
        :return 
        :param estimated transform 
        # """

        if self.timers: self.timers.tic('extract features')

        # pre-processing all the data, all the invalid inputs depth are set to 0
        invD0 = torch.clamp(inv_depth0_infer, 0, 10)
        invD1 = torch.clamp(inv_depth1_infer, 0, 10)

        # 4rknn替换
        # invD0[invD0 == invD0.min()] = 0
        # invD1[invD1 == invD1.min()] = 0
        # invD0[invD0 == invD0.max()] = 0
        # invD1[invD1 == invD1.max()] = 0
        eps = 1e-8  # 数值稳定性阈值

        # 处理invD0的最小值和最大值替换
        min0 = invD0.min()
        max0 = invD0.max()

        # 生成最小值掩码（1.0表示需要替换，0.0表示保留）
        # 等价于 (invD0 == min0)：当invD0与min0的差值绝对值小于eps时为1.0
        diff_min0 = torch.abs(invD0 - min0)
        mask_min0 = torch.sign(eps - diff_min0)  # 差值<eps时为1.0，否则为-1.0
        mask_min0 = torch.clamp(mask_min0, 0.0, 1.0)  # 截断为0.0或1.0（float类型）

        # 生成最大值掩码（等价于 invD0 == max0）
        diff_max0 = torch.abs(invD0 - max0)
        mask_max0 = torch.sign(eps - diff_max0)
        mask_max0 = torch.clamp(mask_max0, 0.0, 1.0)

        # 应用掩码：需要替换的位置置0，其余保留原值
        invD0 = invD0 * (1.0 - mask_min0 - mask_max0)

        # 对invD1执行相同操作
        min1 = invD1.min()
        max1 = invD1.max()

        diff_min1 = torch.abs(invD1 - min1)
        mask_min1 = torch.sign(eps - diff_min1)
        mask_min1 = torch.clamp(mask_min1, 0.0, 1.0)

        diff_max1 = torch.abs(invD1 - max1)
        mask_max1 = torch.sign(eps - diff_max1)
        mask_max1 = torch.clamp(mask_max1, 0.0, 1.0)

        invD1 = invD1 * (1.0 - mask_min1 - mask_max1)

        I0 = color_normalize(img0)
        I1 = color_normalize(img1)

        # x0= self.construct_feature_weight_pyramids(feature0_infer)
        # x1= self.construct_feature_weight_pyramids(feature1_infer)

        x0= feature0_infer
        x1= feature1_infer

        # weight_list= self.construct_feature_weight_pyramids(weight1_infer)

        d0 = self.construct_depth_pyramids(invD0)
        d1 = self.construct_depth_pyramids(invD1)


        d0_w = []
        for x in d0:
            d0_w.append(torch.zeros_like(x))
        d1_w = []
        for x in d1:
            d1_w.append(torch.zeros_like(x))

        if self.timers: self.timers.toc('extract features')

        weight_list = []
        weight_list.append(torch.sigmoid(weight1_infer))
        weight_list.append(self.downsample (weight_list[0]))
        weight_list.append(self.downsample (weight_list[1]))
        weight_list.append(self.downsample (weight_list[2]))

        poses_to_train = [[],[]] # '[rotation, translation]'
        B = invD0.shape[0]
        # R0 = torch.eye(3,dtype=torch.float).expand(B,3,3).type_as(img0)
        # t0 = torch.zeros(B,3,1,dtype=torch.float).type_as(img0)
        poseI = pose0

        # the prior of the mask
        prior_W = torch.ones(d0[3].shape).type_as(d0[3])

        # 更新pose
        if self.timers: self.timers.tic('trust-region update')
        K3 = K/8.0

        output3 = self.tr_update3(poseI, x0[3], x1[3], d0[3], d1[3], K3, prior_W,weight_list[3])

        pose3, mEst_W3= output3[0], output3[1]
        poses_to_train[0].append(pose3[0])
        poses_to_train[1].append(pose3[1])


        # 更新pose
        K2 = K/4.0
        output2 = self.tr_update2(pose3, x0[2], x1[2], d0[2], d1[2], K2, mEst_W3,weight_list[2])
        pose2, mEst_W2 = output2[0], output2[1]
        poses_to_train[0].append(pose2[0])
        poses_to_train[1].append(pose2[1])

        # 更新pose
        K1 = K/2.0
        output1 = self.tr_update1(pose2, x0[1], x1[1], d0[1], d1[1], K1, mEst_W2,weight_list[1])
        pose1, mEst_W1= output1[0], output1[1]
        poses_to_train[0].append(pose1[0])
        poses_to_train[1].append(pose1[1])


        # 更新pose
        output0 = self.tr_update0(pose1, x0[0], x1[0], d0[0], d1[0], K, mEst_W1,weight_list[0])
        pose0= output0[0]
        poses_to_train[0].append(pose0[0])
        poses_to_train[1].append(pose0[1])

        # return pose0,inv_depth0_infer, inv_depth1_infer
        # pose_4x4 = pose_to_homogeneous(pose0[0],pose0[1])
        # return pose_4x4
        return pose0
class PoseNet_i(nn.Module):

    def __init__(self, max_iter_per_pyr, timers = None):
        """
        # encoder_name：param用于回归的主干网络。
        max_iter_per_pyr：param每个金字塔级别的最大迭代次数
        # mEst_type：param表示加权函数的类型。
        # solver_type：param求解器的类型。
        # tr_samples = 10：信任区域求解器中的样本数参数
        # no_weight_sharing = False：param如果我们不想在不同的金字塔级别共享权重，则为True
        timers = None：param（可选）每个步骤消耗的基准时间
        """

        super(PoseNet_i, self).__init__()

        self.construct_feature_weight_pyramids = ImagePyramids([0,1,2,3], pool='avg')
        self.construct_depth_pyramids = ImagePyramids([0,1,2,3], pool='max')
        self.downsample = torch.nn.AvgPool2d(kernel_size=2)

        self.timers = timers
        """ =============================================================== """
        """             Initialize the Robust Estimator                     """
        """ =============================================================== """
        self.mEst_func = DeepRobustEstimator()
        mEst_funcs = [self.mEst_func, self.mEst_func, self.mEst_func,
        self.mEst_func]

        """ =============================================================== """
        """             Initialize the Trust-Region Damping                 """
        """ =============================================================== """

        self.solver_func3 = LearnedSolverNet((30,40))
        self.solver_func2 = LearnedSolverNet((60,80))
        self.solver_func1 = LearnedSolverNet((120,160))
        self.solver_func0 = LearnedSolverNet((240,320))
        solver_funcs = [self.solver_func0, self.solver_func1,
            self.solver_func2, self.solver_func3]
        
        """ =============================================================== """
        """             Initialize the Trust-Region Method                  """
        """ =============================================================== """

        self.tr_update0 = TrustRegionBase_i(max_iter_per_pyr,
            mEst_func   = mEst_funcs[0],
            solver_func = solver_funcs[0],
            timers      = timers)
        self.tr_update1 = TrustRegionBase_i(max_iter_per_pyr,
            mEst_func   = mEst_funcs[1],
            solver_func = solver_funcs[1],
            timers      = timers)
        self.tr_update2 = TrustRegionBase_i(max_iter_per_pyr,
            mEst_func   = mEst_funcs[2],
            solver_func = solver_funcs[2],
            timers      = timers)
        self.tr_update3 = TrustRegionBase_i(max_iter_per_pyr,
            mEst_func   = mEst_funcs[3],
            solver_func = solver_funcs[3],
            timers      = timers)
        
    def forward(self, pose0,img0, img1, inv_depth0_infer, inv_depth1_infer,feature0_infer,feature1_infer, weight1_infer,K,temp_K):

        # img0, img1, depth0_gt, depth1_gt ,feature0_infer,feature1_infer,weight1_infer, K
        """
        :input
        :param the reference image
        :param the target image
        :param the inverse depth of the reference image
        :param the inverse depth of the target image
        :param the pin-hole camera instrinsic (in vector) [fx, fy, cx, cy] 
        :param the initial pose [Rotation, translation]
        --------
        :return 
        :param estimated transform 
        # """

        if self.timers: self.timers.tic('extract features')

        # pre-processing all the data, all the invalid inputs depth are set to 0
        invD0 = torch.clamp(inv_depth0_infer, 0, 10)
        invD1 = torch.clamp(inv_depth1_infer, 0, 10)
        invD0[invD0 == invD0.min()] = 0
        invD1[invD1 == invD1.min()] = 0
        invD0[invD0 == invD0.max()] = 0
        invD1[invD1 == invD1.max()] = 0

        I0 = color_normalize(img0)
        I1 = color_normalize(img1)

        # x0= self.construct_feature_weight_pyramids(feature0_infer)
        # x1= self.construct_feature_weight_pyramids(feature1_infer)

        x0= feature0_infer
        x1= feature1_infer

        # weight_list= self.construct_feature_weight_pyramids(weight1_infer)

        d0 = self.construct_depth_pyramids(invD0)
        d1 = self.construct_depth_pyramids(invD1)


        d0_w = []
        for x in d0:
            d0_w.append(torch.zeros_like(x))
        d1_w = []
        for x in d1:
            d1_w.append(torch.zeros_like(x))

        if self.timers: self.timers.toc('extract features')

        weight_list = []
        epsilon = 1e-8
        # 仅适用于PyTorch 1.8+
        min_vals = torch.amin(weight1_infer, dim=(2, 3), keepdim=True)
        max_vals = torch.amax(weight1_infer, dim=(2, 3), keepdim=True)

        weight1_infer = (weight1_infer - min_vals) / torch.clamp(max_vals - min_vals, min=epsilon)
        
        weight_list.append(torch.sigmoid(weight1_infer))
        weight_list.append(self.downsample (weight_list[0]))
        weight_list.append(self.downsample (weight_list[1]))
        weight_list.append(self.downsample (weight_list[2]))

        poses_to_train = [[],[]] # '[rotation, translation]'
        B = invD0.shape[0]
        # R0 = torch.eye(3,dtype=torch.float).expand(B,3,3).type_as(img0)
        # t0 = torch.zeros(B,3,1,dtype=torch.float).type_as(img0)
        poseI = pose0

        # the prior of the mask
        prior_W = torch.ones(d0[3].shape).type_as(d0[3])

        # 更新pose
        if self.timers: self.timers.tic('trust-region update')
        K3 = K/8.0

        # print(x0[3].shape)
        # print(d0[3].shape)
        # print(weight_list[3].shape)

        output3 = self.tr_update3(poseI, x0[3], x1[3], d0[3], d1[3], K3, prior_W,weight_list[3])

        pose3, mEst_W3= output3[0], output3[1]
        poses_to_train[0].append(pose3[0])
        poses_to_train[1].append(pose3[1])


        # 更新pose
        K2 = K/4.0
        output2 = self.tr_update2(pose3, x0[2], x1[2], d0[2], d1[2], K2, mEst_W3,weight_list[2])
        pose2, mEst_W2 = output2[0], output2[1]
        poses_to_train[0].append(pose2[0])
        poses_to_train[1].append(pose2[1])

        # 更新pose
        K1 = K/2.0
        output1 = self.tr_update1(pose2, x0[1], x1[1], d0[1], d1[1], K1, mEst_W2,weight_list[1])
        pose1, mEst_W1= output1[0], output1[1]
        poses_to_train[0].append(pose1[0])
        poses_to_train[1].append(pose1[1])


        # 更新pose
        output0 = self.tr_update0(pose1, x0[0], x1[0], d0[0], d1[0], K, mEst_W1,weight_list[0])
        pose0= output0[0]
        poses_to_train[0].append(pose0[0])
        poses_to_train[1].append(pose0[1])

        # return pose0,inv_depth0_infer, inv_depth1_infer
        # pose_4x4 = pose_to_homogeneous(pose0[0],pose0[1])
        # return pose_4x4
        return pose0
    
class PoseNet_IMU(nn.Module):

    def __init__(self, max_iter_per_pyr, timers = None):
        """
        # encoder_name：param用于回归的主干网络。
        max_iter_per_pyr：param每个金字塔级别的最大迭代次数
        # mEst_type：param表示加权函数的类型。
        # solver_type：param求解器的类型。
        # tr_samples = 10：信任区域求解器中的样本数参数
        # no_weight_sharing = False：param如果我们不想在不同的金字塔级别共享权重，则为True
        timers = None：param（可选）每个步骤消耗的基准时间
        """

        super(PoseNet_IMU, self).__init__()

        self.construct_feature_weight_pyramids = ImagePyramids([0,1,2,3], pool='avg')
        self.construct_depth_pyramids = ImagePyramids([0,1,2,3], pool='max')
        self.downsample = torch.nn.AvgPool2d(kernel_size=2)

        self.timers = timers
        """ =============================================================== """
        """             Initialize the Robust Estimator                     """
        """ =============================================================== """
        self.mEst_func = DeepRobustEstimator()
        mEst_funcs = [self.mEst_func, self.mEst_func, self.mEst_func,
        self.mEst_func]

        """ =============================================================== """
        """             Initialize the Trust-Region Damping                 """
        """ =============================================================== """

        self.solver_func3 = LearnedSolverNet((30,40))
        self.solver_func2 = LearnedSolverNet((60,80))
        self.solver_func1 = LearnedSolverNet((120,160))
        self.solver_func0 = LearnedSolverNet((240,320))
        solver_funcs = [self.solver_func0, self.solver_func1,
            self.solver_func2, self.solver_func3]
        
        """ =============================================================== """
        """             Initialize the Trust-Region Method                  """
        """ =============================================================== """

        self.tr_update0 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[0],
            solver_func = solver_funcs[0],
            timers      = timers)
        self.tr_update1 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[1],
            solver_func = solver_funcs[1],
            timers      = timers)
        self.tr_update2 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[2],
            solver_func = solver_funcs[2],
            timers      = timers)
        self.tr_update3 = TrustRegion(max_iter_per_pyr,
            mEst_func   = mEst_funcs[3],
            solver_func = solver_funcs[3],
            timers      = timers)
        
    def forward(self,pose_init, img0, img1, inv_depth0_infer, inv_depth1_infer,feature0_infer,feature1_infer, weight1_infer,K,temp_K):

        # img0, img1, depth0_gt, depth1_gt ,feature0_infer,feature1_infer,weight1_infer, K
        """
        :input
        :param the reference image
        :param the target image
        :param the inverse depth of the reference image
        :param the inverse depth of the target image
        :param the pin-hole camera instrinsic (in vector) [fx, fy, cx, cy] 
        :param the initial pose [Rotation, translation]
        --------
        :return 
        :param estimated transform 
        # """

        if self.timers: self.timers.tic('extract features')

        # pre-processing all the data, all the invalid inputs depth are set to 0
        invD0 = torch.clamp(inv_depth0_infer, 0, 10)
        invD1 = torch.clamp(inv_depth1_infer, 0, 10)
        invD0[invD0 == invD0.min()] = 0
        invD1[invD1 == invD1.min()] = 0
        invD0[invD0 == invD0.max()] = 0
        invD1[invD1 == invD1.max()] = 0

        I0 = color_normalize(img0)
        I1 = color_normalize(img1)

        # x0= self.construct_feature_weight_pyramids(feature0_infer)
        # x1= self.construct_feature_weight_pyramids(feature1_infer)

        x0= feature0_infer
        x1= feature1_infer

        weight_list= self.construct_feature_weight_pyramids(weight1_infer)

        d0 = self.construct_depth_pyramids(invD0)
        d1 = self.construct_depth_pyramids(invD1)


        d0_w = []
        for x in d0:
            d0_w.append(torch.zeros_like(x))
        d1_w = []
        for x in d1:
            d1_w.append(torch.zeros_like(x))

        if self.timers: self.timers.toc('extract features')

        # weight_list = []
        # weight_list.append(torch.sigmoid(weight1_infer))
        # weight_list.append(self.downsample (weight_list[0]))
        # weight_list.append(self.downsample (weight_list[1]))
        # weight_list.append(self.downsample (weight_list[2]))

        poses_to_train = [[],[]] # '[rotation, translation]'
        B = invD0.shape[0]
        # R0 = torch.eye(3,dtype=torch.float).expand(B,3,3).type_as(img0)
        # t0 = torch.zeros(B,3,1,dtype=torch.float).type_as(img0)
        poseI = pose_init

        # the prior of the mask
        prior_W = torch.ones(d0[3].shape).type_as(d0[3])

        # 更新pose
        if self.timers: self.timers.tic('trust-region update')
        K3 = K/8.0

        # print(x0[3].shape)
        # print(d0[3].shape)
        # print(weight_list[3].shape)
        # print(poseI[0].device)
        # print(x0.device)
        # print(K3.device)
        # print(weight_list.device)

        output3 = self.tr_update3(poseI, x0[3], x1[3], d0[3], d1[3], K3, prior_W,weight_list[3])

        pose3, mEst_W3= output3[0], output3[1]
        poses_to_train[0].append(pose3[0])
        poses_to_train[1].append(pose3[1])


        # 更新pose
        K2 = K/4.0
        output2 = self.tr_update2(pose3, x0[2], x1[2], d0[2], d1[2], K2, mEst_W3,weight_list[2])
        pose2, mEst_W2 = output2[0], output2[1]
        poses_to_train[0].append(pose2[0])
        poses_to_train[1].append(pose2[1])

        # 更新pose
        K1 = K/2.0
        output1 = self.tr_update1(pose2, x0[1], x1[1], d0[1], d1[1], K1, mEst_W2,weight_list[1])
        pose1, mEst_W1= output1[0], output1[1]
        poses_to_train[0].append(pose1[0])
        poses_to_train[1].append(pose1[1])


        # 更新pose
        output0 = self.tr_update0(pose1, x0[0], x1[0], d0[0], d1[0], K, mEst_W1,weight_list[0])
        pose0= output0[0]
        poses_to_train[0].append(pose0[0])
        poses_to_train[1].append(pose0[1])

        # return pose0,inv_depth0_infer, inv_depth1_infer
        # pose_4x4 = pose_to_homogeneous(pose0[0],pose0[1])
        # return pose_4x4
        return pose0

# def pose_to_homogeneous(rotation, translation):
#     """
#     将旋转矩阵和平移向量转换为齐次变换矩阵
    
#     参数:
#         rotation: 旋转矩阵，形状为 (batch, 3, 3)
#         translation: 平移向量，形状为 (batch, 3)
    
#     返回:
#         齐次变换矩阵，形状为 (batch, 4, 4)
#     """
#     # 确保输入维度正确
#     assert rotation.shape[-2:] == (3, 3), "rotation 必须是 (batch, 3, 3)"
#     assert translation.shape[-1] == 3, "translation 必须是 (batch, 3)"
    
#     # 将平移向量从 (batch, 3) 扩展为 (batch, 3, 1) 的列向量
#     t_column = translation.unsqueeze(-1)  # 形状: (batch, 3, 1)
    
#     # 拼接旋转矩阵和位移列向量，得到上半部分 (batch, 3, 4)
#     upper_part = torch.cat([rotation, t_column], dim=-1)  # 形状: (batch, 3, 4)
    
#     # 创建齐次坐标行：[0, 0, 0, 1]，形状为 (batch, 1, 4)
#     homogeneous_row = torch.zeros((rotation.shape[0], 1, 4), dtype=rotation.dtype, device=rotation.device)
#     homogeneous_row[..., -1] = 1  # 最后一个元素设为 1
    
#     # 拼接上下两部分，得到完整的齐次变换矩阵 (batch, 4, 4)
#     homogeneous_matrix = torch.cat([upper_part, homogeneous_row], dim=-2)  # 沿第 2 维拼接
    
#     return homogeneous_matrix
