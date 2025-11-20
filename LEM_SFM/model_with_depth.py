import torch.nn as nn
import torch
import numpy as np
import random


import LEM_SFM.train_utils as train_utils


def check_cuda(items):
    if torch.cuda.is_available():
        return [x.cuda() for x in items]
    else:
        return items

class DepthPoseNet(nn.Module):
    """
    D:1.6M
    F:
    W:1.6M
    IMU:dt_input
    """
    def __init__(self,model_args=None):
        super(DepthPoseNet, self).__init__()

        self.model_args = model_args
        if model_args==None:
            model_args.max_iter_per_pyr = 3
        
        self.depth_size = (256,320)
        self.img_size = (240,320)


        # depth model
        import LEM_SFM.DM.starnet_src.midas_starnet as depthnet
        self.depthnet = depthnet.MidasNet_small7()

        # feature model
        from LEM_SFM.models.algorithms import FeaturePyramid_new
        self.featurenet = FeaturePyramid_new(D=3)
        
        # # weight model
        self.weightnet = depthnet.MidasNet_small7_weight()

        # pose model
        import LEM_SFM.models.LeastSquareTracking as ICtracking
        self.posenet = ICtracking.PoseNet(max_iter_per_pyr= model_args.max_iter_per_pyr)

        num_params = train_utils.count_parameters(self.depthnet)
        print('There is a total of {:} learnabled parameters in depth_net'.format(num_params))
        num_params = train_utils.count_parameters(self.weightnet)
        print('There is a total of {:} learnabled parameters in weight_net'.format(num_params))
        num_params = train_utils.count_parameters(self.featurenet)
        print('There is a total of {:} learnabled parameters in featurenet'.format(num_params))
        num_params = train_utils.count_parameters(self.posenet)
        print('There is a total of {:} learnabled parameters in posenet'.format(num_params))

    
    def forward(self, img0,img1,K,pose0):
        self.img_size = (img0.shape[2],img0.shape[3])

        temp0 = torch.nn.functional.interpolate(img0,size=self.depth_size,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None)
        temp1 = torch.nn.functional.interpolate(img1,size=self.depth_size,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None)

        depth0 = self.depthnet(temp0).unsqueeze(1)
        depth1 = self.depthnet(temp1).unsqueeze(1)

        depth0 = torch.nn.functional.interpolate(depth0,size=self.img_size,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None)
        depth1 = torch.nn.functional.interpolate(depth1,size=self.img_size,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None)

        inv_depth0 = torch.where(depth0 != 0,1.0/depth0, depth0)
        inv_depth1 = torch.where(depth1 != 0,1.0/depth1,depth1)

        weight1_infer = self.weightnet(temp1)
        
        feature0_infer = self.featurenet(img0)
        feature1_infer = self.featurenet(img1)

        weight1_infer = torch.nn.functional.interpolate(weight1_infer.unsqueeze(1),size=self.img_size,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None)
        
        pose = self.posenet(pose0,img0, img1, inv_depth0, inv_depth1 ,feature0_infer,feature1_infer,weight1_infer,K,K)
        return pose,depth0


class DepthPoseNet_Deeplabv3(nn.Module):
    """
    D:1.6M
    F:
    W:Deeplabv3
    IMU:dt_input
    """
    def __init__(self,model_args=None):
        super(DepthPoseNet_Deeplabv3, self).__init__()

        self.model_args = model_args
        if model_args==None:
            model_args.max_iter_per_pyr = 3
        
        self.depth_size = (256,320)
        self.img_size = (240,320)


        # depth model
        import LEM_SFM.DM.starnet_src.midas_starnet as depthnet
        self.depthnet = depthnet.MidasNet_small7()

        # feature model
        from LEM_SFM.models.algorithms import FeaturePyramid_new
        self.featurenet = FeaturePyramid_new(D=3)

        # pose model
        import LEM_SFM.models.LeastSquareTracking as ICtracking
        # self.posenet = ICtracking.PoseNet(max_iter_per_pyr= model_args.max_iter_per_pyr)
        self.posenet = ICtracking.PoseNet(max_iter_per_pyr= model_args.max_iter_per_pyr)

        # # weight model
        from LEM_SFM.config import get_Deeplabv3_argparser
        deeplabv3_args = get_Deeplabv3_argparser()

        if deeplabv3_args.dataset.lower() == 'voc':
            deeplabv3_args.num_classes = 21
        elif deeplabv3_args.dataset.lower() == 'cityscapes':
            deeplabv3_args.num_classes = 19

        # Setup visualization
        from LEM_SFM.Deeplabv3.utils.visualizer import Visualizer
        from LEM_SFM.Deeplabv3 import network
        from LEM_SFM.Deeplabv3 import utils

        vis = Visualizer(port=deeplabv3_args.vis_port,
                        env=deeplabv3_args.vis_env) if deeplabv3_args.enable_vis else None
        if vis is not None:  # display options
            vis.vis_table("Options", vars(deeplabv3_args))

        # os.environ['CUDA_VISIBLE_DEVICES'] = deeplabv3_args.gpu_id
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % device)

        # Setup random seed
        torch.manual_seed(deeplabv3_args.random_seed)
        np.random.seed(deeplabv3_args.random_seed)
        random.seed(deeplabv3_args.random_seed)

        # Setup dataloader
        if deeplabv3_args.dataset=='voc' and not deeplabv3_args.crop_val:
            deeplabv3_args.val_batch_size = 1
        
        # Set up model
        model_map = {
            'deeplabv3_resnet50': network.deeplabv3_resnet50,
            'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
            'deeplabv3_resnet101': network.deeplabv3_resnet101,
            'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
            'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
            'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
        }

        deeplabv3 = model_map[deeplabv3_args.model](num_classes=deeplabv3_args.num_classes, output_stride=deeplabv3_args.output_stride)
        if deeplabv3_args.separable_conv and 'plus' in deeplabv3_args.model:
            network.convert_to_separable_conv(deeplabv3.classifier)
        
        utils.set_bn_momentum(deeplabv3.backbone, momentum=0.01)

        # self.tempnet = nn.Conv2d()
        from LEM_SFM.models.submodules import convLayer as conv
        from LEM_SFM.models.submodules import initialize_weights
        self.weightnet = nn.Sequential(deeplabv3,
            conv(True, 19, 5,  3, dilation=1),
            conv(True, 5, 1,  3, dilation=1),
            nn.ReLU() )

        num_params = train_utils.count_parameters(self.depthnet)
        print('There is a total of {:} learnabled parameters in depth_net'.format(num_params))
        num_params = train_utils.count_parameters(self.weightnet)
        print('There is a total of {:} learnabled parameters in weight_net'.format(num_params))
        num_params = train_utils.count_parameters(self.featurenet)
        print('There is a total of {:} learnabled parameters in featurenet'.format(num_params))
        num_params = train_utils.count_parameters(self.posenet)
        print('There is a total of {:} learnabled parameters in posenet'.format(num_params))


    def forward(self, img0,img1,K,pose0):

        temp0 = torch.nn.functional.interpolate(img0,size=self.depth_size,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None)
        temp1 = torch.nn.functional.interpolate(img1,size=self.depth_size,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None)

        depth0 = self.depthnet(temp0).unsqueeze(1)
        depth1 = self.depthnet(temp1).unsqueeze(1)

        depth0 = torch.nn.functional.interpolate(depth0,size=self.img_size,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None)
        depth1 = torch.nn.functional.interpolate(depth1,size=self.img_size,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None)

        inv_depth0 = torch.where(depth0 != 0,1.0/depth0, depth0)
        inv_depth1 = torch.where(depth1 != 0,1.0/depth1,depth1)

        weight1_infer = self.weightnet(temp1)
        weight1_infer = torch.nn.functional.interpolate(weight1_infer,size=self.img_size,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None)

        feature0_infer = self.featurenet(img0)
        feature1_infer = self.featurenet(img1)

        pose = self.posenet(pose0,img0, img1, inv_depth0, inv_depth1 ,feature0_infer,feature1_infer,weight1_infer,K,K)

        return pose,depth0
    


   
class DepthPoseNet_Deeplabv3_plot(nn.Module):
    """
    D:1.6M
    F:
    W:Deeplabv3
    IMU:dt_input
    """
    def __init__(self,model_args=None):
        super(DepthPoseNet_Deeplabv3_plot, self).__init__()

        self.model_args = model_args
        if model_args==None:
            model_args.max_iter_per_pyr = 3
        
        self.depth_size = (256,320)
        self.img_size = (240,320)


        # depth model
        import LEM_SFM.DM.starnet_src.midas_starnet as depthnet
        self.depthnet = depthnet.MidasNet_small7()

        # feature model
        from LEM_SFM.models.algorithms import FeaturePyramid_new
        self.featurenet = FeaturePyramid_new(D=3)

        # # weight model
        from LEM_SFM.config import get_Deeplabv3_argparser
        deeplabv3_args = get_Deeplabv3_argparser()

        if deeplabv3_args.dataset.lower() == 'voc':
            deeplabv3_args.num_classes = 21
        elif deeplabv3_args.dataset.lower() == 'cityscapes':
            deeplabv3_args.num_classes = 19

        # Setup visualization
        from LEM_SFM.Deeplabv3.utils.visualizer import Visualizer
        from LEM_SFM.Deeplabv3 import network
        from LEM_SFM.Deeplabv3 import utils

        vis = Visualizer(port=deeplabv3_args.vis_port,
                        env=deeplabv3_args.vis_env) if deeplabv3_args.enable_vis else None
        if vis is not None:  # display options
            vis.vis_table("Options", vars(deeplabv3_args))

        # os.environ['CUDA_VISIBLE_DEVICES'] = deeplabv3_args.gpu_id
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % device)

        # Setup random seed
        torch.manual_seed(deeplabv3_args.random_seed)
        np.random.seed(deeplabv3_args.random_seed)
        random.seed(deeplabv3_args.random_seed)

        # Setup dataloader
        if deeplabv3_args.dataset=='voc' and not deeplabv3_args.crop_val:
            deeplabv3_args.val_batch_size = 1
        
        # Set up model
        model_map = {
            'deeplabv3_resnet50': network.deeplabv3_resnet50,
            'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
            'deeplabv3_resnet101': network.deeplabv3_resnet101,
            'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
            'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
            'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
        }

        deeplabv3 = model_map[deeplabv3_args.model](num_classes=deeplabv3_args.num_classes, output_stride=deeplabv3_args.output_stride)
        if deeplabv3_args.separable_conv and 'plus' in deeplabv3_args.model:
            network.convert_to_separable_conv(deeplabv3.classifier)
        
        utils.set_bn_momentum(deeplabv3.backbone, momentum=0.01)

        # self.tempnet = nn.Conv2d()
        from LEM_SFM.models.submodules import convLayer as conv
        from LEM_SFM.models.submodules import initialize_weights
        self.weightnet = nn.Sequential(deeplabv3,
            conv(True, 19, 5,  3, dilation=1),
            conv(True, 5, 1,  3, dilation=1),
            nn.ReLU() )

        # pose model
        import LEM_SFM.models.LeastSquareTracking as ICtracking
        self.posenet = ICtracking.PoseNet(max_iter_per_pyr= model_args.max_iter_per_pyr)

        num_params = train_utils.count_parameters(self.depthnet)
        print('There is a total of {:} learnabled parameters in depth_net'.format(num_params))
        num_params = train_utils.count_parameters(self.weightnet)
        print('There is a total of {:} learnabled parameters in weight_net'.format(num_params))
        num_params = train_utils.count_parameters(self.featurenet)
        print('There is a total of {:} learnabled parameters in featurenet'.format(num_params))
        num_params = train_utils.count_parameters(self.posenet)
        print('There is a total of {:} learnabled parameters in posenet'.format(num_params))

    
    def forward(self, img0,img1,K,pose0):
        self.img_size = (img0.shape[2],img0.shape[3])

        temp0 = torch.nn.functional.interpolate(img0,size=self.depth_size,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None)
        temp1 = torch.nn.functional.interpolate(img1,size=self.depth_size,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None)

        depth0 = self.depthnet(temp0).unsqueeze(1)
        depth1 = self.depthnet(temp1).unsqueeze(1)

        depth0 = torch.nn.functional.interpolate(depth0,size=self.img_size,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None)
        depth1 = torch.nn.functional.interpolate(depth1,size=self.img_size,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None)

        inv_depth0 = torch.where(depth0 != 0,1.0/depth0, depth0)
        inv_depth1 = torch.where(depth1 != 0,1.0/depth1,depth1)

        weight1_infer = self.weightnet(temp1)
        # print(weight1_infer.shape)
        feature0_infer = self.featurenet(img0)
        feature1_infer = self.featurenet(img1)

        weight1_infer = torch.nn.functional.interpolate(weight1_infer,size=self.img_size,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None)

        pose = self.posenet(pose0,img0, img1, inv_depth0, inv_depth1 ,feature0_infer,feature1_infer,weight1_infer,K,K)
        return pose,depth0,depth1,feature0_infer[0],feature1_infer[0],weight1_infer
    

class LoFTR_class(nn.Module):
    """
    LoFTR_class: 使用LoFTR和SuperGlue计算相对位姿
    input:
        img0, img1: 图像对，形状为 (batch, 3, 320, 240)
        K: 相机内参矩阵，形状为 (batch, 4)，其中每行包含 [fx, fy, cx, cy]
    output:
        [R, t]: 相对位姿，其中 R.shape=(batch, 3, 3), t.shape=(batch, 3)
    """
    

    def __init__(self, model_args=None):
        super(LoFTR_class, self).__init__()
        # # 初始化SuperPoint和SuperGlue模型
        from LoFTR.src.loftr import LoFTR, default_cfg

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'


        from LoFTR.src.loftr.loftr import LoFTR

        # 加载默认配置并初始化 LoFTR
        self.config = default_cfg
        self.loftr = LoFTR(self.config).to(self.device)
        self.loftr.load_state_dict(torch.load("/home/code/reloc3r/reloc3r-main/LoFTR/weights/indoor_ds.ckpt")['state_dict'])


    def forward(self, img0, img1, K):
        batch_size = img0.shape[0]

        # 将图像转换为灰度图像
        img0_gray = torch.mean(img0, dim=1, keepdim=True)  # 转换为灰度图像
        img1_gray = torch.mean(img1, dim=1, keepdim=True)

        R_batch = []
        t_batch = []

        for i in range(batch_size):
            # 单独处理每对图像
            pred = {'image0': img0_gray[i:i+1].to(self.device), 
                                'image1': img1_gray[i:i+1].to(self.device)}
            
            self.loftr(pred)
            # print(pred)
            # 提取匹配点
            mkpts0 = pred['mkpts0_f'].cpu().numpy()  # 匹配点在 img0 中的坐标，形状为 (N, 2)
            mkpts1 = pred['mkpts1_f'].cpu().numpy()  # 匹配点在 img1 中的坐标，形状为 (N, 2)

            # # 筛选有效匹配点
            # valid = matches > -1
            # mkpts0 = kpts0[valid]
            # mkpts1 = kpts1[matches[valid]]

            # 构造内参矩阵
            fx, fy, cx, cy = K[i].cpu().numpy()
            K_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0,  0,  1]])

            # 缩放内参矩阵（假设图像大小未改变）
            scales = np.array([1.0, 1.0])
            K_scaled = scale_intrinsics(K_matrix, scales)

            # 姿态估计
            thresh = 1.0  # 像素阈值
            ret = estimate_pose(mkpts0, mkpts1, K_scaled, K_scaled, thresh)
            if ret is None:
                R_batch.append(torch.eye(3).to(self.device))
                t_batch.append(torch.zeros(3).to(self.device))
            else:
                R, t, inliers = ret
                R_batch.append(torch.tensor(R, dtype=torch.float32).to(self.device))
                t_batch.append(torch.tensor(t, dtype=torch.float32).to(self.device))

        # 堆叠结果
        R_batch = torch.stack(R_batch, dim=0)
        t_batch = torch.stack(t_batch, dim=0)
        # print(R_batch.shape,t_batch.shape)

        return [R_batch, t_batch]
    
class SPSG(nn.Module):
    """
    SPSG: 使用SuperPoint和SuperGlue计算相对位姿
    input:
        img0, img1: 图像对，形状为 (batch, 3, 320, 240)
        K: 相机内参矩阵，形状为 (batch, 4)，其中每行包含 [fx, fy, cx, cy]
    output:
        [R, t]: 相对位姿，其中 R.shape=(batch, 3, 3), t.shape=(batch, 3)
    """
    def __init__(self, model_args=None):
        super(SPSG, self).__init__()
        from SpSg.models.matching import Matching

        # 初始化SuperPoint和SuperGlue模型
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        self.model_args = model_args or {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        self.matching = Matching(self.model_args).eval().to(self.device)
        # self.matching = Matching(self.model_args).eval()


    def forward(self, img0, img1, K):
        from SpSg.models.utils import (compute_pose_error, estimate_pose, scale_intrinsics, 
                    rotate_intrinsics, rotate_pose_inplane, compute_epipolar_error)

        batch_size = img0.shape[0]

        # 将图像转换为灰度图像
        img0_gray = torch.mean(img0, dim=1, keepdim=True)  # 转换为灰度图像
        img1_gray = torch.mean(img1, dim=1, keepdim=True)

        R_batch = []
        t_batch = []

        for i in range(batch_size):
            # 单独处理每对图像
            pred = self.matching({'image0': img0_gray[i:i+1].to(self.device), 
                                  'image1': img1_gray[i:i+1].to(self.device)})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']

            # 筛选有效匹配点
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]

            # 构造内参矩阵
            fx, fy, cx, cy = K[i].cpu().numpy()
            K_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0,  0,  1]])

            # 缩放内参矩阵（假设图像大小未改变）
            scales = np.array([1.0, 1.0])
            K_scaled = scale_intrinsics(K_matrix, scales)

            # 姿态估计
            thresh = 1.0  # 像素阈值
            ret = estimate_pose(mkpts0, mkpts1, K_scaled, K_scaled, thresh)

            if ret is None:
                R_batch.append(torch.eye(3).to(self.device))
                t_batch.append(torch.zeros(3).to(self.device))
            else:
                R, t, inliers = ret
                R_batch.append(torch.tensor(R, dtype=torch.float32).to(self.device))
                t_batch.append(torch.tensor(t, dtype=torch.float32).to(self.device))

        # 堆叠结果
        R_batch = torch.stack(R_batch, dim=0)
        t_batch = torch.stack(t_batch, dim=0)
        # print(R_batch.shape,t_batch.shape)

        return [R_batch, t_batch]
    
class Classic(nn.Module):
    def __init__(self, 
                 feature_detector: str = 'SIFT', 
                 min_matches: int = 10,
                #  focal_length: float = (800.0,600),
                #  principal_point: Tuple[float, float] = (160.0, 120.0),
                 ransac_threshold: float = 5.0):
        """
        初始化相对位姿估计器
        
        参数:
            feature_detector: 使用的特征检测器类型 ('SIFT', 'SURF', 'ORB' 等)
            min_matches: 估计位姿所需的最小匹配点数
            focal_length: 相机焦距 (像素)
            principal_point: 相机主点 (cx, cy)
            ransac_threshold: RANSAC算法的阈值
        """
        # 初始化特征检测器
        super(Classic, self).__init__()
        if feature_detector == 'SIFT':
            self.detector = cv2.SIFT_create()
        elif feature_detector == 'SURF':
            self.detector = cv2.SURF_create()
        elif feature_detector == 'ORB':
            self.detector = cv2.ORB_create()
        else:
            raise ValueError(f"不支持的特征检测器: {feature_detector}")
        
        # 创建特征匹配器
        if feature_detector in ['SIFT', 'SURF']:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:  # 对于ORB等二进制描述符
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        

        
        # 畸变系数 (假设无畸变)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    def detect_and_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测特征点并进行匹配
        
        参数:
            img1, img2: 输入的两张图像 (H, W, 3)
        
        返回:
            pts1, pts2: 匹配的特征点坐标
        """
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 检测特征点和计算描述符
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        # 特征匹配
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # 应用比率测试以过滤匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # 提取匹配点的坐标
        if len(good_matches) < self.min_matches:
            # print(f"匹配点数量不足 ({len(good_matches)} < {self.min_matches})")
            raise ValueError(f"匹配点数量不足 ({len(good_matches)} < {self.min_matches})")
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        return pts1, pts2
    
    def estimate_pose(self, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        估计相对位姿
        
        参数:
            pts1, pts2: 匹配的特征点坐标
        
        返回:
            R: 旋转矩阵 (3x3)
            t: 平移向量 (3x1)
        """
        # 估计本质矩阵
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.camera_matrix, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=self.ransac_threshold
        )
        
        # 从本质矩阵恢复旋转和平移
        _, R, t, _ = cv2.recoverPose(
            E, pts1, pts2, self.camera_matrix, mask
        )
        
        return R, t
    

    def create_homogeneous_matrix(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        创建齐次变换矩阵
        
        参数:
            R: 旋转矩阵 (3x3)
            t: 平移向量 (3x1)
        
        返回:
            T: 齐次变换矩阵 (4x4)
        """
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T
    
    def torch_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """将PyTorch张量转换为NumPy数组并调整维度"""
        # 确保张量在CPU上
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # 转换为NumPy数组并调整维度 (B, C, H, W) -> (B, H, W, C)
        return tensor.permute(0, 2, 3, 1).numpy()
    
    def forward(self, imgs1: torch.Tensor, imgs2: torch.Tensor,K) -> torch.Tensor:
        """
        处理图像批次
        
        参数:
            imgs1, imgs2: 输入的图像批次 (B, 3, H, W), torch.Tensor类型
        
        返回:
            poses: 相对位姿矩阵批次 (B, 4, 4), torch.Tensor类型
        """
        batch_size = imgs1.shape[0]
        poses = np.zeros((batch_size, 4, 4))

        K= K[0].cpu().numpy()
        # print(K)

        # 相机内参矩阵
        self.camera_matrix = np.array([
            [K[0], 0, K[2]],
            [0, K[1], K[3]],
            [0, 0, 1]
        ], dtype=np.float32)
        # 将PyTorch张量转换为NumPy数组
        imgs1_np = self.torch_to_numpy(imgs1)
        imgs2_np = self.torch_to_numpy(imgs2)
        
        for i in range(batch_size):
            try:
                # 转换为uint8类型
                img1 = (imgs1_np[i] * 255).astype(np.uint8)
                img2 = (imgs2_np[i] * 255).astype(np.uint8)
                
                # 检测并匹配特征点
                pts1, pts2 = self.detect_and_match(img1, img2)
                
                # 估计相对位姿
                R, t = self.estimate_pose(pts1, pts2)
                
                # 创建齐次变换矩阵
                T = self.create_homogeneous_matrix(R, t)
                
                poses[i] = T
            except Exception as e:
                # print(f"处理第 {i} 对图像时出错: {str(e)}")
                # 如果出错，返回单位矩阵
                poses[i] = np.eye(4)
        
        # 将结果转换回PyTorch张量
        return torch.from_numpy(poses[:,:3,:3]).float().to(imgs1.device),torch.from_numpy(poses[:,:3,3]).float().to(imgs1.device)
    