import torch.nn as nn
import torch
import numpy as np
import random
from typing import Tuple

import cv2


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
    LoFTR_class: Use LoFTR and SuperGlue to calculate relative pose
    input:
        img0, img1: Image pair, shape (batch, 3, 320, 240)
        K: Camera intrinsic matrix, shape (batch, 4), where each row contains [fx, fy, cx, cy]
    output:
        [R, t]: Relative pose, where R.shape=(batch, 3, 3), t.shape=(batch, 3)
    """
    

    def __init__(self, model_args=None):
        super(LoFTR_class, self).__init__()
        # Initialize SuperPoint and SuperGlue models
        from LoFTR.src.loftr import LoFTR, default_cfg

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'


        from LoFTR.src.loftr.loftr import LoFTR

        # Load default configuration and initialize LoFTR
        self.config = default_cfg
        self.loftr = LoFTR(self.config).to(self.device)
        self.loftr.load_state_dict(torch.load("/home/code/reloc3r/reloc3r-main/LoFTR/weights/indoor_ds.ckpt")['state_dict'])


    def forward(self, img0, img1, K):
        batch_size = img0.shape[0]

        # Convert images to grayscale
        img0_gray = torch.mean(img0, dim=1, keepdim=True)  # Convert to grayscale
        img1_gray = torch.mean(img1, dim=1, keepdim=True)

        R_batch = []
        t_batch = []

        for i in range(batch_size):
            # Process each image pair individually
            pred = {'image0': img0_gray[i:i+1].to(self.device), 
                                'image1': img1_gray[i:i+1].to(self.device)}
            
            self.loftr(pred)
            # print(pred)
            # Extract matching points
            mkpts0 = pred['mkpts0_f'].cpu().numpy()  # Coordinates of matching points in img0, shape (N, 2)
            mkpts1 = pred['mkpts1_f'].cpu().numpy()  # Coordinates of matching points in img1, shape (N, 2)

            # Construct intrinsic matrix
            fx, fy, cx, cy = K[i].cpu().numpy()
            K_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0,  0,  1]])

            # Scale intrinsic matrix (assuming image size remains unchanged)
            scales = np.array([1.0, 1.0])
            K_scaled = scale_intrinsics(K_matrix, scales)

            # Pose estimation
            thresh = 1.0  # Pixel threshold
            ret = estimate_pose(mkpts0, mkpts1, K_scaled, K_scaled, thresh)
            if ret is None:
                R_batch.append(torch.eye(3).to(self.device))
                t_batch.append(torch.zeros(3).to(self.device))
            else:
                R, t, inliers = ret
                R_batch.append(torch.tensor(R, dtype=torch.float32).to(self.device))
                t_batch.append(torch.tensor(t, dtype=torch.float32).to(self.device))

        # Stack results
        R_batch = torch.stack(R_batch, dim=0)
        t_batch = torch.stack(t_batch, dim=0)
        # print(R_batch.shape,t_batch.shape)

        return [R_batch, t_batch]
    
class SPSG(nn.Module):
    """
    SPSG: Use SuperPoint and SuperGlue to calculate relative pose
    input:
        img0, img1: Image pair, shape (batch, 3, 320, 240)
        K: Camera intrinsic matrix, shape (batch, 4), where each row contains [fx, fy, cx, cy]
    output:
        [R, t]: Relative pose, where R.shape=(batch, 3, 3), t.shape=(batch, 3)
    """
    def __init__(self, model_args=None):
        super(SPSG, self).__init__()
        from SpSg.models.matching import Matching

        # Initialize SuperPoint and SuperGlue models
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

        # Convert images to grayscale
        img0_gray = torch.mean(img0, dim=1, keepdim=True)  # Convert to grayscale
        img1_gray = torch.mean(img1, dim=1, keepdim=True)

        R_batch = []
        t_batch = []

        for i in range(batch_size):
            # Process each image pair individually
            pred = self.matching({'image0': img0_gray[i:i+1].to(self.device), 
                                  'image1': img1_gray[i:i+1].to(self.device)})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']

            # Filter valid matches
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]

            # Construct intrinsic matrix
            fx, fy, cx, cy = K[i].cpu().numpy()
            K_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0,  0,  1]])

            # Scale intrinsic matrix (assuming image size remains unchanged)
            scales = np.array([1.0, 1.0])
            K_scaled = scale_intrinsics(K_matrix, scales)

            # Pose estimation
            thresh = 1.0  # Pixel threshold
            ret = estimate_pose(mkpts0, mkpts1, K_scaled, K_scaled, thresh)

            if ret is None:
                R_batch.append(torch.eye(3).to(self.device))
                t_batch.append(torch.zeros(3).to(self.device))
            else:
                R, t, inliers = ret
                R_batch.append(torch.tensor(R, dtype=torch.float32).to(self.device))
                t_batch.append(torch.tensor(t, dtype=torch.float32).to(self.device))

        # Stack results
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
        Initialize relative pose estimator
        
        Parameters:
            feature_detector: Type of feature detector to use ('SIFT', 'SURF', 'ORB', etc.)
            min_matches: Minimum number of matches required to estimate pose
            focal_length: Camera focal length (pixels)
            principal_point: Camera principal point (cx, cy)
            ransac_threshold: Threshold for RANSAC algorithm
        """
        # Initialize feature detector
        super(Classic, self).__init__()
        if feature_detector == 'SIFT':
            self.detector = cv2.SIFT_create()
        elif feature_detector == 'SURF':
            self.detector = cv2.SURF_create()
        elif feature_detector == 'ORB':
            self.detector = cv2.ORB_create()
        else:
            raise ValueError(f"Unsupported feature detector: {feature_detector}")
        
        # Create feature matcher
        if feature_detector in ['SIFT', 'SURF']:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:  # For binary descriptors like ORB
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        

        
        # Distortion coefficients (assuming no distortion)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    def detect_and_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect features and match them
        
        Parameters:
            img1, img2: Input images (H, W, 3)
        
        Returns:
            pts1, pts2: Coordinates of matched features
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Detect features and compute descriptors
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply ratio test to filter matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # Extract coordinates of matched points
        if len(good_matches) < self.min_matches:
            # print(f"Not enough matches ({len(good_matches)} < {self.min_matches})")
            raise ValueError(f"Not enough matches ({len(good_matches)} < {self.min_matches})")
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        return pts1, pts2
    
    def estimate_pose(self, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate relative pose
        
        Parameters:
            pts1, pts2: Coordinates of matched features
        
        Returns:
            R: Rotation matrix (3x3)
            t: Translation vector (3x1)
        """
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.camera_matrix, 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=self.ransac_threshold
        )
        
        # Recover rotation and translation from essential matrix
        _, R, t, _ = cv2.recoverPose(
            E, pts1, pts2, self.camera_matrix, mask
        )
        
        return R, t
    

    def create_homogeneous_matrix(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Create homogeneous transformation matrix
        
        Parameters:
            R: Rotation matrix (3x3)
            t: Translation vector (3x1)
        
        Returns:
            T: Homogeneous transformation matrix (4x4)
        """
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T
    
    def torch_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to NumPy array and adjust dimensions"""
        # Ensure tensor is on CPU
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # Convert to NumPy array and adjust dimensions (B, C, H, W) -> (B, H, W, C)
        return tensor.permute(0, 2, 3, 1).numpy()
    
    def forward(self, imgs1: torch.Tensor, imgs2: torch.Tensor,K) -> torch.Tensor:
        """
        Process a batch of images
        
        Parameters:
            imgs1, imgs2: Input image batches (B, 3, H, W), torch.Tensor type
        
        Returns:
            poses: Batch of relative pose matrices (B, 4, 4), torch.Tensor type
        """
        batch_size = imgs1.shape[0]
        poses = np.zeros((batch_size, 4, 4))

        K= K[0].cpu().numpy()
        # print(K)

        # Camera intrinsic matrix
        self.camera_matrix = np.array([
            [K[0], 0, K[2]],
            [0, K[1], K[3]],
            [0, 0, 1]
        ], dtype=np.float32)
        # Convert PyTorch tensor to NumPy array
        imgs1_np = self.torch_to_numpy(imgs1)
        imgs2_np = self.torch_to_numpy(imgs2)
        
        for i in range(batch_size):
            try:
                # Convert to uint8 type
                img1 = (imgs1_np[i] * 255).astype(np.uint8)
                img2 = (imgs2_np[i] * 255).astype(np.uint8)
                
                # Detect and match features
                pts1, pts2 = self.detect_and_match(img1, img2)
                
                # Estimate relative pose
                R, t = self.estimate_pose(pts1, pts2)
                
                # Create homogeneous transformation matrix
                T = self.create_homogeneous_matrix(R, t)
                
                poses[i] = T
            except Exception as e:
                # print(f"Error processing image pair {i}: {str(e)}")
                # If an error occurs, return identity matrix
                poses[i] = np.eye(4)
        
        # Convert results back to PyTorch tensor
        return torch.from_numpy(poses[:,:3,:3]).float().to(imgs1.device),torch.from_numpy(poses[:,:3,3]).float().to(imgs1.device)
    
