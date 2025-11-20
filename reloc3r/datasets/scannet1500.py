import numpy as np
from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2, cv2
# from pdb import set_trace as bb
from imageio import imread
from cv2 import resize, INTER_NEAREST
import os
import re

DATA_ROOT = './data/scannet1500' 


def label_to_str(label):
    return '_'.join(label)


class ScanNet1500(BaseStereoViewDataset):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_root = DATA_ROOT
        self.pairs_path = '{}/test.npz'.format(self.data_root)
        self.subfolder_mask = 'scannet_test_1500/scene{:04d}_{:02d}'

        with np.load(self.pairs_path) as data:
            self.pair_names = data['name']

        
    def __len__(self):
        return len(self.pair_names)

    def sort_files_by_number(self,scene_path):
        """
        按文件名中的数字编号排序文件
        支持处理两位数和三位数的编号
        """
        # 检查文件夹是否存在
        if not os.path.exists(scene_path):
            print(f"错误：文件夹 {scene_path} 不存在")
            return []
            
        # 获取所有文件名
        all_files = os.listdir(scene_path)
        if not all_files:
            print(f"警告：文件夹 {scene_path} 为空")
            return []
            
        # 定义提取数字编号的函数
        def extract_number(filename):
            # 使用正则表达式匹配数字编号
            match = re.search(r'\d+', filename)
            if match:
                return int(match.group())
            return 0  # 没有数字时返回0
            
        # 按提取的数字编号排序
        sorted_files = sorted(all_files, key=extract_number)
        return sorted_files
        
    def _get_views(self, idx, resolution, rng):
        scene_name, scene_sub_name, name1, _ = self.pair_names[idx]

        views = []
        scene_path = '{}/{}/color'.format(self.data_root, self.subfolder_mask).format(scene_name, scene_sub_name)
        all_files = self.sort_files_by_number(scene_path)
        # print(all_files)

        name1 = str(name1)+'.jpg'
        index = all_files.index(name1)
        
        # 检查是否存在后一位文件
        if index + 1 < len(all_files):
            next_file = all_files[index + 1]
            # print(f"找到后一位文件: {next_file}")
            name2 = next_file

        # print(self.data_root, self.subfolder_mask,name1,name2)
        for name in [name1, name2]: 

            self.image_shape=(640,480)

            color_path = '{}/{}/color/{}'.format(self.data_root, self.subfolder_mask, name).format(scene_name, scene_sub_name)
            image = imread_cv2(color_path)  
            image = cv2.resize(image, (640, 480))
            # image = imread(color_path)[:, :, :3]
            # image = resize(image , self.image_shape,interpolation=INTER_NEAREST)
            # image = image.astype(np.float32)/255.0

            depth_path = '{}/{}/depth/{}'.format(self.data_root, self.subfolder_mask, name).format(scene_name, scene_sub_name)
            depth_path = depth_path.replace('.jpg', '.png')
            depth = imread(depth_path).astype(np.float32)
            depth = resize(depth ,resolution,interpolation=INTER_NEAREST)/1e3
            depth = np.clip(depth, a_min=0.2, a_max=5.0) 
            depth = depth[np.newaxis, :]

            intrinsics_path = '{}/{}/intrinsic/intrinsic_depth.txt'.format(self.data_root, self.subfolder_mask).format(scene_name, scene_sub_name)
            intrinsics = np.loadtxt(intrinsics_path).astype(np.float32)[0:3,0:3]

            pose_path = '{}/{}/pose/{}.txt'.format(self.data_root, self.subfolder_mask, name[:-4]).format(scene_name, scene_sub_name)
            camera_pose = np.loadtxt(pose_path).astype(np.float32)

            # print('color_image',image.shape,image.max(),image.min())
            # print('depth_image',depth.shape,depth.max(),depth.min())


            # print('color_image',color_path)
            image, intrinsics = self._crop_resize_if_necessary(image, 
                                                                     intrinsics, 
                                                                     resolution, 
                                                                     rng=rng)
            
            view_idx_splits = color_path.split('/')

            views.append(dict(
                img = image,
                depth = depth,
                camera_intrinsics = intrinsics,
                camera_pose = camera_pose,
                dataset = 'ScanNet1500',
                label = label_to_str(view_idx_splits[:-1]),
                instance = view_idx_splits[-1],
                ))
        return views
    
# class ScanNet1500(BaseStereoViewDataset):
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.data_root = DATA_ROOT
#         self.pairs_path = '{}/test.npz'.format(self.data_root)
#         self.subfolder_mask = 'scannet_test_1500/scene{:04d}_{:02d}'
#         with np.load(self.pairs_path) as data:
#             self.pair_names = data['name']

        
#     def __len__(self):
#         return len(self.pair_names)

#     def _get_views(self, idx, resolution, rng):
#         scene_name, scene_sub_name, name1, name2 = self.pair_names[idx]

#         views = []

#         for name in [name1, name2]: 

#             self.image_shape=(640,480)
            
#             color_path = '{}/{}/color/{}.jpg'.format(self.data_root, self.subfolder_mask, name).format(scene_name, scene_sub_name)
#             image = imread_cv2(color_path)  
#             image = cv2.resize(image, (640, 480))
#             # image = imread(color_path)[:, :, :3]
#             # image = resize(image , self.image_shape,interpolation=INTER_NEAREST)
#             # image = image.astype(np.float32)/255.0

#             depth_path = '{}/{}/depth/{}.png'.format(self.data_root, self.subfolder_mask, name).format(scene_name, scene_sub_name)
#             # depth = imread_cv2(depth_path)  
#             # depth = cv2.resize(depth, (640, 480))
#             depth = imread(depth_path).astype(np.float32)
#             depth = resize(depth ,resolution,interpolation=INTER_NEAREST)/1e3
#             depth = np.clip(depth, a_min=0.2, a_max=5.0) 
#             depth = depth[np.newaxis, :]

#             intrinsics_path = '{}/{}/intrinsic/intrinsic_depth.txt'.format(self.data_root, self.subfolder_mask).format(scene_name, scene_sub_name)
#             intrinsics = np.loadtxt(intrinsics_path).astype(np.float32)[0:3,0:3]

#             pose_path = '{}/{}/pose/{}.txt'.format(self.data_root, self.subfolder_mask, name).format(scene_name, scene_sub_name)
#             camera_pose = np.loadtxt(pose_path).astype(np.float32)

#             # print('color_image',image.shape,image.max(),image.min())
#             # print('depth_image',depth.shape,depth.max(),depth.min())


#             # print('color_image',color_path)
#             image, intrinsics = self._crop_resize_if_necessary(image, 
#                                                                      intrinsics, 
#                                                                      resolution, 
#                                                                      rng=rng)
            
#             view_idx_splits = color_path.split('/')

#             views.append(dict(
#                 img = image,
#                 depth = depth,
#                 camera_intrinsics = intrinsics,
#                 camera_pose = camera_pose,
#                 dataset = 'ScanNet1500',
#                 label = label_to_str(view_idx_splits[:-1]),
#                 instance = view_idx_splits[-1],
#                 ))
#         return views

