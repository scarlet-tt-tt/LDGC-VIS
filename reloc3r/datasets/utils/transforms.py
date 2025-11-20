# transform utilities adapted from DUSt3R
import torchvision.transforms as tvf
from reloc3r.utils.image import ImgNorm

# define the standard image transforms
ColorJitter = tvf.Compose([tvf.ColorJitter(0.05, 0.05, 0.05, 0.001), ImgNorm])
# ColorJitter = tvf.Compose([tvf.ColorJitter(0.05, 0.05, 0.05, 0.01)])

# from torchvision import transforms as tvf

# 定义图像预处理
resize_and_ColorJitter = tvf.Compose([
    tvf.ColorJitter(0.5, 0.5, 0.5, 0.1),  # 颜色变化
    tvf.Resize((240, 320)),  # 按比例缩放到目标大小
    tvf.Pad((0, 0, 0, 0), fill=0),  # 填充边缘（如果需要填充，可以指定填充颜色）
    ImgNorm  # 图像归一化
])