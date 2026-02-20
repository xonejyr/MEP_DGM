import numpy as np
from ..utils import scale_to_targetsize  # 导入 scale_to_original

def successful_detection_rate(predictions, ground_truth, target_size, radii, spacing=0.1):
    if np.min(predictions) >= -0.5 and np.max(predictions) <= 0.5:
        # 将coords从[-0.5,0.5]映射到target_size
        predictions = scale_to_targetsize(predictions, target_size)
        ground_truth = scale_to_targetsize(ground_truth, target_size)

    distances = np.sqrt(np.sum((predictions - ground_truth) ** 2, axis=-1))
    total_points = distances.size  # 总点数 = batchsize * num_joints
    # numel() 的底层实现实际上是调用张量的 size() 方法（返回形状），然后计算所有维度大小的乘积。
    sdr = {}
    for radius in radii:
        threshold = radius / spacing
        successful_detections = np.sum(distances <= threshold).item()  # 转换为标量
        sdr[radius] = successful_detections / total_points
        sdr[radius] = sdr[radius] * 100 # 转为百分数
    return sdr
