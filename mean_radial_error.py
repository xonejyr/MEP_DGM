import numpy as np
from ..utils import scale_to_targetsize  # 导入 scale_to_original

def mean_radial_error(predictions, ground_truth, target_size, spacing=0.1):
    if np.min(predictions) >= -0.5 and np.max(predictions) <= 0.5:
        # 将coords从[-0.5,0.5]映射到target_size
        predictions = scale_to_targetsize(predictions, target_size)
        ground_truth = scale_to_targetsize(ground_truth, target_size)
    # 计算径向距离并返回均值和标准差
    distances = np.sqrt(np.sum((predictions - ground_truth) ** 2, axis=-1))
    mre = distances.mean().item() * spacing
    sd = distances.std().item() * spacing
    return mre, sd