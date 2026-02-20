""" can do with bone soft split exps """
import csv
from datetime import datetime
import logging
import os
import random
import shutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# add the parent directory for importing modules

import numpy as np
import torch
import torch.utils.data


import importlib

# 定义模块名称常量
MODULE_NAME = "HeatmapBasis"  # 切换时改为 "Unet" 即可

# 动态导入模块
opt_module = importlib.import_module(f"{MODULE_NAME}.opt")
trainer_module = importlib.import_module(f"{MODULE_NAME}.trainer_only")
utils_module = importlib.import_module(f"{MODULE_NAME}.utils")
builder_module = importlib.import_module(f"{MODULE_NAME}.builder")

# 分配变量
opt = opt_module.opt
cfg = opt_module.cfg
logger = opt_module.logger
train = trainer_module.train
validate = trainer_module.validate
NullWriter = utils_module.NullWriter
get_coord = utils_module.get_coord
builder = builder_module


#from Unet.opt import opt, cfg, logger 
#from Unet.trainer import train, validate
#from Unet.utils import NullWriter, get_coord
#from Unet import builder


def setup_seed(seed):
    '''
    设置 Python 内置的随机数生成器种子。
    设置 NumPy 和 PyTorch 的随机数种子。
    确保使用固定的计算图，避免 GPU 上的不确定性。
    提高实验的可复现性。
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    '''
    如果设置了随机种子，调用 setup_seed。
    判断分布式训练的启动方式：
        如果是 slurm 调度系统，直接调用 main_worker。
        否则，利用 torch.multiprocessing 的 spawn 启动多个进程，每个 GPU 一个训练进程。
    '''
    if opt.seed is not None:
        setup_seed(opt.seed)

    #best_mre, best_sd = main_worker(opt.gpu_id, opt, cfg)
    best_mre, best_sd = main_worker(None, opt, cfg)

    return best_mre, best_sd


def main_worker(gpu, opt, cfg, log_file=None):
    '''
    初始化分布式训练环境。
    设置日志和日志文件保存路径。
    初始化模型、优化器、学习率调度器和数据加载器。
    执行训练和验证。

    args:
        gpu：当前分配的 GPU 编号。
    opt：命令行参数。
    cfg：配置文件对象。
    '''
    if opt.seed is not None:
        setup_seed(opt.seed)

    if gpu is not None:
        opt.gpu = gpu
    else:
        # 手动设置为 GPU 0（如果你只有一个 GPU）
        opt.gpu = 0

    # new for param_searh
    if opt.log:
        cfg_file_name = os.path.basename(opt.cfg).split('.')[0]
        log_dir = f'./exp/{opt.exp_id}-{cfg_file_name}'
        os.makedirs(log_dir, exist_ok=True)
        filehandler = logging.FileHandler(log_file or f'{log_dir}/train.log')
        streamhandler = logging.StreamHandler()
        logger.setLevel(logging.INFO)
        logger.handlers = []  # 清空已有handler
        logger.addHandler(filehandler)
        logger.addHandler(streamhandler)
    # old for base experiment
    #if opt.log:
    #    cfg_file_name = os.path.basename(opt.cfg).split('.')[0]
    #    filehandler = logging.FileHandler(
    #        './exp/{}-{}/train.log'.format(opt.exp_id, cfg_file_name))
    #    streamhandler = logging.StreamHandler()
    #    logger.setLevel(logging.INFO)
    #    logger.addHandler(filehandler)
    #    logger.addHandler(streamhandler)
    else:
        null_writer = NullWriter()
        sys.stdout = null_writer

    logger.info('******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')
    now = datetime.now()
    logger.info(f"Start Time is {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # for data_loader
    train_dataset = builder.build_dataset(cfg.DATASET, cfg.DATASET.PRESET, subset='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    val_dataset = builder.build_dataset(cfg.DATASET, cfg.DATASET.PRESET, subset='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)

    # for Model
    m = preset_model(cfg)
    m.cuda(opt.gpu)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #m.cuda(device)

    # for Loss, Optimizer and Scheduler
    criterion = builder.build_loss(cfg.LOSS, cfg.DATASET.PRESET)


    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(m.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(m.parameters(), lr=cfg.TRAIN.LR, momentum=0.95, weight_decay=0.001)

    if cfg.TRAIN.LR_SCHEDULE.TYPE == 'linearLR':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=cfg.TRAIN.LR_SCHEDULE.END_FACTOR, total_iters=(cfg.TRAIN.END_EPOCH - cfg.TRAIN.BEGIN_EPOCH))
    elif cfg.TRAIN.LR_SCHEDULE.TYPE == 'plateauLR':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    elif cfg.TRAIN.LR_SCHEDULE.TYPE == 'cosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.LR_SCHEDULE.T_MAX)
    else:
        raise ValueError('Unsupported LR Schedule.')

    heatmap_to_coord = get_coord(cfg, cfg.DATASET.PRESET.HEATMAP_SIZE)

    
    opt.trainIters = 0
    best_mre, best_sd = 999, 999
    best_mre_mre, best_mre_sd = 999, 999

    # 将模型snapshot保存到指定目录
    if opt.model_dir is not None:
        os.makedirs(opt.model_dir, exist_ok=True)
        save_dir = opt.model_dir
    else:
        save_dir = './exp/{}-{}/model_snapshot'.format(opt.exp_id, cfg.FILE_NAME)
    # 确保 model_snapshot 目录存在，如果已存在，则清空其中的所有文件
    if os.path.exists(save_dir):
        for file_name in os.listdir(save_dir):
            file_path = os.path.join(save_dir, file_name)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 递归删除文件夹

    csv_dir = './exp/{}-{}/history_csv'.format(opt.exp_id, cfg.FILE_NAME) 
    os.makedirs(csv_dir, exist_ok=True)
    # 目标文件路径（训练结束后复制到这里）
    if opt.csv_file == None or opt.csv_file == "None" or opt.csv_file == 'None':
        newest_csv = f"./exp/{opt.exp_id}-{cfg.FILE_NAME}/train_history.csv"
    else:
        newest_csv = opt.csv_file

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 例如：20250319_143022
    if opt.csv_file == None or opt.csv_file == "None" or opt.csv_file == 'None':
        output_file = f"{csv_dir}/loss_metrics_history_{timestamp}.csv"
    else:
        params_seach_csv_dir = os.path.join(csv_dir, 'params_search')
        os.makedirs(params_seach_csv_dir, exist_ok=True)
        output_file = f"{params_seach_csv_dir}/loss_metrics_history_{timestamp}.csv"
    
    # 写入表头（因为是新文件，总是从头开始）
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'mre', 'sd', 'sdr'])  # 表头

    ##################################################################################################

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')

        # Training
        loss = train(opt, cfg, train_loader, m, criterion, optimizer)
        logger.epochInfo('Train', opt.epoch, loss)
        lr_scheduler.step()

        # 模拟 metrics（每隔 log_interval 个 epoch 计算一次）
        mre, sd, sdr = None, None, None

        if (i + 1) % opt.snapshot == 0:
            # Save checkpoint if opt.log: #
            
            #save_path = os.path.join(save_dir, 'model_{}.pth'.format(opt.epoch))

            # 确保目录存在（防止 model_snapshot 目录被删除后 torch.save 失败）
            os.makedirs(save_dir, exist_ok=True)

            # 保存模型
            #if opt.model_dir is None:
            #    # if for param_search, not save
            #    torch.save(m.state_dict(), save_path)
            
            # validation
            with torch.no_grad():
                mre, sd, sdr = validate(opt, cfg, val_loader, m, heatmap_to_coord)
                #simple_sdr = {key: value.item() for key, value in sdr.items()}
                logger.info (f"############# Validation Result Epoch {opt.epoch} #############')")
                logger.info(f'MRE:\t\t{mre:.4f}mm, \n\rSD:\t\t{sd:.4f}mm\n\r')
                for radius, rate in sdr.items():
                    logger.info(f'SDR ({radius}mm):\t{rate:.4f}%')


                if best_mre > mre and best_sd > sd:
                    if opt.model_dir is not None:
                        torch.save(m.state_dict(), f"{opt.model_dir}/best.pth") # for param_search
                    else:
                        torch.save(m.state_dict(),
                               './exp/{}-{}/best.pth'.format(opt.exp_id, cfg.FILE_NAME)) # here no distributive learning is used, just m.state_dict()
                    best_mre = mre
                    best_sd = sd
                    logger.info(f'best mean: {best_mre} | best sd: {best_sd} #####')
                elif best_mre > mre:
                    if opt.model_dir is not None:
                        torch.save(m.state_dict(), f"{opt.model_dir}/best_mre.pth") # for param_search
                    else:
                        torch.save(m.state_dict(),
                               './exp/{}-{}/best_mre.pth'.format(opt.exp_id, cfg.FILE_NAME)) # here no distributive learning is used, just m.state_dict()
                    best_mre = mre
                    best_sd = sd
                logger.info(f'best mean: {best_mre} | best sd: {best_sd} #####')
                
            logger.info(f'final mean: {mre} | final sd: {sd} #####')

        # 记录到 CSV,如果
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i+1, loss, mre, sd, sdr])

    # 最新的csv到根目录下
    shutil.copy2(output_file, newest_csv)
        

    if opt.model_dir is not None:
        torch.save(m.state_dict(), f"{opt.model_dir}/final.pth") # for param_search
    else:
        torch.save(m.state_dict(), './exp/{}-{}/final.pth'.format(opt.exp_id, cfg.FILE_NAME))

    ##################################################################################################

    logger.info('******************************')
    end_time = datetime.now()
    #logger.info(f"End Time is {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_diff = end_time - now
    # 显示小时:分钟:秒格式
    hours, remainder = divmod(time_diff.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"- Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    #logger.info(f"Total execution time: {time_diff.total_seconds()} seconds")

    # 返回最佳指标
    return best_mre, best_sd

def preset_model(cfg):
    model = builder.build_model(cfg.MODEL,preset_cfg=cfg.DATASET.PRESET)
        #3,cfg.DATASET.PRESET.NUM_JOINTS, bilinear=False, **cfg)

    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD)
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info('Create new model')
        logger.info('=> init weights')
        model._initialize()

    return model

def parse_output(output):
    # parse the log output to find the best mre and sd
    best_mre, best_sd = 999, 999
    for line in output.splitlines():
        if "best mean:" in line: # 原代码写的是best mean
            parts = line.split('|')
            # 从"best mre: 0.123"中提取0.123赋给best_mre
            best_mre = min(float(parts[0].split(':')[-1].strip()), best_mre)
            #从"best sd: 0.123 ###"中提取0.123赋给best_sd
            best_sd = min(float(parts[1].split(':')[-1].strip().replace('#', '')), best_sd)
    return best_mre, best_sd

if __name__ == "__main__":

    best_mre, best_sd = main()
    #now = datetime.now()
    #logger.info(f"Start Time is {now.strftime('%Y-%m-%d %H:%M:%S')}")

    #best_mre, best_sd = 1, 2
    #output = f"best mean: {best_mre} | best sd: {best_sd} #####"
    #print(output)
