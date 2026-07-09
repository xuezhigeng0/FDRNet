# Copyright (c) CAIRI AI Lab. All rights reserved
import sys
import os
# 请确保这个路径指向你的 OpenSTL 项目根目录
sys.path.append('/home/xuezhi/method/OpenSTL')

import os.path as osp
import warnings
warnings.filterwarnings('ignore')

# --- [STEP 1] 导入 ClearML ---
from clearml import Task, Logger

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           update_config)

if __name__ == '__main__':
    # 解析命令行参数
    args = create_parser().parse_args()
    config = args.__dict__

    # 加载配置文件逻辑
    cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py') \
        if args.config_file is None else args.config_file
        
    if args.overwrite:
        config = update_config(config, load_config(cfg_path),
                               exclude_keys=['method'])
    else:
        loaded_cfg = load_config(cfg_path)
        config = update_config(config, loaded_cfg,
                               exclude_keys=['method', 'val_batch_size',
                                             'drop_path', 'warmup_epoch'])
        default_values = default_parser()
        for attribute in default_values.keys():
            if config[attribute] is None:
                config[attribute] = default_values[attribute]

    # --- [STEP 2] 初始化 ClearML 任务 ---
    # project_name: 网页端显示的项目文件夹
    # task_name: 具体的实验名称
    task = Task.init(
        project_name='OpenSTL_Experiments', 
        task_name=f"{args.method}_{args.dataname}_{args.ex_name if hasattr(args, 'ex_name') else ''}"
    )
    
    # --- [STEP 3] 同步超参数 ---
    # 这样在 Web 端的 "Configuration" 栏目里就能看到所有 args 参数
    task.connect(config)

    print('>'*35 + ' training ' + '<'*35)
    
    # 初始化实验
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    
    # --- [STEP 4] 开始训练 ---
    # OpenSTL 内部如果使用了 TensorBoard，ClearML 会自动提取 Loss 并绘制折线图
    exp.train()

    if rank == 0:
        print('>'*35 + ' testing  ' + '<'*35)
        mse = exp.test()
        
        # --- [STEP 5] (可选) 手动记录最终结果 ---
        if mse is not None:
            Logger.current_logger().report_scalar(
                title="Final Metrics", 
                series="MSE", 
                value=float(mse), 
                iteration=0
            )

    # 显式结束任务
    task.close()