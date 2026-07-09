import sys
sys.path.append('/home/jiabing/methods/OpenSTL')
import argparse
import os
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib as mpl  

from vis_module import (show_video_gif_multiple, show_video_gif_single, show_video_line,
                           show_taxibj, show_weather_bench)
dataset_parameters = {
    'bair': {
        'in_shape': [4, 3, 64, 64],
        'pre_seq_length': 4,
        'aft_seq_length': 12,
        'total_length': 16,
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    },
    'mfmnist': {
        'in_shape': [10, 1, 64, 64],
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'data_name': 'fmnist',
        'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
    'mmnist': {
        'in_shape': [10, 1, 64, 64],
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'data_name': 'mnist',
        'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
    'mmnist_cifar': {
        'in_shape': [10, 3, 64, 64],
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'data_name': 'mnist_cifar',
        'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
    'noisymmnist': {
        'in_shape': [10, 1, 64, 64],
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'data_name': 'noisymmnist',
        'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
    'taxibj': {
        'in_shape': [4, 2, 32, 32],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'metrics': ['mse', 'mae', 'ssim', 'psnr'],
    },
    'human': {
        'in_shape': [4, 3, 256, 256],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    },
    **dict.fromkeys(['kth20', 'kth'], {
        'in_shape': [10, 1, 128, 128],
        'pre_seq_length': 10,
        'aft_seq_length': 20,
        'total_length': 30,
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    }),
    'kth40': {
        'in_shape': [10, 1, 128, 128],
        'pre_seq_length': 10,
        'aft_seq_length': 40,
        'total_length': 50,
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    },
    'kitticaltech': {
        'in_shape': [10, 3, 128, 160],
        'pre_seq_length': 10,
        'aft_seq_length': 1,
        'total_length': 11,
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    },
    **dict.fromkeys(['kinetics400', 'kinetics'], {
        'in_shape': [4, 3, 256, 256],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'data_name': 'kinetics400',
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    }),
    'kinetics600': {
        'in_shape': [4, 3, 256, 256],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'data_name': 'kinetics600',
        'metrics': ['mse', 'mae', 'ssim', 'psnr', 'lpips'],
    },
    **dict.fromkeys(['weather', 'weather_t2m_5_625'], {  # 2m_temperature
        'in_shape': [12, 1, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 't2m',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    }),
    'weather_mv_4_28_s6_5_625': {  # multi-variant weather bench, 4->28 (7 days)
        'in_shape': [4, 12, 32, 64],
        'pre_seq_length': 4,
        'aft_seq_length': 28,
        'total_length': 32,
        'data_name': 'mv',
        'train_time': ['1979', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'idx_in': [1+i*6 for i in range(-3, 0)] + [0,],
        'idx_out': [i*6 + 1 for i in range(28)],
        'step': 6,
        'levels': [150, 500, 850],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_mv_4_4_s6_5_625': {  # multi-variant weather bench, 4->4 (1 day)
        'in_shape': [4, 12, 32, 64],
        'pre_seq_length': 4,
        'aft_seq_length': 4,
        'total_length': 8,
        'data_name': 'mv',
        'train_time': ['1979', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'idx_in': [1+i*6 for i in range(-3, 0)] + [0,],
        'idx_out': [i*6 + 1 for i in range(4)],
        'step': 6,
        'levels': [150, 500, 850],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_r_5_625': {  # relative_humidity
        'in_shape': [12, 1, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'r',
        'levels': [1000,],
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_uv10_5_625': {  # u10+v10, component_of_wind
        'in_shape': [12, 2, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'uv10',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_tcc_5_625': {  # total_cloud_cover
        'in_shape': [12, 1, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'tcc',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_tp_5_625': {  # total_cloud_cover
        'in_shape': [12, 1, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'tp',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
      'weather_tp_2_8125': {  # total_cloud_cover
        'in_shape': [12, 1, 32, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'tp',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_t2m_1_40625': {  # relative_humidity
        'in_shape': [12, 1, 128, 256],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 't2m',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_tcc_2_8125': {  # relative_humidity
        'in_shape': [12, 1, 64, 128],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'tcc',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_r_1_40625': {  # relative_humidity
        'in_shape': [12, 1, 128, 256],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'r',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_uv10_1_40625': {  # u10+v10, component_of_wind
        'in_shape': [12, 2, 128, 256],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'uv10',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'weather_tcc_1_40625': {  # total_cloud_cover
        'in_shape': [12, 1, 128, 256],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'tcc',
        'train_time': ['2010', '2015'], 'val_time': ['2016', '2016'], 'test_time': ['2017', '2018'],
        'metrics': ['mse', 'rmse', 'mae'],
    },
    'sevir_vis':{
        'in_shape': [13, 1, 768, 768],
        'pre_seq_length': 13,
        'aft_seq_length': 12,
        'total_length': 25,
        'data_name': 'vis', 
        'metrics': ['mse', 'mae', 'pod', 'sucr', 'csi', 'lpips'],
    },
    'sevir_ir069':{
        'in_shape': [13, 1, 192, 192],
        'pre_seq_length': 13,
        'aft_seq_length': 12,
        'total_length': 25,
        'data_name': 'ir069',
        'metrics': ['mse', 'mae', 'pod', 'sucr', 'csi', 'lpips'],
    },
    'sevir_ir107':{
        'in_shape': [13, 1, 192, 192],
        'pre_seq_length': 13,
        'aft_seq_length': 12,
        'total_length': 25,
        'data_name': 'ir107',
        'metrics': ['mse', 'mae', 'pod', 'sucr', 'csi', 'lpips'],
    },
    'sevir_vil':{
        'in_shape': [13, 1, 384, 384],
        'pre_seq_length': 13,
        'aft_seq_length': 12,
        'total_lenght': 25,
        'data_name': 'vil', 
        'metrics': ['mse', 'mae', 'pod', 'sucr', 'csi', 'lpips'],
    },
}

# 在文件开头创建自定义 colormap
final_colors = ['#F0F8FF', '#F0FFFF', '#E0FFFF', '#B0E0E6', '#5F9EA0', '#4682B4', '#4169E1']
custom_final_cmap = mcolors.LinearSegmentedColormap.from_list('custom_final', final_colors)
mpl.colormaps.register(name='custom_final', cmap=custom_final_cmap)


def min_max_norm(data):
    _min, _max = np.min(data), np.max(data)
    data = (data - _min) / (_max - _min)
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualization of a STL model')

    parser.add_argument('--dataname', '-d', default=None, type=str,
                        help='The name of dataset (default: "mmnist")')
    parser.add_argument('--index', '-i', default=1, type=int, help='The index of a video sequence to show')
    parser.add_argument('--work_dirs', '-w', default=None, type=str,
                        help='Path to the work_dir or the path to a set of work_dirs')
    parser.add_argument('--vis_dirs', '-v', action='store_true', default=False,
                        help='Whether to visualize a set of work_dirs')
    parser.add_argument('--reload_input', action='store_true', default=False,
                        help='Whether to reload the input and true for each method')
    parser.add_argument('--save_dirs', '-s', default='vis_figures', type=str,
                        help='The path to save visualization results')
    parser.add_argument('--vis_channel', '-vc', default=-1, type=int,
                        help='Select a channel to visualize as the heatmap')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.dataname is not None and args.work_dirs is not None, \
        'The name of dataset and the path to work_dirs are required'

    # setup results of the STL methods
    base_dir = args.work_dirs
    assert os.path.isdir(args.work_dirs)
    if args.vis_dirs:
        method_list = os.listdir(args.work_dirs)
    else:
        method_list = [args.work_dirs.split('/')[-1]]
        base_dir = base_dir.split(method_list[0])[0]

    use_rgb = False if args.dataname in ['mfmnist', 'mmnist', 'kth20', 'kth', 'kth40'] else True
    config = args.__dict__
    config.update(dataset_parameters[args.dataname])
    idx, ncols = args.index, config['aft_seq_length']
    if not os.path.isdir(args.save_dirs):
        os.mkdir(args.save_dirs)
    if args.vis_channel != -1:  # choose a channel
        c_surfix = f"_C{args.vis_channel}"
        assert 0 <= args.vis_channel <= config['in_shape'][1], 'Channel index out of range'
    else:
        c_surfix = ""
        assert args.dataname not in ['taxibj', 'weather_uv10_5_625'], 'Please select a channel'

    # loading results
    predicts_dict, inputs_dict, trues_dict = dict(), dict(), dict()
    empty_keys = list()
    for method in method_list:
        try:
            predicts_dict[method] = np.load(os.path.join(base_dir, method, 'saved/preds.npy'))
            if 'weather' in args.dataname:
                predicts_dict[method] = np.clip(predicts_dict[method], 0, 1)
        except:
            empty_keys.append(method)
            print('Failed to read the results of', method)
    assert len(predicts_dict.keys()) >= 1, 'The results should not be empty'
    for k in empty_keys:
        method_list.pop(method_list.index(k))

    inputs = np.load(os.path.join(base_dir, method_list[0], 'saved/inputs.npy'))
    trues = np.load(os.path.join(base_dir, method_list[0], 'saved/trues.npy'))
    for idx in range(0, 17408, 100):
        for method in method_list:
            inputs = np.load(os.path.join(base_dir, method_list[0], 'saved/inputs.npy'))
            trues = np.load(os.path.join(base_dir, method_list[0], 'saved/trues.npy'))
            if 'weather' in args.dataname:
                inputs = np.clip(inputs, 0, 1)
                trues = np.clip(trues, 0, 1)
                inputs = show_weather_bench(inputs[idx, 0:ncols, ...], src_img=None, cmap='custom_final')
                inputs = inputs.transpose(0, 3, 1, 2)
                trues = show_weather_bench(trues[idx, 0:ncols, ...], src_img=None, cmap='custom_final')
                trues = trues.transpose(0, 3, 1, 2)
            elif 'taxibj' in args.dataname:
                inputs = show_taxibj(inputs[idx, 0:ncols, ...], cmap='viridis').transpose(0, 3, 1, 2)
                trues = show_taxibj(trues[idx, 0:ncols, ...], cmap='viridis').transpose(0, 3, 1, 2)
            else:
                inputs, trues = inputs[idx], trues[idx]
            if not args.reload_input:  # load the input and true for each method
                break
            else:
                inputs_dict[method], trues_dict[method] = inputs, trues

        # plot gifs and figures of the STL methods
        for i, method in enumerate(method_list):
            print(method, predicts_dict[method][idx].shape)
            if args.reload_input:
                inputs, trues = inputs_dict[method], trues_dict[method]
            if 'weather' in args.dataname:
                preds = show_weather_bench(predicts_dict[method][idx, 0:ncols, ...],
                                        src_img=None, cmap='custom_final')
                preds = preds.transpose(0, 3, 1, 2)
            elif 'taxibj' in args.dataname:
                preds = show_taxibj(predicts_dict[method][idx, 0:ncols, ...],
                                    cmap='viridis', vis_channel=args.vis_channel)
                preds = preds.transpose(0, 3, 1, 2)
            else:
                preds = predicts_dict[method][idx]

            if i == 0:
                show_video_line(inputs.copy(), ncols=config['pre_seq_length'], vmax=0.6, cbar=False,
                    out_path='{}/{}_input{}'.format(args.save_dirs, args.dataname+c_surfix, str(idx)+'.png'),
                    format='png', use_rgb=use_rgb)
                show_video_line(trues.copy(), ncols=config['aft_seq_length'], vmax=0.6, cbar=False,
                    out_path='{}/{}_true{}'.format(args.save_dirs, args.dataname+c_surfix, str(idx)+'.png'),
                    format='png', use_rgb=use_rgb)
                show_video_gif_single(inputs.copy(), use_rgb=use_rgb,
                    out_path='{}/{}_{}_{}_input'.format(args.save_dirs, args.dataname+c_surfix, method, idx))
                show_video_gif_single(trues.copy(), use_rgb=use_rgb,
                    out_path='{}/{}_{}_{}_true'.format(args.save_dirs, args.dataname+c_surfix, method, idx))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
            show_video_line(preds, ncols=ncols, vmax=0.6, cbar=False,
                            out_path='{}/{}_{}_{}'.format(args.save_dirs, args.dataname+c_surfix, method, str(idx)+'.png'),
                            format='png', use_rgb=use_rgb)
            show_video_gif_multiple(inputs, trues, preds, use_rgb=use_rgb,
                                    out_path='{}/{}_{}_{}'.format(args.save_dirs, args.dataname+c_surfix, method, idx))
            show_video_gif_single(preds, use_rgb=use_rgb,
                                out_path='{}/{}_{}_{}_pred'.format(args.save_dirs, args.dataname+c_surfix, method, idx))


if __name__ == '__main__':
    main()
#python3 visualize.py -d weather_tp_5_625 -i 0 --work_dirs /home/jiabing/methods/multi-5/results/B -s results/B/vis