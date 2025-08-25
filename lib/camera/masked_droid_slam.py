import sys
sys.path.insert(0, 'thirdparty/DROID-SLAM/droid_slam')
sys.path.insert(0, 'thirdparty/DROID-SLAM')
# 添加ZoeDepth到Python路径，解决内部模块导入问题
sys.path.insert(0, 'thirdparty/ZoeDepth')

from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image
from glob import glob
from torchvision.transforms import Resize
import os
import time
from droid import Droid
from .slam_utils import slam_args, parser
from .slam_utils import get_dimention, est_calib, image_stream, preprocess_masks
from .est_scale import est_scale_hybrid
from ..utils.rotation_conversions import quaternion_to_matrix
from thirdparty.ZoeDepth.zoedepth.models.builder import build_model_from_pretrained
from thirdparty.ZoeDepth.zoedepth.utils.config import get_config
torch.multiprocessing.set_start_method('spawn')
from pathlib import Path
# 修复导入：使用正确的ZoeDepth类导入
from thirdparty.ZoeDepth.zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth

def run_metric_slam(img_folder, masks=None, calib=None, is_static=False):
    '''
    Input:
        img_folder: directory that contain image files 
        masks: list or array of 2D masks for human. 
               If None, no masking applied during slam.
        calib: camera intrinsics [fx, fy, cx, cy]. 
               If None, will be naively estimated.
    '''

    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

    ##### If static camera, simply return static camera motion #####
    if is_static:
        pred_cam_t = torch.zeros([len(imgfiles), 3])
        pred_cam_r = torch.eye(3).expand(len(imgfiles), 3, 3)

        return pred_cam_r, pred_cam_t

    ##### Masked droid slam #####
    droid, traj = run_slam(img_folder, masks=masks, calib=calib)
    n = droid.video.counter.value
    tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
    disps = droid.video.disps_up.cpu().numpy()[:n]
    del droid
    torch.cuda.empty_cache()

    ##### Estimate Metric Depth #####

    # 确保GPU内存清理，防止第二次运行冲突
    torch.cuda.empty_cache()
    
    # 本地加载ZoeDepth模型
    print("Loading ZoeDepth model...")
    start_time = time.time()
    ckpt_path = "/root/.cache/torch/hub/checkpoints/ZoeD_M12_N.pt"
    config = get_config("zoedepth", "infer", pretrained_resource=ckpt_path)
    model_zoe_n = build_model_from_pretrained(config, ckpt_path)
    end_time = time.time()
    print(f"ZoeDepth模型加载时间: {end_time - start_time:.2f}秒")
    # _ = model_zoe_n.eval()
    model_zoe_n = model_zoe_n.to('cuda')
    print("✅ ZoeDepth模型加载成功")
    # # 检查torch.hub缓存中是否有MiDaS模型
    # hub_dir = torch.hub.get_dir()
    # midas_cache_path = os.path.join(hub_dir, "intel-isl_MiDaS_master")
    
    # if not os.path.exists(midas_cache_path):
    #     print(f"⚠️ MiDaS模型架构缓存不存在: {midas_cache_path}")
    #     print("请先运行以下命令预下载MiDaS架构:")
    #     print("python -c \"import torch; torch.hub.load('intel-isl/MiDaS', 'DPT_BEiT_L_384', pretrained=False)\"")
    #     # 返回默认值，避免程序崩溃
    #     pred_cam_t = torch.zeros([len(imgfiles), 3])
    #     pred_cam_r = torch.eye(3).expand(len(imgfiles), 3, 3)
    #     return pred_cam_r, pred_cam_t
    # else:
    #     print(f"✅ 找到MiDaS架构缓存: {midas_cache_path}")
    
    # # 使用本地预训练的MiDaS模型，避免网络下载  
    # conf = get_config("zoedepth", "infer", 
    #                  use_pretrained_midas=False,  # ✅ 关键修复：不加载MiDaS权重
    #                  force_reload=False)          # 避免重新下载
    # print("配置信息:", {k: v for k, v in conf.items() if 'midas' in k.lower() or 'pretrained' in k.lower()})
    # print("🔧 权重加载策略: ZoeD_M12_N.pt提供完整权重，不单独加载MiDaS")
    
    # model_zoe_n = build_model_from_pretrained(conf, "/root/.cache/torch/hub/checkpoints/ZoeD_M12_N.pt")
    # # model_zoe_n.to('cuda').eval()
    # print("✅ ZoeDepth模型加载成功")

    pred_depths = []
    H, W = get_dimention(img_folder)
    
    try:
        for t in tqdm(tstamp):
            img = cv2.imread(imgfiles[t])[:,:,::-1]
            img = cv2.resize(img, (W, H))
            
            img_pil = Image.fromarray(img)
            pred_depth = model_zoe_n.infer_pil(img_pil)
            pred_depths.append(pred_depth)

        ##### Estimate Metric Scale #####
        scales_ = []
        n = len(tstamp)   # for each keyframe
        for i in tqdm(range(n)):
            t = tstamp[i]
            disp = disps[i]
            pred_depth = pred_depths[i]
            slam_depth = 1/disp
            
            if masks is None:
                msk = None
            else:
                msk = masks[t].numpy()

            scale = est_scale_hybrid(slam_depth, pred_depth, msk=msk)
            scales_.append(scale)
        scale = np.median(scales_)
        
        # convert to metric-scale camera extrinsics: R_wc, T_wc
        pred_cam_t = torch.tensor(traj[:, :3]) * scale
        pred_cam_q = torch.tensor(traj[:, 3:])
        pred_cam_r = quaternion_to_matrix(pred_cam_q[:,[3,0,1,2]])

    except Exception as e:
        print(f"❌ 深度估计或尺度计算失败: {e}")
        # 返回默认值
        pred_cam_t = torch.zeros([len(imgfiles), 3])
        pred_cam_r = torch.eye(3).expand(len(imgfiles), 3, 3)
    
    finally:
        # 无论成功失败都要清理ZoeDepth模型，释放GPU内存
        print("清理ZoeDepth模型...")
        try:
            del model_zoe_n
            del core
        except:
            pass  # 如果对象不存在也没关系
        torch.cuda.empty_cache()
        print("✅ 模型清理完成")

    return pred_cam_r, pred_cam_t


def run_slam(imagedir, masks=None, calib=None, depth=None):
    """ Maksed DROID-SLAM """
    droid = None
    if calib is None:
        calib = est_calib(imagedir)

    if masks is not None:
        img_msks, conf_msks = preprocess_masks(imagedir, masks)

    for (t, image, intrinsics) in tqdm(image_stream(imagedir, calib)):

        if droid is None:
            slam_args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(slam_args)
        
        if masks is not None:
            img_msk = img_msks[t]
            conf_msk = conf_msks[t]
            image = image * (img_msk < 0.5)
            droid.track(t, image, intrinsics=intrinsics, depth=depth, mask=conf_msk)
        else:
            droid.track(t, image, intrinsics=intrinsics, depth=depth, mask=None)  

    traj = droid.terminate(image_stream(imagedir, calib))

    return droid, traj


def search_focal_length(img_folder, masks=None, stride=10, max_frame=50,
                        low=500, high=1500, step=100):
    """ Search for a good focal length by SLAM reprojection error """
    if masks is not None:
        masks = masks[::stride]
        masks = masks[:max_frame]
        img_msks, conf_msks = preprocess_masks(img_folder, masks)
        input_msks = (img_msks, conf_msks)
    else:
        input_msks = None

    # default estimate
    calib = np.array(est_calib(img_folder))
    best_focal = calib[0]
    best_err = test_slam(img_folder, input_msks, 
                         stride=stride, calib=calib, max_frame=max_frame)
    
    # search based on slam reprojection error
    for focal in range(low, high, step):
        calib[:2] = focal
        err = test_slam(img_folder, input_msks, 
                        stride=stride, calib=calib, max_frame=max_frame)

        if err < best_err:
            best_err = err
            best_focal = focal

    return best_focal


def calibrate_intrinsics(img_folder, masks=None, stride=10, max_frame=50,
                        low=500, high=1500, step=100, is_static=False):
    """ Search for a good focal length by SLAM reprojection error """
    if masks is not None:
        masks = masks[::stride]
        masks = masks[:max_frame]
        img_msks, conf_msks = preprocess_masks(img_folder, masks)
        input_msks = (img_msks, conf_msks)
    else:
        input_msks = None

    # User indicate it's static camera
    if is_static:
        calib = np.array(est_calib(img_folder))
        return calib, is_static
    
    # Estimate camera whether static and intrinsics
    else:
        calib = np.array(est_calib(img_folder))
        best_focal = calib[0]
        best_err, is_static = test_slam(img_folder, input_msks, 
                                        stride=stride, calib=calib, max_frame=max_frame)
        if is_static:
            print('The camera is likely static.')
            return calib, is_static
        
        for focal in range(low, high, step):
            calib[:2] = focal
            err, is_static = test_slam(img_folder, input_msks, 
                                        stride=stride, calib=calib, max_frame=max_frame)
            if err < best_err:
                best_err = err
                best_focal = focal
        calib[0] = calib[1] = best_focal

        return calib, is_static


def test_slam(imagedir, masks, calib, stride=10, max_frame=50):
    """ Shorter SLAM step to test reprojection error """
    args = parser.parse_args([])
    args.stereo = False
    args.upsample = False
    args.disable_vis = True
    args.frontend_window = 10
    args.frontend_thresh = 10
    droid = None

    for (t, image, intrinsics) in image_stream(imagedir, calib, stride, max_frame):
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        if masks is not None:
            img_msk = masks[0][t]
            conf_msk = masks[1][t]
            image = image * (img_msk < 0.5)
            droid.track(t, image, intrinsics=intrinsics, mask=conf_msk)  
        else:
            droid.track(t, image, intrinsics=intrinsics, mask=None)  
    
    if droid.video.counter.value <= 1:
        # If less than 2 keyframes, likely static camera
        static_camera = True
        reprojection_error = None
    else:
        static_camera = False
        reprojection_error = droid.compute_error()

    del droid

    return reprojection_error, static_camera


