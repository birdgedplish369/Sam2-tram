import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import torch
import argparse
import numpy as np
from glob import glob
from pycocotools import mask as masktool
import time  # 添加时间模块
from datetime import datetime  # 添加日期时间模块
import json  # 添加JSON模块用于日志记录

from lib.pipeline import video2frames, visualize_tram, grounding_sam2_tracking
from lib.camera import run_metric_slam, calibrate_intrinsics, align_cam_to_world
from lib.models import get_hmr_vimo
from lib.vis import npy2bvh
from lib.pipeline_logger import PipelineLogger

def load_tracks_safely(tracks_file):
    """
    安全地加载tracks文件，处理各种数据格式
    
    Args:
        tracks_file: tracks.npy文件路径
    
    Returns:
        dict: 标准化的tracks字典
    """
    try:
        tracks = np.load(tracks_file, allow_pickle=True)
        
        # 如果是0维数组，先转换为item
        if hasattr(tracks, 'ndim') and tracks.ndim == 0:
            tracks = tracks.item()
        
        return tracks
    except Exception as e:
        print(f"Error loading tracks from {tracks_file}: {e}")
        return {}


def normalize_tracks_format(tracks):
    """
    标准化tracks数据格式为字典
    
    Args:
        tracks: 各种格式的tracks数据
    
    Returns:
        dict: 标准化的tracks字典
    """
    if isinstance(tracks, np.ndarray):
        print("Converting tracks from numpy array to dictionary format...")
        tracks_dict = {}
        
        # 处理0维数组的情况
        if tracks.ndim == 0:
            # 0维数组应该包含字典数据，尝试获取
            try:
                tracks_data = tracks.item()
                if isinstance(tracks_data, dict):
                    return tracks_data
                else:
                    print("Warning: tracks 0-d array does not contain dictionary data")
                    return {}
            except:
                print("Warning: Failed to extract data from 0-d array, creating empty dictionary")
                return {}
        else:
            # 处理多维数组
            for i, track in enumerate(tracks):
                if track is not None and len(track) > 0:
                    tracks_dict[i] = track
            return tracks_dict
    elif isinstance(tracks, dict):
        # 如果已经是字典格式，直接使用
        return tracks
    else:
        print(f"Warning: tracks has unexpected type {type(tracks)}, returning empty dict")
        return {}


def main():
    parser = argparse.ArgumentParser(description='Complete TRAM Pipeline: Camera Estimation + Human Pose Estimation + Visualization')
    
    # 输入参数
    parser.add_argument("--video", type=str, default=None, help='input video file')
    parser.add_argument("--image_folder", type=str, default=None, help='path to pre-processed image folder')
    parser.add_argument("--output_folder", type=str, default=None, help='output folder name')
    
    # 相机参数
    parser.add_argument("--visualize_mask", action='store_true', help='save deva vos for visualization')
    
    # 人体参数
    parser.add_argument('--max_humans', type=int, default=1, help='maximum number of multiple humans to reconstruct')
    parser.add_argument('--skip_visualization', action='store_true', help='skip visualization')
    
    # 新的检测方法参数
    parser.add_argument('--keyframe_interval', type=int, default=20, help='keyframe interval for detection')
    parser.add_argument('--person_conf_threshold', type=float, default=0.25, help='person detection confidence threshold')
    
    # 可视化参数
    parser.add_argument('--floor_scale', type=int, default=3, help='size of the floor for visualization')
    parser.add_argument('--bin_size', type=int, default=-1, help='rasterization bin_size for visualization')
    
    args = parser.parse_args()
    
    
    # 记录整个pipeline的开始时间
    pipeline_start_time = time.time()
    
    # 验证输入参数
    if args.video is None and args.image_folder is None:
        print("Error: Either --video or --image_folder must be provided")
        return
    
    if args.video is not None and args.image_folder is not None:
        print("Warning: Both --video and --image_folder provided. Using --image_folder and ignoring --video")
    
    # 确定序列名称和文件夹
    if args.image_folder is not None:
        img_folder = args.image_folder
        if args.output_folder is None:
            seq = os.path.basename(args.image_folder)
        else:
            seq = args.output_folder
        video_path = None
        print(f'Using pre-processed images from: {img_folder}')
    else:
        file = args.video
        video_path = file
        if args.output_folder is None:
            seq = os.path.basename(file).split('.')[0]
        else:
            seq = args.output_folder
        img_folder = f'results/{seq}/images'
        print(f'Processing video: {file}')
    
    seq_folder = f'results/{seq}'
    hps_folder = f'{seq_folder}/hps'
    os.makedirs(seq_folder, exist_ok=True)
    os.makedirs(hps_folder, exist_ok=True)
    
    print(f'Output folder: {seq_folder}')
    
    # 初始化pipeline日志记录器
    logger = PipelineLogger(video_path=video_path, output_folder=seq_folder)
    
    # 步骤1: 视频帧提取（如果需要）
    video_fps = 30.0  # 默认fps值
    if args.video is not None and args.image_folder is None:
        os.makedirs(img_folder, exist_ok=True)
        print('Step 1: Extracting frames ...')
        step1_start = logger.start_step("step1_frame_extraction")
        
        try:
            nframes, video_fps = video2frames(args.video, img_folder)
            logger.complete_step("step1_frame_extraction", step1_start, success=True,
                               extracted_frames=nframes, video_fps=video_fps)
            step1_time = time.time() - step1_start
            print(f'Extracted {nframes} frames in {step1_time:.2f} seconds')
            print(f'Video FPS: {video_fps:.2f}')
        except Exception as e:
            error_msg = f"Frame extraction failed: {str(e)}"
            logger.complete_step("step1_frame_extraction", step1_start, success=False, error_msg=error_msg)
            logger.finalize(success=False, error_msg=error_msg)
            logger.save_to_json(f'{seq_folder}/pipeline_log.json')
            print(f'Error: {error_msg}')
            return
    else:
        step1_time = 0  # 如果跳过视频提取，设置时间为0
        # 记录跳过的步骤
        if args.image_folder is not None:
            logger.complete_step("step1_frame_extraction", 0, success=True,
                               extracted_frames=0, video_fps=0,
                               error="Skipped - using pre-processed images")
        
    # 步骤2: 人体检测、分割和跟踪
    print('Step 2: Detection, Segmentation, and Tracking ...')
    step2_start = logger.start_step("step2_detection_tracking")
    tracks_file = f'{seq_folder}/tracks.npy'
    
    try:
        if os.path.exists(tracks_file):
            print(f'Skipping detection/segmentation/tracking, using existing {tracks_file}')
            tracks_ = np.load(tracks_file, allow_pickle=True).item()
            # boxes_ = np.load(f'{seq_folder}/boxes.npy', allow_pickle=True)
            masks_ = np.load(f'{seq_folder}/masks.npy', allow_pickle=True)
            # 仍然需要获取图像文件列表，因为后续步骤会用到
            imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
            if len(imgfiles) == 0:
                imgfiles = sorted(glob(f'{img_folder}/*.png'))
            if len(imgfiles) == 0:
                error_msg = f'No image files found in {img_folder}'
                print(f'Error: {error_msg}')
                logger.complete_step("step2_detection_tracking", step2_start, success=False, error_msg=error_msg)
                logger.finalize(success=False, error_msg=error_msg)
                logger.save_to_json(f'{seq_folder}/pipeline_log.json')
                return
            print(f'Found {len(imgfiles)} images')
            
            # 计算已存在的tracks数量
            if isinstance(tracks_, dict):
                num_tracks = len(tracks_)
            else:
                num_tracks = len(tracks_) if hasattr(tracks_, '__len__') else 0
                
            logger.complete_step("step2_detection_tracking", step2_start, success=True,
                               detected_tracks=num_tracks, 
                               error="Skipped - using existing files")
        else:
            imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
            if len(imgfiles) == 0:
                imgfiles = sorted(glob(f'{img_folder}/*.png'))
            if len(imgfiles) == 0:
                error_msg = f'No image files found in {img_folder}'
                print(f'Error: {error_msg}')
                logger.complete_step("step2_detection_tracking", step2_start, success=False, error_msg=error_msg)
                logger.finalize(success=False, error_msg=error_msg)
                logger.save_to_json(f'{seq_folder}/pipeline_log.json')
                return
            
            print(f'Found {len(imgfiles)} images')
            
            # 选择检测方法
            print('使用Grounding SAM2进行多人检测分割追踪')
            masks_, tracks_ = grounding_sam2_tracking(
                img_folder, seq_folder, 
                person_conf_threshold=args.person_conf_threshold,
                keyframe_interval=args.keyframe_interval,
                vis=args.visualize_mask,
            )
            np.save(f'{seq_folder}/tracks.npy', tracks_)
            np.save(f'{seq_folder}/masks.npy', masks_)
            
            # 计算检测到的tracks数量
            if isinstance(tracks_, dict):
                num_tracks = len(tracks_)
            else:
                num_tracks = len(tracks_) if hasattr(tracks_, '__len__') else 0
                
            logger.complete_step("step2_detection_tracking", step2_start, success=True,
                               detected_tracks=num_tracks)

        step2_time = time.time() - step2_start
        print(f'Detection, segmentation, and tracking completed in {step2_time:.2f} seconds')
        
    except Exception as e:
        error_msg = f"Detection, segmentation, and tracking failed: {str(e)}"
        print(f'Error: {error_msg}')
        logger.complete_step("step2_detection_tracking", step2_start, success=False, error_msg=error_msg)
        logger.finalize(success=False, error_msg=error_msg)
        logger.save_to_json(f'{seq_folder}/pipeline_log.json')
        return
    
    # 步骤3: 相机参数估计
    print('Step 3: Camera Parameter Estimation ...')
    step3_start = logger.start_step("step3_camera_estimation")
    
    try:
        masks = np.array([masktool.decode(m) for m in masks_])
        masks = torch.from_numpy(masks)
        
        # 内参标定
        calibration_success = True
        try:
            start_time = time.time()
            cam_int, is_static = calibrate_intrinsics(img_folder, masks, is_static=False)
            end_time = time.time()
            print(f'Calibrate intrinsics completed in {end_time - start_time:.2f} seconds')
        except Exception as e:
            calibration_success = False
            raise Exception(f"Camera calibration failed: {str(e)}")
        
        # SLAM
        slam_success = True
        try:
            start_time = time.time()
            cam_R, cam_T = run_metric_slam(img_folder, masks=masks, calib=cam_int, is_static=is_static)
            end_time = time.time()
            print(f'Run metric slam completed in {end_time - start_time:.2f} seconds')
        except Exception as e:
            slam_success = False
            raise Exception(f"SLAM failed: {str(e)}")
        
        # 世界坐标系对齐
        alignment_success = True
        try:
            start_time = time.time()
            wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R, cam_T)
            end_time = time.time()
            print(f'Align cam to world completed in {end_time - start_time:.2f} seconds')
        except Exception as e:
            alignment_success = False
            raise Exception(f"Camera alignment failed: {str(e)}")
        
        camera = {'pred_cam_R': cam_R.numpy(), 'pred_cam_T': cam_T.numpy(), 
                    'world_cam_R': wd_cam_R.numpy(), 'world_cam_T': wd_cam_T.numpy(),
                    'img_focal': cam_int[0], 'img_center': cam_int[2:], 'spec_focal': spec_f}
        
        np.save(f'{seq_folder}/camera.npy', camera)
        
        logger.complete_step("step3_camera_estimation", step3_start, success=True,
                           calibration_successful=calibration_success,
                           slam_successful=slam_success,
                           alignment_successful=alignment_success)
        
        step3_time = time.time() - step3_start
        print(f'Camera estimation completed in {step3_time:.2f} seconds')
        
    except Exception as e:
        error_msg = f"Camera parameter estimation failed: {str(e)}"
        print(f'Error: {error_msg}')
        logger.complete_step("step3_camera_estimation", step3_start, success=False, error_msg=error_msg)
        logger.finalize(success=False, error_msg=error_msg)
        logger.save_to_json(f'{seq_folder}/pipeline_log.json')
        return
    
    # 步骤4: 人体姿态估计
    print('Step 4: Human Pose Estimation ...')
    step4_start = logger.start_step("step4_pose_estimation")
    
    try:
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tracks = tracks_
        
        # 确保tracks是字典格式
        tracks = normalize_tracks_format(tracks)
        
        if not tracks:
            error_msg = "No valid tracks found for human pose estimation"
            print(f"Error: {error_msg}")
            logger.complete_step("step4_pose_estimation", step4_start, success=False, error_msg=error_msg)
            logger.finalize(success=False, error_msg=error_msg)
            logger.save_to_json(f'{seq_folder}/pipeline_log.json')
            return
        
        img_focal = camera['img_focal']
        img_center = camera['img_center']
        
        # 按轨迹长度排序
        tid = [k for k in tracks.keys()]
        lens = [len(trk) for trk in tracks.values()]
        rank = np.argsort(lens)[::-1]
        tracks = [tracks[tid[r]] for r in rank]
        
        # 记录模型加载时间
        model_load_start = time.time()
        print("Loading HMR-VIMO model...")
        model = get_hmr_vimo(checkpoint='data/pretrain/vimo_checkpoint.pth.tar')
        model_load_time = time.time() - model_load_start
        print(f"Model loading time: {model_load_time:.2f} seconds")
        
        # 记录推理总时间
        inference_start_time = time.time()
        total_frames_processed = 0
        successful_humans = 0  # 统计成功处理的人数
        max_humans_to_process = args.max_humans
        
        for k, trk in enumerate(tracks):
            print(f'Processing human {k+1}/{min(len(tracks), max_humans_to_process)}')
            
            valid = np.array([t['det'] for t in trk])
            # boxes = np.concatenate([t['det_box'] for t in trk])
            boxes = np.array([t['seg_box'] for t in trk])
            frame = np.array([t['frame'] for t in trk])
            
            start_time = time.time()
            results = model.inference(imgfiles, boxes, valid=valid, frame=frame,
                                        img_focal=img_focal, img_center=img_center)
            end_time = time.time()
            
            # 定义文件路径
            npy_file = f'{hps_folder}/hps_{timestamp}_{k}.npy'
            bvh_file = f'{hps_folder}/hps_{timestamp}_{k}.bvh'
            error_file = f'{hps_folder}/hps_{timestamp}_{k}.error'
            
            if results is not None:
                # 尝试保存npy文件
                npy_success = False
                bvh_success = False
                
                try:
                    np.save(npy_file, results)
                    # 验证npy文件是否正确保存
                    if os.path.exists(npy_file) and os.path.getsize(npy_file) > 0:
                        npy_success = True
                        print(f'  NPY saved: {npy_file}')
                    else:
                        print(f'  Error: NPY file not properly saved: {npy_file}')
                except Exception as e:
                    print(f'  Error saving NPY file: {e}')
                
                # 尝试转换并保存bvh文件
                if npy_success:
                    try:
                        # 转换为bvh格式并保存
                        # 创建npy2bvh转换器
                        npy2bvh_convert = npy2bvh(fps=min(video_fps, 30))
                        bvh_file = npy_file.replace('.npy', '.bvh')
                        npy2bvh_convert.convert_npy_to_bvh(npy_file, bvh_file)
                        
                        # 验证bvh文件是否正确生成
                        if os.path.exists(bvh_file) and os.path.getsize(bvh_file) > 0:
                            bvh_success = True
                            print(f'  BVH saved: {bvh_file}')
                        else:
                            print(f'  Error: BVH file not properly generated: {bvh_file}')
                    except Exception as e:
                        print(f'  Error converting to BVH: {e}')
                
                # 检查是否完全成功
                if npy_success and bvh_success:
                    successful_humans += 1
                    processing_time = end_time - start_time
                    total_frames_processed += len(trk)
                    print(f'  Human {k+1}: {len(trk)} frames processed, time: {processing_time:.2f} seconds')
                    print(f'  Average time per frame: {processing_time/len(trk):.3f} seconds')
                    print(f'  Success: Both NPY and BVH files generated')
                else:
                    # 部分失败，生成error文件
                    error_msg = f"Partial failure for human {k+1}:\n"
                    error_msg += f"NPY success: {npy_success}\n"
                    error_msg += f"BVH success: {bvh_success}\n"
                    error_msg += f"Timestamp: {datetime.now().isoformat()}\n"
                    error_msg += f"Frame count: {len(trk)}\n"
                    
                    try:
                        with open(error_file, 'w', encoding='utf-8') as f:
                            f.write(error_msg)
                        print(f'  Error file created: {error_file}')
                    except Exception as e:
                        print(f'  Failed to create error file: {e}')
                    
                    print(f'  Human {k+1}: Partial failure - check error file')
            
            if k+1 >= max_humans_to_process:
                break
        
        # 检查是否有任何人成功处理
        if successful_humans == 0:
            # 生成整体失败error文件
            overall_error_file = f'{hps_folder}/pipeline_{timestamp}.error'
            error_msg = f"Complete pipeline failure:\n"
            error_msg += f"No humans successfully processed\n"
            error_msg += f"Total humans attempted: {min(len(tracks), max_humans_to_process)}\n"
            error_msg += f"Timestamp: {datetime.now().isoformat()}\n"
            
            try:
                with open(overall_error_file, 'w', encoding='utf-8') as f:
                    f.write(error_msg)
                print(f'Overall error file created: {overall_error_file}')
            except Exception as e:
                print(f'Failed to create overall error file: {e}')
        
        # 计算总时间统计
        total_inference_time = time.time() - inference_start_time
        total_time = time.time() - step4_start
        
        # 计算平均每帧时间
        average_time_per_frame = total_inference_time / total_frames_processed if total_frames_processed > 0 else 0
        
        print('\n=== Human Pose Estimation Time Statistics ===')
        print(f'Model loading time: {model_load_time:.2f} seconds')
        print(f'Total inference time: {total_inference_time:.2f} seconds')
        print(f'Total processing time: {total_time:.2f} seconds')
        print(f'Total humans attempted: {min(len(tracks), max_humans_to_process)}')
        print(f'Successfully processed humans: {successful_humans}')
        print(f'Total frames processed: {total_frames_processed}')
        if total_frames_processed > 0:
            print(f'Average time per frame: {average_time_per_frame:.3f} seconds')
            print(f'Frames per second: {total_frames_processed/total_inference_time:.2f} fps')
        print('=============================================\n')
        
        step4_time = total_time
        
        # 记录步骤4完成状态
        step4_success = successful_humans > 0
        step4_error = None if step4_success else "No humans processed successfully"
        
        logger.complete_step("step4_pose_estimation", step4_start, success=step4_success, error_msg=step4_error,
                           total_humans_attempted=min(len(tracks), max_humans_to_process),
                           successful_humans=successful_humans,
                           total_frames_processed=total_frames_processed,
                           model_load_time=model_load_time,
                           average_time_per_frame=average_time_per_frame)
        
        if successful_humans > 0:
            print(f'Human pose estimation completed: {successful_humans}/{min(len(tracks), max_humans_to_process)} humans processed successfully')
        else:
            print('Human pose estimation completed with errors: No humans processed successfully')
            print('Check error files in the hps folder for details')
    
    except Exception as e:
        error_msg = f"Human pose estimation failed: {str(e)}"
        print(f'Error: {error_msg}')
        logger.complete_step("step4_pose_estimation", step4_start, success=False, error_msg=error_msg)
        logger.finalize(success=False, error_msg=error_msg)
        logger.save_to_json(f'{seq_folder}/pipeline_log.json')
        return

    
    # 步骤5: 可视化
    if not args.skip_visualization:
        print('Step 5: Visualization ...')
        step5_start = time.time()
        # 传递正确的图像文件夹路径
        if args.image_folder is not None:
            # 如果使用外部图像文件夹，需要修改visualize_tram函数调用
            print(f'Using external image folder: {img_folder}')
            # 临时创建符号链接或复制图像到期望的位置
            expected_img_folder = f'{seq_folder}/images'
            if os.path.exists(expected_img_folder):
                if os.path.islink(expected_img_folder):
                    os.unlink(expected_img_folder)
                else:
                    import shutil
                    shutil.rmtree(expected_img_folder)
            
            # 创建符号链接到外部图像文件夹
            os.symlink(os.path.abspath(img_folder), expected_img_folder)
            print(f'Created symbolic link: {expected_img_folder} -> {img_folder}')
        
        visualize_tram(seq_folder, floor_scale=args.floor_scale, bin_size=args.bin_size)
        step5_time = time.time() - step5_start
        print(f'Visualization completed in {step5_time:.2f} seconds')
        print(f'Visualization saved to: {seq_folder}/tram_output.mp4')
    else:
        print('Step 5: Skipping visualization')
        step5_time = 0
    
    # 计算总时间
    total_pipeline_time = time.time() - pipeline_start_time
    
    print('\n=== TRAM Pipeline Time Summary ===')
    if args.video is not None and args.image_folder is None:
        print(f'Step 1 - Frame extraction: {step1_time:.2f} seconds')
    print(f'Step 2 - Detection/Segmentation/Tracking: {step2_time:.2f} seconds')
    print(f'Step 3 - Camera estimation: {step3_time:.2f} seconds')
    print(f'Step 4 - Human pose estimation: {step4_time:.2f} seconds')
    if not args.skip_visualization:
        print(f'Step 5 - Visualization: {step5_time:.2f} seconds')
    print(f'Total pipeline time: {total_pipeline_time:.2f} seconds')
    print('===================================\n')
    
    print('\n=== TRAM Pipeline Completed ===')
    print(f'Results saved to: {seq_folder}')
    print(f'Camera parameters: {seq_folder}/camera.npy')
    print(f'Human poses: {seq_folder}/hps/')
    if not args.skip_visualization:
        print(f'Visualization: {seq_folder}/tram_output.mp4')

    logger.finalize(success=True, processed_humans=successful_humans, total_frames=total_frames_processed)
    logger.save_to_json(f'{seq_folder}/pipeline_log.json')

if __name__ == '__main__':
    main() 