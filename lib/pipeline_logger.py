import os
import time
import json
from datetime import datetime

class PipelineLogger:
    """Pipeline执行日志记录器"""
    
    def __init__(self, video_path=None, output_folder=None):
        self.start_time = time.time()
        self.log_data = {
            "pipeline_info": {
                "video_name": os.path.basename(video_path) if video_path else "N/A",
                "video_path": video_path,
                "output_folder": output_folder,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "total_duration_seconds": 0,
                "success": False,
                "error_message": None,
                "processed_humans": 0,
                "total_frames": 0
            },
            "steps": {
                "step1_frame_extraction": {
                    "name": "视频帧提取",
                    "completed": False,
                    "duration_seconds": 0,
                    "start_time": None,
                    "end_time": None,
                    "details": {
                        "extracted_frames": 0,
                        "video_fps": 0,
                        "error": None
                    }
                },
                "step2_detection_tracking": {
                    "name": "人体检测分割和跟踪", 
                    "completed": False,
                    "duration_seconds": 0,
                    "start_time": None,
                    "end_time": None,
                    "details": {
                        "detected_tracks": 0,
                        "detection_method": "Grounding SAM2",
                        "error": None
                    }
                },
                "step3_camera_estimation": {
                    "name": "相机参数估计",
                    "completed": False,
                    "duration_seconds": 0,
                    "start_time": None,
                    "end_time": None,
                    "details": {
                        "calibration_successful": False,
                        "slam_successful": False,
                        "alignment_successful": False,
                        "error": None
                    }
                },
                "step4_pose_estimation": {
                    "name": "人体姿态估计",
                    "completed": False,
                    "duration_seconds": 0,
                    "start_time": None,
                    "end_time": None,
                    "details": {
                        "total_humans_attempted": 0,
                        "successful_humans": 0,
                        "total_frames_processed": 0,
                        "model_load_time": 0,
                        "average_time_per_frame": 0,
                        "error": None
                    }
                },
                "step5_visualization": {
                    "name": "可视化",
                    "completed": False,
                    "duration_seconds": 0,
                    "start_time": None,
                    "end_time": None,
                    "details": {
                        "skipped": False,
                        "error": None
                    }
                }
            }
        }
    
    def start_step(self, step_name):
        """开始记录某个步骤"""
        if step_name in self.log_data["steps"]:
            self.log_data["steps"][step_name]["start_time"] = datetime.now().isoformat()
            return time.time()
        return None
    
    def complete_step(self, step_name, start_time, success=True, error_msg=None, **details):
        """完成某个步骤的记录"""
        if step_name in self.log_data["steps"]:
            end_time = time.time()
            self.log_data["steps"][step_name]["completed"] = success
            self.log_data["steps"][step_name]["duration_seconds"] = round(end_time - start_time, 2)
            self.log_data["steps"][step_name]["end_time"] = datetime.now().isoformat()
            
            if error_msg:
                self.log_data["steps"][step_name]["details"]["error"] = error_msg
            
            # 更新详细信息
            for key, value in details.items():
                if key in self.log_data["steps"][step_name]["details"]:
                    self.log_data["steps"][step_name]["details"][key] = value
    
    def finalize(self, success=True, error_msg=None, processed_humans=0, total_frames=0):
        """完成整个pipeline的记录"""
        end_time = time.time()
        self.log_data["pipeline_info"]["end_time"] = datetime.now().isoformat()
        self.log_data["pipeline_info"]["total_duration_seconds"] = round(end_time - self.start_time, 2)
        self.log_data["pipeline_info"]["success"] = success
        self.log_data["pipeline_info"]["processed_humans"] = processed_humans
        self.log_data["pipeline_info"]["total_frames"] = total_frames
        
        if error_msg:
            self.log_data["pipeline_info"]["error_message"] = error_msg
    
    def save_to_json(self, file_path):
        """保存日志到JSON文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.log_data, f, indent=2, ensure_ascii=False)
            print(f'Pipeline log saved to: {file_path}')
            return True
        except Exception as e:
            print(f'Failed to save pipeline log: {e}')
            return False