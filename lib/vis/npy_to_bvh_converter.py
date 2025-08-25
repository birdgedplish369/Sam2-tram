#!/usr/bin/env python3
"""
NPY到BVH转换脚本 - 修复版本
解决关节通道、数据缩放和旋转问题
"""

import numpy as np
import torch
import os
import argparse
from scipy.spatial.transform import Rotation as R

# BVH关节层次结构 - 修复通道设置，并调整为真实人体比例
BVH_JOINT_HIERARCHY = {
    'Hips': {
        'parent': None,
        'children': ['Spine', 'LeftUpLeg', 'RightUpLeg'],
        'offset': [0.0, 0.0, 0.0],  # 根关节偏移设为0
        'channels': ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Xrotation', 'Yrotation']
    },
    'Spine': {
        'parent': 'Hips',
        'children': ['Spine1'],
        'offset': [0.0, 0.18, 0.0],  # 增加到18cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']  # 只有旋转通道
    },
    'Spine1': {
        'parent': 'Spine',
        'children': ['Spine2'],
        'offset': [0.0, 0.15, 0.0],  # 增加到15cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'Spine2': {
        'parent': 'Spine1',
        'children': ['Spine3'],
        'offset': [0.0, 0.15, 0.0],  # 增加到15cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'Spine3': {
        'parent': 'Spine2',
        'children': ['LeftShoulder', 'RightShoulder', 'Neck'],
        'offset': [0.0, 0.15, 0.0],  # 增加到15cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'LeftShoulder': {
        'parent': 'Spine3',
        'children': ['LeftArm'],
        'offset': [0.22, 0.06, 0.0],  # 增加肩膀宽度到22cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'LeftArm': {
        'parent': 'LeftShoulder',
        'children': ['LeftForeArm'],
        'offset': [0.32, 0.0, 0.0],  # 增加上臂长度到32cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'LeftForeArm': {
        'parent': 'LeftArm',
        'children': ['LeftHand'],
        'offset': [0.28, 0.0, 0.0],  # 增加前臂长度到28cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'LeftHand': {
        'parent': 'LeftForeArm',
        'children': [],
        'offset': [0.18, 0.0, 0.0],  # 增加手长到18cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'RightShoulder': {
        'parent': 'Spine3',
        'children': ['RightArm'],
        'offset': [-0.22, 0.06, 0.0],  # 增加肩膀宽度到22cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'RightArm': {
        'parent': 'RightShoulder',
        'children': ['RightForeArm'],
        'offset': [-0.32, 0.0, 0.0],  # 增加上臂长度到32cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'RightForeArm': {
        'parent': 'RightArm',
        'children': ['RightHand'],
        'offset': [-0.28, 0.0, 0.0],  # 增加前臂长度到28cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'RightHand': {
        'parent': 'RightForeArm',
        'children': [],
        'offset': [-0.18, 0.0, 0.0],  # 增加手长到18cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'Neck': {
        'parent': 'Spine3',
        'children': ['Head'],
        'offset': [0.0, 0.12, 0.0],  # 增加颈部长度到12cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'Head': {
        'parent': 'Neck',
        'children': [],
        'offset': [0.0, 0.22, 0.0],  # 增加头部高度到22cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'LeftUpLeg': {
        'parent': 'Hips',
        'children': ['LeftLeg'],
        'offset': [0.12, 0.0, 0.0],  # 增加髋部宽度到12cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'LeftLeg': {
        'parent': 'LeftUpLeg',
        'children': ['LeftFoot'],
        'offset': [0.0, -0.48, 0.0],  # 增加大腿长度到48cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'LeftFoot': {
        'parent': 'LeftLeg',
        'children': ['LeftToeBase'],
        'offset': [0.0, -0.42, 0.0],  # 增加小腿长度到42cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'LeftToeBase': {
        'parent': 'LeftFoot',
        'children': [],
        'offset': [0.0, -0.08, 0.25],  # 增加脚长到25cm，脚掌高度到8cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'RightUpLeg': {
        'parent': 'Hips',
        'children': ['RightLeg'],
        'offset': [-0.12, 0.0, 0.0],  # 增加髋部宽度到12cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'RightLeg': {
        'parent': 'RightUpLeg',
        'children': ['RightFoot'],
        'offset': [0.0, -0.48, 0.0],  # 增加大腿长度到48cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'RightFoot': {
        'parent': 'RightLeg',
        'children': ['RightToeBase'],
        'offset': [0.0, -0.42, 0.0],  # 增加小腿长度到42cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    },
    'RightToeBase': {
        'parent': 'RightFoot',
        'children': [],
        'offset': [0.0, -0.08, 0.25],  # 增加脚长到25cm，脚掌高度到8cm
        'channels': ['Zrotation', 'Xrotation', 'Yrotation']
    }
}

# 关节顺序 - 与BVH文件中的数据顺序对应
BVH_JOINT_ORDER = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'Neck', 'Head',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'
]

# SMPL关节到BVH关节的映射 - 标准24关节SMPL
SMPL_TO_BVH_MAPPING = {
    0: 'Hips',           # pelvis
    1: 'LeftUpLeg',      # left_hip  
    2: 'RightUpLeg',     # right_hip
    3: 'Spine',          # spine1
    4: 'LeftLeg',        # left_knee
    5: 'RightLeg',       # right_knee
    6: 'Spine1',         # spine2
    7: 'LeftFoot',       # left_ankle
    8: 'RightFoot',      # right_ankle
    9: 'Spine2',         # spine3
    10: 'LeftToeBase',   # left_foot
    11: 'RightToeBase',  # right_foot
    12: 'Neck',          # neck
    13: 'LeftShoulder',  # left_collar
    14: 'RightShoulder', # right_collar
    15: 'Head',          # head
    16: 'LeftArm',       # left_shoulder
    17: 'RightArm',      # right_shoulder
    18: 'LeftForeArm',   # left_elbow
    19: 'RightForeArm',  # right_elbow
    20: 'LeftHand',      # left_wrist
    21: 'RightHand',     # right_wrist
    22: 'Spine3',        # left_hand (映射到spine3)
    23: 'Spine3'         # right_hand (映射到spine3)
}

class NPYToBVHConverter:
    def __init__(self, fps=30):
        self.fps = fps
        self.frame_time = 1.0 / fps
        
    def load_npy_data(self, npy_file):
        """加载NPY文件并解析数据结构"""
        print(f"📁 加载NPY文件: {npy_file}")
        
        try:
            data = np.load(npy_file, allow_pickle=True)
            
            # 检查是否是字典格式
            if isinstance(data, np.ndarray) and data.ndim == 0:
                data = data.item()
            
            print(f"📊 数据类型: {type(data)}")
            
            if isinstance(data, dict):
                print(f"🔑 数据键: {list(data.keys())}")
                return data
            else:
                print(f"❌ 不支持的数据格式: {type(data)}")
                return None
                
        except Exception as e:
            print(f"❌ 加载NPY文件失败: {e}")
            return None
    
    def extract_pose_data(self, data):
        """从NPY数据中提取姿态信息"""
        pose_data = {}
        
        # 提取关键字段，自动处理torch.Tensor
        for key in ['pred_rotmat', 'pred_pose', 'pred_trans', 'pred_shape', 'pred_cam', 'frame']:
            if key in data:
                value = data[key]
                if isinstance(value, torch.Tensor):
                    pose_data[key] = value.cpu().detach().numpy()
                else:
                    pose_data[key] = value
        
        # 打印提取的数据信息
        print(f"🎯 提取的姿态数据:")
        for key, value in pose_data.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
        
        return pose_data
    
    def rotmat_to_euler(self, rotmat):
        """将旋转矩阵转换为欧拉角 (ZXY顺序, 度)"""
        if rotmat.shape[-2:] != (3, 3):
            print(f"❌ 错误的旋转矩阵维度: {rotmat.shape}")
            return None
        
        # 重塑为 (N, 3, 3) 格式
        original_shape = rotmat.shape[:-2]
        rotmat_reshaped = rotmat.reshape(-1, 3, 3)
        
        # 使用scipy进行转换 - ZXY顺序
        rotations = R.from_matrix(rotmat_reshaped)
        euler_angles = rotations.as_euler('ZXY', degrees=True)
        
        return euler_angles.reshape(*original_shape, 3)
    
    def axis_angle_to_euler(self, axis_angle):
        """将轴角表示转换为欧拉角 (ZXY顺序, 度)"""
        if axis_angle.shape[-1] != 3:
            print(f"❌ 错误的轴角维度: {axis_angle.shape}")
            return None
        
        # 使用scipy进行转换
        rotations = R.from_rotvec(axis_angle.reshape(-1, 3))
        euler_angles = rotations.as_euler('ZXY', degrees=True)
        
        return euler_angles.reshape(*axis_angle.shape[:-1], 3)
    
    def write_bvh_header(self, file_handle):
        """写入BVH文件头部"""
        file_handle.write("HIERARCHY\n")
        self._write_joint_hierarchy(file_handle, 'Hips', 0)
        
    def _write_joint_hierarchy(self, file_handle, joint_name, depth):
        """递归写入关节层次结构"""
        indent = "\t" * depth
        joint_info = BVH_JOINT_HIERARCHY[joint_name]
        
        if joint_info['parent'] is None:
            file_handle.write(f"ROOT {joint_name}\n")
        else:
            file_handle.write(f"{indent}JOINT {joint_name}\n")
        
        file_handle.write(f"{indent}{{\n")
        
        # 写入偏移量
        offset = joint_info['offset']
        file_handle.write(f"{indent}\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        
        # 写入通道
        channels = joint_info['channels']
        file_handle.write(f"{indent}\tCHANNELS {len(channels)} {' '.join(channels)}\n")
        
        # 递归写入子关节
        for child in joint_info['children']:
            self._write_joint_hierarchy(file_handle, child, depth + 1)
        
        # 如果没有子关节，写入End Site
        if not joint_info['children']:
            file_handle.write(f"{indent}\tEnd Site\n")
            file_handle.write(f"{indent}\t{{\n")
            file_handle.write(f"{indent}\t\tOFFSET 0.0 0.0 0.0\n")
            file_handle.write(f"{indent}\t}}\n")
        
        file_handle.write(f"{indent}}}\n")
    
    def convert_pose_to_bvh_frame(self, pose_data, frame_idx, initial_offset):
        """将单帧姿态数据转换为BVH帧数据"""
        bvh_frame = []
        
        # 处理旋转数据
        euler_angles = None
        if 'pred_rotmat' in pose_data:
            # 使用旋转矩阵
            rotmat = pose_data['pred_rotmat'][frame_idx]  # (24, 3, 3)
            euler_angles = self.rotmat_to_euler(rotmat)
        elif 'pred_pose' in pose_data:
            # 使用轴角表示
            pose = pose_data['pred_pose'][frame_idx]
            # 处理不同维度的姿态参数
            if len(pose) >= 72:
                pose_24joints = pose[:72].reshape(24, 3)
                euler_angles = self.axis_angle_to_euler(pose_24joints)
            else:
                print(f"⚠️ 姿态参数维度不足: {len(pose)}")
                euler_angles = np.zeros((24, 3))
        else:
            # 如果没有旋转数据，使用零旋转
            euler_angles = np.zeros((24, 3))
        
        # 限制旋转角度范围到合理值
        euler_angles = np.clip(euler_angles, -180, 180)
        
        # 处理全局位置
        global_trans = np.zeros(3)
        if 'pred_trans' in pose_data:
            trans = pose_data['pred_trans'][frame_idx]
            if trans.ndim == 2:
                trans = trans.flatten()  # (1, 3) -> (3,)
            # 应用缩放并减去初始偏移量，确保从原点开始
            global_trans = trans * 1  
            global_trans[1] = -global_trans[1]  # 翻转Y轴方向
            # 减去初始偏移量，使第一帧位于原点
            global_trans = global_trans - initial_offset
        
        # 按照BVH关节顺序生成帧数据
        for joint_name in BVH_JOINT_ORDER:
            if joint_name == 'Hips':
                # 根关节：全局位置 + 旋转
                bvh_frame.extend([global_trans[0], global_trans[1], global_trans[2]])
                
                # 根关节旋转 - 添加X轴180度旋转来翻转整个模型
                root_rot = euler_angles[0]  # SMPL的第一个关节是pelvis
                # 在X轴上加180度来翻转模型
                root_rot[1] = root_rot[1] + 180  # X轴旋转加180度
                # 确保角度在合理范围内
                if root_rot[1] > 180:
                    root_rot[1] -= 360
                elif root_rot[1] < -180:
                    root_rot[1] += 360
                    
                bvh_frame.extend([root_rot[0], root_rot[1], root_rot[2]])  # ZXY顺序
                
            else:
                # 非根关节：只有旋转
                # 查找对应的SMPL关节
                smpl_joint_idx = None
                for smpl_idx, bvh_joint in SMPL_TO_BVH_MAPPING.items():
                    if bvh_joint == joint_name:
                        smpl_joint_idx = smpl_idx
                        break
                
                if smpl_joint_idx is not None and smpl_joint_idx < len(euler_angles):
                    rot = euler_angles[smpl_joint_idx]
                    bvh_frame.extend([rot[0], rot[1], rot[2]])  # ZXY顺序
                else:
                    # 如果没有对应的SMPL关节，使用零旋转
                    bvh_frame.extend([0.0, 0.0, 0.0])
        
        return bvh_frame
    
    def convert_npy_to_bvh(self, npy_file, output_file):
        """将NPY文件转换为BVH文件"""
        print(f"🔄 开始转换: {npy_file} -> {output_file}")
        
        # 加载数据
        data = self.load_npy_data(npy_file)
        if data is None:
            return False
        
        # 提取姿态数据
        pose_data = self.extract_pose_data(data)
        if not pose_data:
            print("❌ 无法提取姿态数据")
            return False
        
        # 确定帧数
        frame_count = 0
        for key, value in pose_data.items():
            if hasattr(value, 'shape') and len(value.shape) > 0:
                frame_count = max(frame_count, value.shape[0])
        
        if frame_count == 0:
            print("❌ 没有找到有效的帧数据")
            return False
        
        # 计算第一帧的全局位置作为基准偏移（居中处理）
        initial_offset = np.zeros(3)
        if 'pred_trans' in pose_data:
            first_trans = pose_data['pred_trans'][0]
            if first_trans.ndim == 2:
                first_trans = first_trans.flatten()
            # 使用第一帧位置作为基准偏移，确保从原点开始
            initial_offset = first_trans * 1  # 保持原始缩放
            initial_offset[1] = -initial_offset[1]  # Y轴翻转
            print(f"📍 第一帧位置偏移: [{initial_offset[0]:.3f}, {initial_offset[1]:.3f}, {initial_offset[2]:.3f}]")
        
        # 计算总通道数
        total_channels = 0
        for joint_name in BVH_JOINT_ORDER:
            total_channels += len(BVH_JOINT_HIERARCHY[joint_name]['channels'])
        
        print(f"📊 总帧数: {frame_count}")
        print(f"📊 帧率: {self.fps} FPS")
        print(f"📊 每帧数据通道数: {total_channels}")
        print(f"📊 关节数: {len(BVH_JOINT_ORDER)}")
        
        # 写入BVH文件
        try:
            with open(output_file, 'w') as f:
                # 写入头部
                self.write_bvh_header(f)
                
                # 写入动画数据头
                f.write("MOTION\n")
                f.write(f"Frames: {frame_count}\n")
                f.write(f"Frame Time: {self.frame_time:.7f}\n")
                
                # 写入每帧数据
                for frame_idx in range(frame_count):
                    frame_data = self.convert_pose_to_bvh_frame(pose_data, frame_idx, initial_offset)
                    frame_str = ' '.join(f"{val:.6f}" for val in frame_data)
                    f.write(f"{frame_str}\n")
                
                print(f"✅ 转换成功: {output_file}")
                print(f"📝 BVH文件包含 {frame_count} 帧, {len(BVH_JOINT_ORDER)} 个关节")
                print(f"📝 每帧 {total_channels} 个通道数据")
                print(f"📝 人物从原点(0,0,0)开始")
                return True
                
        except Exception as e:
            print(f"❌ 写入BVH文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description='将TRAM项目的NPY文件转换为BVH文件 - 修复版本')
    parser.add_argument('input', help='输入的NPY文件路径')
    parser.add_argument('--output', '-o', help='输出的BVH文件路径')
    parser.add_argument('--fps', type=int, default=30, help='BVH文件的帧率 (默认: 30)')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    
    args = parser.parse_args()
    
    converter = NPYToBVHConverter(fps=args.fps)
    
    if args.batch:
        # 批量处理模式
        if os.path.isdir(args.input):
            input_dir = args.input
            output_dir = args.output or f"{input_dir}_bvh_fixed"
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
            
            print(f"🔍 找到 {len(npy_files)} 个NPY文件")
            
            success_count = 0
            for npy_file in npy_files:
                input_path = os.path.join(input_dir, npy_file)
                output_path = os.path.join(output_dir, npy_file.replace('.npy', '.bvh'))
                
                print(f"\n{'='*60}")
                if converter.convert_npy_to_bvh(input_path, output_path):
                    success_count += 1
            
            print(f"\n🎉 批量转换完成: {success_count}/{len(npy_files)} 个文件成功转换")
        else:
            print("❌ 批量模式需要输入目录")
    else:
        # 单文件处理模式
        input_file = args.input
        output_file = args.output or input_file.replace('.npy', '_fixed.bvh')
        
        if not os.path.exists(input_file):
            print(f"❌ 输入文件不存在: {input_file}")
            return
        
        converter.convert_npy_to_bvh(input_file, output_file)

if __name__ == "__main__":
    main() 