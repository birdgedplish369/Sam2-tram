from os import replace
import torch
import numpy as np
import argparse
import pickle
import smplx

from lib.smpl2bvh.utils import bvh, quat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/root/tram/data")
    parser.add_argument("--model_type", type=str, default="smpl", choices=["smpl", "smplx"])
    parser.add_argument("--gender", type=str, default="MALE", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser.add_argument("--num_betas", type=int, default=10, choices=[10, 300])
    parser.add_argument("--file_path", type=str, default="/root/tram/results/walk_01")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mirror", action="store_true")
    return parser.parse_args()

def mirror_rot_trans(lrot, trans, names, parents):
    joints_mirror = np.array([(
        names.index("Left"+n[5:]) if n.startswith("Right") else (
        names.index("Right"+n[4:]) if n.startswith("Left") else 
        names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([1, 1, -1, -1])
    grot = quat.fk_rot(lrot, parents)
    trans_mirror = mirror_pos * trans
    grot_mirror = mirror_rot * grot[:,joints_mirror]
    
    return quat.ik_rot(grot_mirror, parents), trans_mirror

def smpl_to_world(pred_rotmat, pred_trans, world_cam_R, world_cam_T):
    """
    pred_rotmat: (N,24,3,3)  局部旋转（相机系/模型系）
    pred_trans:  (N,3) 或 (N,1,3)  根平移（相机系）
    world_cam_R: (3,3) 或 (N,3,3)  相机->世界 旋转
    world_cam_T: (3,)  或 (N,3)    相机->世界 平移
    return:
      world_rotmat: (N,24,3,3)  仅根关节做相机->世界转换；其余关节保持局部旋转不变
      world_trans:  (N,3)
    """
    pred_rotmat = np.asarray(pred_rotmat)
    N = pred_rotmat.shape[0]

    # 统一 pred_trans 形状
    pred_trans = np.asarray(pred_trans).reshape(N, 3)

    # 扩展/对齐相机外参到批量
    world_cam_R = np.asarray(world_cam_R)
    world_cam_T = np.asarray(world_cam_T)
    if world_cam_R.ndim == 2:
        world_cam_R = np.broadcast_to(world_cam_R[None, ...], (N, 3, 3))
    if world_cam_T.ndim == 1:
        world_cam_T = np.broadcast_to(world_cam_T[None, ...], (N, 3))

    # 仅根关节旋转左乘 R_wc，其余关节保持局部旋转
    world_rotmat = pred_rotmat.copy()
    # (N,3,3) @ (N,3,3) -> (N,3,3)
    world_rotmat[:, 0] = world_cam_R @ pred_rotmat[:, 0]

    # 根平移: t_w = R_wc * t_c + T_wc
    world_trans = (world_cam_R @ pred_trans[..., None]).squeeze(-1) + world_cam_T
    return world_rotmat, world_trans

# ---- 数学小工具：hat/exp/log on SO(3) ----
def _hat(v):  # (3,) -> (3,3)
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)

def _exp_so3(w):  # rotvec (3,) -> R(3,3)
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + _hat(w)
    k = w / theta
    K = _hat(k)
    ct, st = np.cos(theta), np.sin(theta)
    return ct*np.eye(3) + (1-ct)*np.outer(k, k) + st*K

def _log_so3(R):  # R(3,3) -> rotvec(3,)
    R = R.astype(np.float64)
    tr = np.trace(R)
    cos_th = np.clip((tr - 1.0)/2.0, -1.0, 1.0)
    th = np.arccos(cos_th)
    if th < 1e-12:
        return 0.5*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], dtype=np.float64)
    w = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], dtype=np.float64)
    return (th/(2.0*np.sin(th))) * w

def _project_to_so3(R):
    # 极分解，拉回最近的正交矩阵
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn

# ---- 1D 高斯平滑（时间维），反射 padding ----
def _gauss_kernel(sigma, truncate=3.0):
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64), 0
    r = max(1, int(truncate*sigma + 0.5))
    x = np.arange(-r, r+1, dtype=np.float64)
    k = np.exp(-0.5*(x/sigma)**2); k /= k.sum()
    return k, r

def _smooth_1d(x_TxC, sigma):
    if sigma <= 0: return x_TxC
    T, C = x_TxC.shape
    k, r = _gauss_kernel(sigma)
    y = np.empty_like(x_TxC, dtype=np.float64)
    for c in range(C):
        x = x_TxC[:, c]
        pad = np.r_[x[r:0:-1], x, x[-2:-r-2:-1]]
        y[:, c] = np.correlate(pad, k, mode='valid')
    return y.astype(x_TxC.dtype)


# ---------- 1) 高斯核与一维平滑 ----------
def _gaussian_kernel1d(sigma: float, truncate: float = 3.0):
    if sigma <= 0:
        k = np.array([1.0], dtype=np.float64)
        return k, 0
    radius = max(1, int(truncate * float(sigma) + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / float(sigma))**2)
    k /= k.sum()
    return k.astype(np.float64), radius

def _smooth_time_series(x: np.ndarray, sigma: float):
    """
    x: (T, C)  对时间维做高斯平滑；反射 padding，零相位（对称核）
    """
    if sigma <= 0:
        return x
    T, C = x.shape
    k, r = _gaussian_kernel1d(sigma)
    # 反射填充，逐通道做相关（等价于卷积，因核对称）
    y = np.empty_like(x, dtype=np.float64)
    for c in range(C):
        xc = x[:, c]
        pad = np.r_[xc[r:0:-1], xc, xc[-2:-r-2:-1]]  # reflect
        yc = np.correlate(pad, k, mode='valid')      # 长度 T
        y[:, c] = yc
    return y.astype(x.dtype)

# ---------- 2) 旋转：rotmat <-> rotation vector ----------
def rotmat_to_rotvec_batch(R: np.ndarray) -> np.ndarray:
    """
    R: (T, J, 3, 3)
    return: (T, J, 3)  轴角（弧度）
    """
    R = np.asarray(R, dtype=np.float64)
    T, J = R.shape[:2]

    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]           # (T,J)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)                              # (T,J)

    # vee(R - R^T)/2
    vee = 0.5 * np.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], axis=-1)                                               # (T,J,3)

    rotvec = np.zeros((T, J, 3), dtype=np.float64)

    small = theta < 1e-5          # 小角度：直接用一阶近似
    if np.any(small):
        rotvec[small] = vee[small]

    large = ~small
    if np.any(large):
        sin_theta = np.sin(theta[large])
        denom = (2.0 * sin_theta)[..., None]                  # (K,1)
        axis = vee[large] / denom                             # (K,3)
        rotvec[large] = axis * theta[large][..., None]        # (K,3)

    return rotvec.astype(np.float64)

def rotvec_to_rotmat_batch(rv: np.ndarray) -> np.ndarray:
    """
    rv: (T, J, 3)
    return: (T, J, 3, 3)
    """
    rv = np.asarray(rv, dtype=np.float64)
    T, J = rv.shape[:2]

    theta = np.linalg.norm(rv, axis=-1)           # (T,J)
    axis  = np.zeros_like(rv, dtype=np.float64)   # (T,J,3)

    nz = theta > 1e-12
    if np.any(nz):
        axis[nz] = rv[nz] / theta[nz][..., None]  # <-- 关键修正：(K,3)/(K,1)

    ax, ay, az = axis[..., 0], axis[..., 1], axis[..., 2]     # (T,J)

    ct = np.cos(theta)
    st = np.sin(theta)
    one_ct = 1.0 - ct

    R = np.empty((T, J, 3, 3), dtype=np.float64)
    R[..., 0, 0] = ct + ax*ax*one_ct
    R[..., 0, 1] = ax*ay*one_ct - az*st
    R[..., 0, 2] = ax*az*one_ct + ay*st

    R[..., 1, 0] = ay*ax*one_ct + az*st
    R[..., 1, 1] = ct + ay*ay*one_ct
    R[..., 1, 2] = ay*az*one_ct - ax*st

    R[..., 2, 0] = az*ax*one_ct - ay*st
    R[..., 2, 1] = az*ay*one_ct + ax*st
    R[..., 2, 2] = ct + az*az*one_ct

    # 极小角度直接置 I
    tiny = ~nz
    if np.any(tiny):
        R[tiny] = np.eye(3, dtype=np.float64)

    return R

# ---------- 3) 旋转/平移：时间平滑（面向 BVH） ----------
def smooth_rots_trans(
    rots,                # (T,24,3,3)  世界系旋转（仅 root 是全局，其余为局部）
    trans,               # (T,3) 或 (T,1,3) 世界系根平移（单位：米）
    mode="ema",          # "ema" 或 "gaussian"（根旋转在 SO(3) 上的去抖方式）
    alpha=0.35,          # EMA 强度（越小越稳：0.2~0.5 常用）
    sigma_rot=3.0,       # gaussian 模式下对“相对旋转”平滑的 σ（帧）
    sigma_trans=3.0,     # 平移平滑的 σ（帧）——同时作用 X/Y/Z
    smooth_height=True,  # True：平滑根的高度（Y）
    floor_lock="template",   # None/"template"/"sequence"：平滑后自动贴地
    offsets=None, names=None, # 用于估计骨盆->脚底高度；来自你 BVH 的 offsets/names（米）
    floor_quantile=2.0,      # sequence 模式下，以第 q 分位数作为地面
    floor_margin=0.0         # 贴地后留一点余量（米），例如 0.01
):
    """
    返回： rots_s (T,24,3,3), trans_s (T,3)
    - 根旋转：在 SO(3) 上对“帧间增量”做低通（不会改变整体朝向）
    - 平移：对 X/Y/Z 做高斯平滑；若 smooth_height=False 则仅 XZ 平滑
    - 贴地：用 offsets/names 估计脚底→骨盆的竖直偏移，再整体竖直平移使脚不低于地面
    """
    rots = rots.astype(np.float64)
    T = rots.shape[0]
    trans = np.asarray(trans, dtype=np.float64).reshape(T, 3)

    # ---------- 平移 ----------
    tr_in = trans.copy()
    if not smooth_height:
        # 只平滑 XZ：Y 保持原值
        xz = _smooth_1d(tr_in[:, [0, 2]], sigma_trans)
        trans_s = tr_in.copy()
        trans_s[:, 0] = xz[:, 0]; trans_s[:, 2] = xz[:, 1]
    else:
        trans_s = _smooth_1d(tr_in, sigma_trans)

    # ---------- 根关节旋转（index=0） ----------
    R_seq = rots[:, 0].copy()
    R_seq = np.stack([_project_to_so3(R) for R in R_seq], axis=0)

    if mode == "ema":
        R_hat = np.empty_like(R_seq)
        R_hat[0] = R_seq[0]
        for t in range(1, T):
            dRt = R_hat[t-1].T @ R_seq[t]  # 相对增量
            w = _log_so3(dRt)
            R_hat[t] = R_hat[t-1] @ _exp_so3(alpha * w)
        R_root = R_hat
    else:
        inc = np.zeros((T, 3), dtype=np.float64)
        for t in range(1, T):
            inc[t] = _log_so3(R_seq[t-1].T @ R_seq[t])
        inc_s = _smooth_1d(inc, sigma_rot)
        R_root = np.empty_like(R_seq)
        R_root[0] = R_seq[0]
        for t in range(1, T):
            R_root[t] = R_root[t-1] @ _exp_so3(inc_s[t])

    rots_s = rots.copy()
    rots_s[:, 0] = R_root

    # ---------- 自动贴地（避免平滑后的高度“穿地”） ----------
    if floor_lock in ("template", "sequence") and (offsets is not None) and (names is not None):
        # 关节名索引：如与你的 names 不同，请调整
        try:
            pelvis_id = names.index("Pelvis")
        except ValueError:
            pelvis_id = 0
        # 常见名称备选（如你的骨架命名不同请改）
        cand_L = [n for n in ["Left_foot", "L_Foot", "LeftFoot"] if n in names]
        cand_R = [n for n in ["Right_foot", "R_Foot", "RightFoot"] if n in names]
        lfoot_id = names.index(cand_L[0]) if cand_L else None
        rfoot_id = names.index(cand_R[0]) if cand_R else None

        # 骨盆->脚底的竖直偏移（米，通常为负）
        if lfoot_id is not None and rfoot_id is not None:
            foot_rel_m = min(offsets[lfoot_id, 1], offsets[rfoot_id, 1]) - offsets[pelvis_id, 1]
        else:
            # 兜底：用所有关节的最小 Y 近似
            foot_rel_m = np.min(offsets[:, 1]) - offsets[pelvis_id, 1]

        # 估计脚底高度（米）
        est_foot_y = trans_s[:, 1] + float(foot_rel_m)

        if floor_lock == "template":
            floor_level = np.min(est_foot_y)
        else:  # "sequence"
            floor_level = np.percentile(est_foot_y, floor_quantile)

        shift = (0.0 + float(floor_margin)) - float(floor_level)
        if shift > 0:
            trans_s[:, 1] += shift

    return rots_s.astype(rots.dtype), trans_s.astype(trans.dtype)



def smpl2bvh(file_path :str, 
             bvh_output_path :str,
             model_path = '/root/tram/data', 
             model_type = 'smpl', 
             gender = 'MALE', 
             num_betas = 10, 
             fps = 30) -> None:
    """Save bvh file created by smpl parameters.

    Args:
        model_path (str): Path to smpl models.
        poses (str): Path to npz or pkl file.
        output (str): Where to save bvh.
        mirror (bool): Whether save mirror motion or not.
        model_type (str, optional): I prepared "smpl" only. Defaults to "smpl".
        gender (str, optional): Gender Information. Defaults to "MALE".
        num_betas (int, optional): How many pca parameters to use in SMPL. Defaults to 10.
        fps (int, optional): Frame per second. Defaults to 30.
    """
    
    names = [
        "Pelvis",
        "Left_hip",
        "Right_hip",
        "Spine1",
        "Left_knee",
        "Right_knee",
        "Spine2",
        "Left_ankle",
        "Right_ankle",
        "Spine3",
        "Left_foot",
        "Right_foot",
        "Neck",
        "Left_collar",
        "Right_collar",
        "Head",
        "Left_shoulder",
        "Right_shoulder",
        "Left_elbow",
        "Right_elbow",
        "Left_wrist",
        "Right_wrist",
        "Left_palm",
        "Right_palm",
    ]
    # Pose setting.
    poses = bvh_output_path.replace('.bvh', '.npy')
    camera = file_path + '/camera.npy'
    output = bvh_output_path
    if poses.endswith(".npz"):
        poses = np.load(poses)
        
        # rots = np.squeeze(poses["poses"], axis=0) # (N, 24, 3)
        # trans = np.squeeze(poses["trans"], axis=0) # (N, 3)
        rots = poses["poses"]
        trans = poses["trans"]
    elif poses.endswith(".npy"):
        poses = np.load(poses, allow_pickle=True)
        camera = np.load(camera, allow_pickle=True)
        camera = camera.item()
        world_cam_R = camera['world_cam_R']  # 相机外参旋转矩阵 (3, 3)
        world_cam_T = camera['world_cam_T']  # 相机外参位移向量 (3,)

        d = poses.item()
        rots_cam = np.array(d["pred_rotmat"])  # (N, 24, 3, 3), 相机坐标系下的旋转矩阵
        trans_cam = np.array(d["pred_trans"].squeeze(1))  # (N, 3), 相机坐标系下的位移
        betas = np.array(d["pred_shape"])  # (N, 10)
        
        rots, trans = smpl_to_world(rots_cam, trans_cam, world_cam_R, world_cam_T)

    elif poses.endswith(".pkl"):
        with open(poses, "rb") as f:
            poses = pickle.load(f)
            rots = poses["smpl_poses"] # (N, 72)
            rots = rots.reshape(rots.shape[0], -1, 3) # (N, 24, 3)
            scaling = poses["smpl_scaling"]  # (1,)
            trans = poses["smpl_trans"]  # (N, 3)
    
    else:
        raise Exception("This file type is not supported!")
    # I prepared smpl models only, 
    # but I will release for smplx models recently.
    model = smplx.create(model_path=model_path, 
                        model_type=model_type,
                        gender=gender, 
                        batch_size=1)
    
    parents = model.parents.detach().cpu().numpy()
    
    # You can define betas like this.(default betas are 0 at all.)
    # 使用实际的betas参数而不是默认值
    if 'betas' in locals() and betas is not None:
        mean_betas = betas.mean(axis=0)  # 使用平均形状参数
        rest = model(
            betas = torch.tensor(mean_betas, dtype=torch.float32).unsqueeze(0)
        )
    else:
        rest = model()
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:24,:]
    
    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    # offsets改成cm
    offsets = offsets * 100.0

    # 脚掌对踝关节的offset更正，由于视频提取动作不是赤脚，加上本来脚掌就有点下倾（6cm），所以需要更正，总体更正到3cm
    lfoot_id = names.index("Left_foot")
    rfoot_id = names.index("Right_foot")
    offsets[lfoot_id][1] = -3.0
    offsets[rfoot_id][1] = -3.0

    scaling = None
    if scaling is not None:
        trans /= scaling

    # 使用新的平滑函数，包含脚掌修正
    rots, trans = smooth_rots_trans(
        rots, trans,
        mode="ema",
        alpha=0.30,
        sigma_rot=3.0,
        sigma_trans=3.0,
        smooth_height=True,
        floor_lock="template",
        floor_quantile=2.0,
        floor_margin=0.0
    )

    # rots = quat.from_axis_angle(rots)
    rots = quat.from_xform(rots) # 从旋转矩阵转换到四元数
    
    order = "zyx"
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    positions[:,0] += trans * 100.0


    pelvis_id = names.index("Pelvis")
    lfoot_id  = names.index("Left_foot")
    rfoot_id  = names.index("Right_foot")

    # 模板关节（厘米）
    pelvis_y = rest_pose[pelvis_id, 1]
    foot_y_rel_m = min(rest_pose[lfoot_id, 1], rest_pose[rfoot_id, 1]) - pelvis_y
    foot_y_rel_cm = foot_y_rel_m * 100.0
    lift_cm = - foot_y_rel_cm    
    positions[:, 0, 1] += lift_cm    

    # 地面校正：调整Y轴位置使人物站在地面上
    foot_rel_cm = (min(rest_pose[lfoot_id,1], rest_pose[rfoot_id,1]) - rest_pose[pelvis_id,1]) * 100.0
    foot_est_cm = positions[:, 0, 1] + foot_rel_cm          # 每帧估计脚高
    floor_cm    = np.percentile(foot_est_cm, 2)              # 稳健一点取低分位
    positions[:, 0, 1] -= floor_cm
    
    rotations = np.degrees(quat.to_euler(rots, order=order))
    
    bvh_data ={
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": names,
        "order": order,
        "frametime": 1 / fps,
    }
    
    if not output.endswith(".bvh"):
        output = output + ".bvh"
    
    bvh.save(output, bvh_data)
    
    # if mirror:
    #     rots_mirror, trans_mirror = mirror_rot_trans(
    #             rots, trans, names, parents)
    #     positions_mirror = pos.copy()
    #     positions_mirror[:,0] += trans_mirror
    #     rotations_mirror = np.degrees(
    #         quat.to_euler(rots_mirror, order=order))
        
    #     bvh_data ={
    #         "rotations": rotations_mirror,
    #         "positions": positions_mirror,
    #         "offsets": offsets,
    #         "parents": parents,
    #         "names": names,
    #         "order": order,
    #         "frametime": 1 / fps,
    #     }
        
    #     output_mirror = output.split(".")[0] + "_mirror.bvh"
    #     bvh.save(output_mirror, bvh_data)

if __name__ == "__main__":
    args = parse_args()
    # args.poses = '/home/bridge_shd/lish369/Code/tram-main/results/example_video/hps/hps_track_0.npy'
    # args.output = '/home/bridge_shd/lish369/Code/tram-main/results/example_video/hps/hps_track_0.bvh'
    # args.camera = '/home/bridge_shd/lish369/Code/tram-main/results/example_video/camera.npy'
    # args.poses = './lib/smpl2bvh-main/0005_Walking001_poses.npz'
    # args.output = './lib/smpl2bvh-main/0005_Walking001_poses.bvh'
    smpl2bvh(file_path=args.file_path, model_path=args.model_path, model_type=args.model_type, 
             mirror = args.mirror, gender=args.gender,
             num_betas=args.num_betas, 
             fps=args.fps)
    print("finished!")