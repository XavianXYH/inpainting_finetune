import trimesh
import argparse
import os
import numpy as np
import torch
import imageio
import nvdiffrast.torch as dr
import numpy as np
import torch
from PIL import Image

#----------------------------------------------------------------------------
# Projection and transformation matrix helpers.
#----------------------------------------------------------------------------

def projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0],
                     [  0,  n/x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]).astype(np.float32)

def projection_torch(tan_half_fov_x, tan_half_fov_y=None, near=1e-3, far=50.0):
    projection = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, -(far+near)/(far-near), -(2.0*far*near)/(far-near)],
                               [0.0, 0.0, -1.0, 0.0]], dtype=torch.float32).cuda()
    projection[0, 0] = 1.0 / tan_half_fov_x
    projection[1, 1] = 1.0 / tan_half_fov_x if tan_half_fov_y is None else 1.0 / tan_half_fov_y
    return projection


def translate_torch(trans, z_only=False):
    matrix = torch.eye(4, dtype=torch.float32).cuda()
    if z_only:
        matrix[2, 3] = trans[2]
    else:
        matrix[:3, 3] = trans
    return matrix
                        

def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0],
                     [0,  c, s, 0],
                     [0, -s, c, 0],
                     [0,  0, 0, 1]]).astype(np.float32)

def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]]).astype(np.float32)

def random_rotation_translation(t):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return m

#----------------------------------------------------------------------------
# Bilinear downsample by 2x.
#----------------------------------------------------------------------------

def bilinear_downsample(x):
    w = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=torch.float32, device=x.device) / 64.0
    w = w.expand(x.shape[-1], 1, 4, 4) 
    x = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, padding=1, stride=2, groups=x.shape[-1])
    return x.permute(0, 2, 3, 1)

#----------------------------------------------------------------------------
# Image display function using OpenGL.
#----------------------------------------------------------------------------

_glfw_window = None
def display_image(image, zoom=None, size=None, title=None): # HWC
    # Import OpenGL and glfw.
    import OpenGL.GL as gl
    import glfw

    # Zoom image if requested.
    image = np.asarray(image)
    if size is not None:
        assert zoom is None
        zoom = max(1, size // image.shape[0])
    if zoom is not None:
        image = image.repeat(zoom, axis=0).repeat(zoom, axis=1)
    height, width, channels = image.shape

    # Initialize window.
    if title is None:
        title = 'Debug window'
    global _glfw_window
    if _glfw_window is None:
        glfw.init()
        _glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(_glfw_window)
        glfw.show_window(_glfw_window)
        glfw.swap_interval(0)
    else:
        glfw.make_context_current(_glfw_window)
        glfw.set_window_title(_glfw_window, title)
        glfw.set_window_size(_glfw_window, width, height)

    # Update window.
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {'uint8': gl.GL_UNSIGNED_BYTE, 'float32': gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(_glfw_window)
    if glfw.window_should_close(_glfw_window):
        return False
    return True

#----------------------------------------------------------------------------
# Image save helper.
#----------------------------------------------------------------------------

def save_image(fn, x):
    import imageio
    x = np.rint(x * 255.0)
    x = np.clip(x, 0, 255).astype(np.uint8)
    imageio.imsave(fn, x)

#----------------------------------------------------------------------------
# Quaternion math.
#----------------------------------------------------------------------------

# Unit quaternion.
def q_unit():
    return np.asarray([1, 0, 0, 0], np.float32)

# Get a random normalized quaternion.
def q_rnd():
    u, v, w = np.random.uniform(0.0, 1.0, size=[3])
    v *= 2.0 * np.pi
    w *= 2.0 * np.pi
    return np.asarray([(1.0-u)**0.5 * np.sin(v), (1.0-u)**0.5 * np.cos(v), u**0.5 * np.sin(w), u**0.5 * np.cos(w)], np.float32)

# Get a random quaternion from the octahedral symmetric group S_4.
_r2 = 0.5**0.5
_q_S4 = [[ 1.0, 0.0, 0.0, 0.0], [ 0.0, 1.0, 0.0, 0.0], [ 0.0, 0.0, 1.0, 0.0], [ 0.0, 0.0, 0.0, 1.0],
         [-0.5, 0.5, 0.5, 0.5], [-0.5,-0.5,-0.5, 0.5], [ 0.5,-0.5, 0.5, 0.5], [ 0.5, 0.5,-0.5, 0.5],
         [ 0.5, 0.5, 0.5, 0.5], [-0.5, 0.5,-0.5, 0.5], [ 0.5,-0.5,-0.5, 0.5], [-0.5,-0.5, 0.5, 0.5],
         [ _r2,-_r2, 0.0, 0.0], [ _r2, _r2, 0.0, 0.0], [ 0.0, 0.0, _r2, _r2], [ 0.0, 0.0,-_r2, _r2],
         [ 0.0, _r2, _r2, 0.0], [ _r2, 0.0, 0.0,-_r2], [ _r2, 0.0, 0.0, _r2], [ 0.0,-_r2, _r2, 0.0],
         [ _r2, 0.0, _r2, 0.0], [ 0.0, _r2, 0.0, _r2], [ _r2, 0.0,-_r2, 0.0], [ 0.0,-_r2, 0.0, _r2]]
def q_rnd_S4():
    return np.asarray(_q_S4[np.random.randint(24)], np.float32)

# Quaternion slerp.
def q_slerp(p, q, t):
    d = np.dot(p, q)
    if d < 0.0:
        q = -q
        d = -d
    if d > 0.999:
        a = p + t * (q-p)
        return a / np.linalg.norm(a)
    t0 = np.arccos(d)
    tt = t0 * t
    st = np.sin(tt)
    st0 = np.sin(t0)
    s1 = st / st0
    s0 = np.cos(tt) - d*s1
    return s0*p + s1*q

# Quaterion scale (slerp vs. identity quaternion).
def q_scale(q, scl):
    return q_slerp(q_unit(), q, scl)

# Quaternion product.
def q_mul(p, q):
    s1, V1 = p[0], p[1:]
    s2, V2 = q[0], q[1:]
    s = s1*s2 - np.dot(V1, V2)
    V = s1*V2 + s2*V1 + np.cross(V1, V2)
    return np.asarray([s, V[0], V[1], V[2]], np.float32)

# Angular difference between two quaternions in degrees.
def q_angle_deg(p, q):
    p = p.detach().cpu().numpy()
    q = q.detach().cpu().numpy()
    d = np.abs(np.dot(p, q))
    d = min(d, 1.0)
    return np.degrees(2.0 * np.arccos(d))

# Quaternion product
def q_mul_torch(p, q):
    a = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    b = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    c = p[0]*q[2] + p[2]*q[0] + p[3]*q[1] - p[1]*q[3]
    d = p[0]*q[3] + p[3]*q[0] + p[1]*q[2] - p[2]*q[1]
    return torch.stack([a, b, c, d])

# Convert quaternion to 4x4 rotation matrix.
def q_to_mtx(q):
    r0 = torch.stack([1.0-2.0*q[1]**2 - 2.0*q[2]**2, 2.0*q[0]*q[1] - 2.0*q[2]*q[3], 2.0*q[0]*q[2] + 2.0*q[1]*q[3]])
    r1 = torch.stack([2.0*q[0]*q[1] + 2.0*q[2]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[2]**2, 2.0*q[1]*q[2] - 2.0*q[0]*q[3]])
    r2 = torch.stack([2.0*q[0]*q[2] - 2.0*q[1]*q[3], 2.0*q[1]*q[2] + 2.0*q[0]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[1]**2])
    rr = torch.transpose(torch.stack([r0, r1, r2]), 1, 0)
    rr = torch.cat([rr, torch.tensor([[0], [0], [0]], dtype=torch.float32).cuda()], dim=1) # Pad right column.
    rr = torch.cat([rr, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).cuda()], dim=0)  # Pad bottom row.
    return rr

# Transform vertex positions to clip space
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]


def render(glctx, resolution, mtx, pos, face, col=None, col_idx=None):
    if col_idx is None or col is None:
        col = torch.ones_like(pos[:1, :], dtype=torch.float32).cuda()
        col_idx = torch.zeros_like(face)
    pos_clip    = transform_pos(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, face, resolution=[resolution[0], resolution[1]])
    color   , _ = dr.interpolate(col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, face)
    return color


def resize_foreground(
    image: torch.Tensor,
    ratio: float = 0.85,
    resize_to_origin = True,
) -> torch.Tensor:
    if image.ndim == 4:
        image = image.squeeze(0)
    H, W = image.shape[:2]
    if image.shape[-1] == 3:
        alpha = (image.abs().sum(dim=-1) > 0).float()
    else:
        alpha = (image[..., 3] > 0).float()
    alpha = torch.where(alpha > 0)
    if alpha[0].shape[0] < 4:
        return image[None]
    y1, y2, x1, x2 = (
        alpha[0].min().item(),
        alpha[0].max().item(),
        alpha[1].min().item(),
        alpha[1].max().item(),
    )
    # Crop the foreground
    fg = image[y1:y2, x1:x2]
    # Pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = torch.nn.functional.pad(
        fg,
        (0, 0, pw0, pw1, ph0, ph1),
        mode="constant",
        value=0,
    )

    # Compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = torch.nn.functional.pad(
        new_image,
        (0, 0, pw0, pw1, ph0, ph1),
        mode="constant",
        value=0,
    )

    if resize_to_origin:
        # Resize to the original size
        new_image = torch.nn.functional.interpolate(
            new_image.permute(2, 0, 1).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
    return new_image[None]

def load_image_fixed(image_path):
    img_array = np.asarray(Image.open(image_path)) / 255.0
    
    # 根据图像通道数选择处理方式
    if img_array.ndim == 2:  # 灰度图
        mask = img_array
    elif img_array.shape[2] == 4:  # RGBA图像，取alpha通道
        mask = img_array[..., 3]
    else:  # RGB图像，转换为灰度
        mask = np.mean(img_array, axis=-1)
    
    return torch.from_numpy(mask).to(torch.float32).cuda()[None, ..., None].expand(-1, -1, -1, 3)
    


def euler_to_quaternion(yaw, pitch, roll):
    """欧拉角转四元数 (Z-Y-X顺序)"""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return torch.stack([w, x, y, z])


def fit_pose_crop(
             mesh_path,
             image_path,
             max_iter           = 10000,
             repeats            = 1,
             log_interval       = 10,
             display_interval   = None,
             display_res        = 512,
             lr_falloff         = 1e-3,
             nr_base            = 1.0,
             nr_falloff         = 1e-4,
             grad_phase_start   = 0.50,
             out_dir            = None,
             log_fn             = None,
             mp4save_interval   = None,
             mp4save_fn         = None,
             use_opengl         = False,
             pitch_range        = (-20, 60),  # 俯仰角范围（度）
             roll_range         = (-5, 5)):   # 横滚角范围（度）

    log_file = None
    writer = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if log_fn:
            log_file = open(out_dir + '/' + log_fn, 'wt')
        if mp4save_interval != 0:
            if mp4save_fn.endswith('.gif'):
                writer = imageio.get_writer(f'{out_dir}/{mp4save_fn}', mode='I', codec='libx264', bitrate='16M')
            else:
                writer = imageio.get_writer(f'{out_dir}/{mp4save_fn}', fps=30)
    else:
        mp4save_interval = None

    mesh = trimesh.load_mesh(mesh_path)
    vertices, face = mesh.vertices, mesh.faces
    print("Mesh has %d triangles and %d vertices." % (face.shape[0], vertices.shape[0]))

    image = load_image_fixed(image_path)
    image = resize_foreground(image, resize_to_origin=False)
    face = torch.from_numpy(face.astype(np.int32)).cuda()
    vertices = torch.from_numpy(vertices.astype(np.float32)).cuda()
    center = (vertices.max(0)[0] + vertices.min(0)[0]) * 0.5
    H, W = image.shape[1:3]

    alpha = torch.where(image[0, ..., -1] > 0)
    min_w, max_w, min_h, max_h = alpha[1].min(), alpha[1].max(), alpha[0].min(), alpha[0].max()
    avg_img_size = (max_w - min_w + max_h - min_h) / 2
    avg_3d_size = vertices.max() - vertices.min()
    z_init = 1.
    focal_init = z_init / avg_3d_size * avg_img_size / (W / 2) * 1.5

    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

    # 尝试多个初始yaw角
    best_overall_loss = np.inf
    best_overall_rot = None
    
    # 尝试4个主要方向作为初始值
    initial_yaws = [0, np.pi/2, np.pi, -np.pi/2]
    
    for init_idx, init_yaw in enumerate(initial_yaws):
        print(f"\nTrying initial yaw: {np.degrees(init_yaw):.0f} degrees")
        
        # 使用欧拉角参数化
        yaw = torch.tensor(init_yaw, dtype=torch.float32, device='cuda', requires_grad=True)
        pitch = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=True)
        roll = torch.tensor(0.0, dtype=torch.float32, device='cuda', requires_grad=True)
        trans = torch.tensor([0.0, 0.0, -1.5], dtype=torch.float32, device='cuda', requires_grad=True)
        trans.data -= center
        focal = torch.tensor(focal_init, dtype=torch.float32, device='cuda', requires_grad=True)

        loss_best = np.inf
        rot_best = None
        proj_best = None

        # 定义基础学习率字典
        base_lr_dict = {
            'yaw': 2e-1,     # 主要自由度
            'pitch': 5e-2,   # 次要自由度
            'roll': 1e-2,    # 最小自由度
            'trans': 1e-1,
            'focal': 1e-1
        }

        # Adam optimizer with different learning rates for different parameters
        optimizer = torch.optim.Adam([
            {'params': [yaw],   'lr': base_lr_dict['yaw'],   'name': 'yaw'},
            {'params': [pitch], 'lr': base_lr_dict['pitch'], 'name': 'pitch'},
            {'params': [roll],  'lr': base_lr_dict['roll'],  'name': 'roll'},
            {'params': [trans], 'lr': base_lr_dict['trans'], 'name': 'trans'},
            {'params': [focal], 'lr': base_lr_dict['focal'], 'name': 'focal'}
        ])

        # Render.
        for it in range(max_iter // 4):  # 每个初始值运行1/4的迭代
            # Set learning rate.
            itf = 1.0 * it / (max_iter // 4)
            nr = nr_base * nr_falloff**itf
            
            # 更新学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr_dict[param_group['name']] * lr_falloff ** itf

            # 约束角度范围
            with torch.no_grad():
                pitch.data = torch.clamp(pitch.data, 
                                       np.radians(pitch_range[0]), 
                                       np.radians(pitch_range[1]))
                roll.data = torch.clamp(roll.data, 
                                       np.radians(roll_range[0]), 
                                       np.radians(roll_range[1]))

            # Noise input - 主要是yaw噪声
            if itf >= grad_phase_start:
                yaw_noise = 0.0
                pitch_noise = 0.0
                roll_noise = 0.0
                t_noise = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')
            else:
                # 根据阶段调整噪声
                yaw_noise = (torch.rand(1).item() - 0.5) * 2 * np.pi * nr  # 大范围yaw搜索
                pitch_noise = (torch.rand(1).item() - 0.5) * np.radians(30) * nr * 0.5  # 中等pitch搜索
                roll_noise = (torch.rand(1).item() - 0.5) * np.radians(5) * nr * 0.2   # 小范围roll
                t_noise = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')

            # 转换欧拉角到四元数
            rot_total = euler_to_quaternion(
                yaw + yaw_noise,
                pitch + pitch_noise,
                roll + roll_noise
            )

            # Render.
            tan_half_fov_x = 1.0 / focal
            tan_half_fov_y = tan_half_fov_x * (H / W)
            intr_matrix = projection_torch(tan_half_fov_x, tan_half_fov_y, near=1e-2, far=50.0)
            tran_matrix = translate_torch(trans + t_noise, z_only=True)
            mvp = torch.matmul(intr_matrix, tran_matrix)
            proj_matrix = torch.matmul(mvp, q_to_mtx(rot_total))
            color_opt = render(glctx, [H, W], proj_matrix, vertices, face)

            # Crop the foreground
            if itf < grad_phase_start:
                color_opt = resize_foreground(color_opt)

            # 改进的损失函数
            # 1. 获取mask
            mask_opt = color_opt[0, ..., 0]
            mask_ref = image[0, ..., 0]

            # 2. IoU损失
            intersection = torch.min(mask_opt, mask_ref).sum()
            union = torch.max(mask_opt, mask_ref).sum()
            iou = intersection / (union + 1e-6)
            iou_loss = 1.0 - iou

            # 3. 像素级损失
            diff = (color_opt - image)**2
            diff = torch.tanh(3.0 * torch.max(diff, dim=-1)[0])
            pixel_loss = torch.mean(diff)

            # 4. 正则化项：保持接近竖直
            if itf >= grad_phase_start:
                # 期望模型保持相对竖直
                upright_penalty = 0.01 * (pitch**2 + roll**2)
            else:
                upright_penalty = 0.0

            # 5. 组合损失
            loss = 0.5 * iou_loss + 0.5 * pixel_loss + upright_penalty

            # Measure image-space loss and update best found pose.
            loss_val = float(loss)
            if (loss_val < loss_best) and (loss_val > 0.0):
                proj_best = proj_matrix.detach().clone()
                rot_best = rot_total.detach().clone()
                focal_best = focal.detach().clone()
                trans_best = (trans + t_noise).detach().clone()
                loss_best = loss_val
                yaw_best = (yaw + yaw_noise).detach().clone()
                pitch_best = (pitch + pitch_noise).detach().clone()
                roll_best = (roll + roll_noise).detach().clone()

            # Print/save log.
            if log_interval and (it % log_interval == 0):
                s = "loss=%.3f, loss_best=%.3f, yaw=%.1f°, pitch=%.1f°, roll=%.1f°, focal=%.2f" % (
                    loss_val, loss_best, 
                    np.degrees(yaw.item()), np.degrees(pitch.item()), np.degrees(roll.item()),
                    focal.item())
                print(s)
                if log_file:
                    log_file.write(s + "\n")

            # Run gradient training step.
            if itf >= grad_phase_start:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # 在进入梯度阶段时，使用最佳参数初始化
            if itf >= grad_phase_start and 1.0 * (it - 1) / (max_iter // 4) <= grad_phase_start:
                with torch.no_grad():
                    yaw.data = yaw_best
                    pitch.data = pitch_best
                    roll.data = roll_best
                    focal.data = focal_best
                    trans[:] = trans_best

            # Show/save image.
            display_image_flag = display_interval and (it % display_interval == 0)
            save_mp4 = mp4save_interval and (it % mp4save_interval == 0)

            if display_image_flag or save_mp4:
                img_ref = image[0].detach().cpu().numpy()
                img_opt = color_opt[0].detach().cpu().numpy()
                if proj_best is not None:
                    img_best = render(glctx, [H, W], proj_best, vertices, face)
                    img_best = resize_foreground(img_best)[0].detach().cpu().numpy()
                else:
                    img_best = img_opt
                result_image = np.concatenate([img_ref, img_best, img_opt], axis=1)

                if display_image_flag:
                    display_image(result_image, size=display_res, title='Init %d: (%d) / %d' % (init_idx, it, max_iter//4))
                if save_mp4:
                    writer.append_data(np.clip(np.rint(result_image*255.0), 0, 255).astype(np.uint8))
        
        # 保存这个初始值的最佳结果
        if loss_best < best_overall_loss:
            best_overall_loss = loss_best
            best_overall_rot = rot_best
            print(f"New best loss: {best_overall_loss:.4f} from initial yaw {np.degrees(init_yaw):.0f}°")

    # Done.
    if writer is not None:
        writer.close()
    if log_file:
        log_file.close()
    
    print(f"\nFinal best loss: {best_overall_loss:.4f}")
    return best_overall_rot


def fit_pose_full(
             mesh_path,
             image_path,
             rot_init,
             max_iter           = 10000,
             repeats            = 1,
             log_interval       = 10,
             display_interval   = None,
             display_res        = 512,
             lr_falloff         = 1e-3,
             out_dir            = None,
             log_fn             = None,
             mp4save_interval   = None,
             mp4save_fn         = None,
             use_opengl         = False):

    log_file = None
    writer = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if log_fn:
            log_file = open(out_dir + '/' + log_fn, 'wt')
        if mp4save_interval != 0:
            if mp4save_fn.endswith('.gif'):
                writer = imageio.get_writer(f'{out_dir}/{mp4save_fn}', mode='I', codec='libx264', bitrate='16M')
            else:
                writer = imageio.get_writer(f'{out_dir}/{mp4save_fn}', fps=30)
    else:
        mp4save_interval = None

    mesh = trimesh.load_mesh(mesh_path)
    vertices, face = mesh.vertices, mesh.faces
    print("Mesh has %d triangles and %d vertices." % (face.shape[0], vertices.shape[0]))

    image = load_image_fixed(image_path)
    alpha = torch.where(image[0, ..., -1] > 0)
    min_w, max_w, min_h, max_h = alpha[1].min(), alpha[1].max(), alpha[0].min(), alpha[0].max()
    face = torch.from_numpy(face.astype(np.int32)).cuda()
    vertices = torch.from_numpy(vertices.astype(np.float32)).cuda()
    center = (vertices.max(0)[0] + vertices.min(0)[0]) * 0.5
    H, W = image.shape[1:3]
    avg_img_size = (max_w - min_w + max_h - min_h) / 2
    avg_3d_size = vertices.max() - vertices.min()
    z_init = 1.
    focal_init = z_init / avg_3d_size * avg_img_size / (W / 2) * 1.2
    offset_u, offset_v = (max_w + min_w - W) / 2, (max_h + min_h - H) / 2
    offset_x, offset_y = offset_u / (focal_init * (W / 2)) * z_init, offset_v / (focal_init * (W / 2)) * z_init

    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

    for rep in range(repeats):
        rot        = torch.tensor(rot_init / (rot_init**2).sum()**0.5, dtype=torch.float32, device='cuda', requires_grad=True)
        trans      = torch.tensor([offset_x, offset_y, -z_init], dtype=torch.float32, device='cuda', requires_grad=True)
        trans.data -= center
        focal      = torch.tensor(focal_init, dtype=torch.float32, device='cuda', requires_grad=True)

        loss_best   = np.inf
        proj_best   = None

        # Adam optimizer for texture with a learning rate ramp.
        base_lr_dict = {'rot': 1e-2, 'trans': 1e-2, 'focal': 1e-2}
        optimizer = torch.optim.Adam([
            {'params': [rot],   'lr': base_lr_dict['rot'],   'name': 'rot'},
            {'params': [trans], 'lr': base_lr_dict['trans'], 'name': 'trans'},
            {'params': [focal], 'lr': base_lr_dict['focal'], 'name': 'focal'}
        ],)

        # Render.
        for it in range(max_iter + 1):
            # Set learning rate.
            itf = 1.0 * it / max_iter
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr_dict[param_group['name']] * lr_falloff ** itf

            # Render.
            tan_half_fov_x = 1.0 / focal
            tan_half_fov_y = tan_half_fov_x * (H / W)
            intr_matrix = projection_torch(tan_half_fov_x, tan_half_fov_y, near=1e-2, far=50.0)
            tran_matrix = translate_torch(trans, z_only=False)
            mvp = torch.matmul(intr_matrix, tran_matrix)
            proj_matrix  = torch.matmul(mvp, q_to_mtx(rot))
            color_opt      = render(glctx, [H, W], proj_matrix, vertices, face)

            # Image-space loss.
            diff = (color_opt - image)**2 # L2 norm.
            diff = torch.tanh(5.0 * torch.max(diff, dim=-1)[0])
            loss = torch.mean(diff)

            # Measure image-space loss and update best found pose.
            loss_val = float(loss)
            if (loss_val < loss_best) and (loss_val > 0.0):
                proj_best = proj_matrix.detach().clone()
                rot_best  = rot.detach().clone()
                focal_best = focal.detach().clone()
                trans_best = trans.detach().clone()
                loss_best = loss_val

            # Print/save log.
            if log_interval and (it % log_interval == 0):
                s = "loss=%.2f, loss_best=%.2f, focal=%.2f, trans=[%.2f, %.2f, %.2f]" % (loss_val, loss_best, focal.item(), trans[0].item(), trans[1].item(), trans[2].item())
                print(s)
                if log_file:
                    log_file.write(s + "\n")

            # Run gradient training step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                rot /= torch.sum(rot**2)**0.5

            # Show/save image.
            display_image = display_interval and (it % display_interval == 0)
            save_mp4      = mp4save_interval and (it % mp4save_interval == 0)

            if display_image or save_mp4:
                img_ref  = image[0].detach().cpu().numpy()
                img_opt  = color_opt[0].detach().cpu().numpy()
                img_best = render(glctx, [H, W], proj_best, vertices, face)[0].detach().cpu().numpy()
                result_image = np.concatenate([img_ref, img_best, img_opt], axis=1)

                if display_image:
                    display_image(result_image, size=display_res, title='(%d) %d / %d' % (rep, it, max_iter))
                if save_mp4:
                    writer.append_data(np.clip(np.rint(result_image*255.0), 0, 255).astype(np.uint8))

    # Done.
    if writer is not None:
        writer.close()
    if log_file:
        log_file.close()
    
    return rot_best, trans_best, focal_best * W / 2


def main():
    parser = argparse.ArgumentParser(description='Cube pose fitting example')
    parser.add_argument('--opengl', help='enable OpenGL rendering', action='store_true', default=False)
    parser.add_argument('--outdir', help='specify output directory', default='./output/')
    parser.add_argument('--display-interval', type=int, default=0)
    parser.add_argument('--mp4save-interval', type=int, default=10)
    parser.add_argument('--max-iter-crop', type=int, default=2000)
    parser.add_argument('--max-iter-full', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()

    # Set up logging.
    if args.outdir:
        out_dir = f'{args.outdir}/pose'
        print (f'Saving results under {out_dir}')
    else:
        out_dir = None
        print ('No output directory specified, not saving log or images')

    mesh_path='/home/xyhugo/CNtest/Archive/sample.glb'
    # mesh_path='/home/yihua/disk8T/dyaigc/repos/TripoSG/sequences/camel_3d/00004.obj'
    # image_path='/home/yihua/disk8T/dyaigc/repos/TripoSR/seq_out_crop/input_00004.png'
    image_path='/home/xyhugo/CNtest/Archive/car_object.png'

    # Run.
    rot_init = fit_pose_crop(
        mesh_path=mesh_path,
        image_path=image_path,
        max_iter=args.max_iter_crop,
        repeats=args.repeats,
        log_interval=100,
        display_interval=args.display_interval,
        out_dir=out_dir,
        log_fn='log.txt',
        mp4save_interval=args.mp4save_interval,
        mp4save_fn='progress_crop.mp4',
        use_opengl=args.opengl
    )

    fit_pose_full(
        mesh_path=mesh_path,
        image_path=image_path,
        rot_init=rot_init,
        max_iter=args.max_iter_full,
        repeats=args.repeats,
        log_interval=100,
        display_interval=args.display_interval,
        out_dir=out_dir,
        log_fn='log.txt',
        mp4save_interval=args.mp4save_interval,
        mp4save_fn='progress_full.mp4',
        use_opengl=args.opengl
    )

    # Done.
    print("Done.")


if __name__ == "__main__":
    main()