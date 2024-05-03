import skimage
import os
import cv2
import scipy
import random
import imageio
import colorsys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from typing import Optional
from PIL import Image

import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plyfile import (PlyData, PlyElement)
from scipy import signal

mse2psnr = lambda x : -10. * torch.log10(x)

parent_dir = os.path.dirname(os.path.abspath(__file__))
adam_cuda = load(
        name='adam_upd_cuda',
        sources=[os.path.join(parent_dir, path) for path in ['cuda/adam_upd.cpp', 'cuda/adam_upd_kernel.cu']],
        verbose=True)

render_cuda = load(
        name='render_utils_cuda',
        sources=[os.path.join(parent_dir, path) for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

tv_cuda = load(
        name='total_variation_cuda',
        sources=[os.path.join(parent_dir, path) for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None

def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS

''' Extend Adam optimizer
1. support per-voxel learning rate
2. masked update (ignore zero grad) which speeduping training
'''
class MaskedAdam(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        self.per_lr = None
        super(MaskedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MaskedAdam, self).__setstate__(state)

    def set_pervoxel_lr(self, count):
        assert self.param_groups[0]['params'][0].shape == count.shape
        self.per_lr = count.float() / count.max()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            skip_zero_grad = group['skip_zero_grad']

            for param in group['params']:
                if param.grad is not None:
                    state = self.state[param]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

                    state['step'] += 1

                    if self.per_lr is not None and param.shape == self.per_lr.shape:
                        adam_cuda.adam_upd_with_perlr(
                                param, param.grad, state['exp_avg'], state['exp_avg_sq'], self.per_lr,
                                state['step'], beta1, beta2, lr, eps)
                    elif skip_zero_grad:
                        adam_cuda.masked_adam_upd(
                                param.float(), param.grad.float(), state['exp_avg'].float(), state['exp_avg_sq'].float(),
                                state['step'], beta1, beta2, lr, eps)
                    else:
                        adam_cuda.adam_upd(
                                param.float(), param.grad.float(), state['exp_avg'].float(), state['exp_avg_sq'].float(),
                                state['step'], beta1, beta2, lr, eps)

def create_optimizer_or_freeze_model(model, cfg_train, logfile, global_step):
    cfg_train = cfg_train.__dict__
    decay_steps = cfg_train['n_iters']
    decay_factor = 0.1 ** (global_step/decay_steps)
    param_group = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]
        if not hasattr(model, k):
            continue
        param = getattr(model, k)
        if param is None:
            with open(logfile, "a") as f: 
                f.write(f'create_optimizer_or_freeze_model: param {k} not exist \n')
            continue
        lr = cfg_train[f'lrate_{k}'] * decay_factor
        if lr > 0:
            with open(logfile, "a") as f: 
                f.write(f'create_optimizer_or_freeze_model: param {k} lr {lr} \n')
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append({'params_name': k, 'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train['skip_zero_grad'])})
        else:
            with open(logfile, "a") as f: 
                f.write(f'create_optimizer_or_freeze_model: param {k} freeze \n')
            param.requires_grad = False
    return MaskedAdam(param_group)

def farthest_point_sample(xyz, npoint): 
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    xyz = xyz.transpose(2,1)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)     
    distance = torch.ones(B, N).to(device) * 1e10                       
    batch_indices = torch.arange(B, dtype=torch.long).to(device)        
    barycenter = torch.sum((xyz), 1)                                    
    barycenter = barycenter/xyz.shape[1]
    barycenter = barycenter.view(B, 1, 3)
    dist = torch.sum((xyz - barycenter) ** 2, -1)
    farthest = torch.max(dist,1)[1]                                     
    for i in range(npoint):
        centroids[:, i] = farthest                                      
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)        
        dist = torch.sum((xyz - centroid) ** 2, -1)                     
        mask = dist < distance
        distance[mask] = dist[mask]                                     
        farthest = torch.max(distance, -1)[1]                           
    return centroids

def rotat_from_6d(ortho6d):
    def normalize_vector( v, return_mag =False):
        batch=v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))# batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
        v = v/v_mag
        if(return_mag==True): return v, v_mag[:,0]
        else: return v
    
    def cross_product( u, v):
        batch = u.shape[0]
        i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
        j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
        k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        return out

    x_raw = ortho6d[:,0:3]#batch*3  100
    y_raw = ortho6d[:,3:6]#batch*3
    x = normalize_vector(x_raw) #batch*3  100
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z)#batch*3
    y = cross_product(z,x)#batch*3
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

def plot_3d_grads(pc, grads, save_path, save_name, save_types):
    '''
    Parameters:
        grads: [resolution, resolution, resolution, 3] the sdf of the space
        cut_axis: 'x', 'y' or 'z' 
        cut_idx: the coordinate of cut axis
        save_path: the output path
        save_name: the name of output file
        save_types: the types of output file
    Return:
        files of plot of 2d cut of 3d isosurface
    '''
    data = go.Cone( x=pc[:,0],
                    y=pc[:,1],
                    z=pc[:,2],
                    u=grads[:,0],
                    v=grads[:,1],
                    w=grads[:,2],
                    colorscale='Blues',
                    sizemode="absolute",
                    sizeref=2)

    layout = go.Layout(height=1200, width=1200)
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(scene_aspectmode='data')
    for save_type in save_types:
        save_file_path = save_path + save_name + '.' + save_type
        if save_type == 'html': 
            fig.write_html(save_file_path)
        elif save_type == 'png' or 'jpg':
            fig.write_image(save_file_path)
        elif save_type == 'pdf': 
            fig.write_image(save_file_path)

def export_pc(points, colors, filename):
    num_points = points.shape[0]
    if colors is not None:
        vertices = np.empty(num_points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        vertices['x'] = points[:,0].astype('f4')
        vertices['y'] = points[:,1].astype('f4')
        vertices['z'] = points[:,2].astype('f4')
        vertices['red'] = colors[:,0].astype('u1')
        vertices['green'] = colors[:,1].astype('u1')
        vertices['blue'] = colors[:,2].astype('u1')
        ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
        ply.write(filename)
    else:
        vertices = np.empty(num_points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertices['x'] = points[:,0].astype('f4')
        vertices['y'] = points[:,1].astype('f4')
        vertices['z'] = points[:,2].astype('f4')
        ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
        ply.write(filename)

def generate_ncolors(num):
    def get_n_hls_colors(num):
        hls_colors = []
        i = 0
        step = 360.0 / num
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
            _hlsc = [h / 360.0, l / 100.0, s / 100.0]
            hls_colors.append(_hlsc)
            i += step
        return hls_colors
    rgb_colors = np.zeros((0,3))
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors = np.concatenate((rgb_colors,np.array([r,g,b])[np.newaxis,:]))
    return rgb_colors


def vis_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
        bg_mask = ((x < mi)+(x > ma))
    else:
        # mi,ma = minmax
        # bg_mask = ((x < mi)+(x > ma))
        try:
            mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        except:
            mi = 0
        ma = np.max(x)
        bg_mask = ((x < mi)+(x > ma))

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = x * (1-bg_mask)
    x = (255*x).astype(np.uint8)
    x_ = (255 - x)[:,:,None].repeat(3,axis=-1)

    # x = x[]
    # x_ = cv2.applyColorMap(x, cmap)

    # all_white = np.ones_like(x_) * 255
    # x_ = x_ * (1-bg_mask)[:,:,None] + all_white * bg_mask[:,:,None]
    x_ = x_.astype(np.uint8)
    # x_ = cv2.cvtColor(x_, cv2.COLOR_BGR2RGB)
    return x_, [mi,ma]


def sdf2ply(pytorch_3d_sdf_tensor, ply_filename_out, bbox,level=0.5, offset=None, scale=None,):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=level, spacing=voxel_size)
    except: return
    faces = faces[...,::-1] # inverse face orientation
    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]
    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset
    # try writing to the ply file
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]
    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])
    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])
    el_verts = PlyElement.describe(verts_tuple, "vertex")
    el_faces = PlyElement.describe(faces_tuple, "face")
    ply_data = PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

def plot_3d_isosurface(voxel, isomin, isomax, surface_count, opacity, save_path, save_name, save_types):
    '''
    Parameters:
        voxel: [resolution,resolution,resolution] the sdf of the space
        isomin: The displayed lowest isosurface
        isomax: The displayed higest isosurface
        surface_count: The number of displayed isosurface
        opacity: the opacity of surface
        save_path: the output path
        save_name: the name of output file
        save_types: the types of output file
    Return:
        files of plot of 3d isosurface
    '''
    resolution = voxel.shape[0]
    X, Y, Z = np.mgrid[-1:1:complex(0,resolution), -1:1:complex(0,resolution), -1:1:complex(0,resolution)]
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    #######################################################################################
    # values = voxel.swapaxes(1,2).flatten()
    values = voxel.flatten()

    data = go.Isosurface(   x=X,
                            y=Y,
                            z=Z,
                            value=values,
                            opacity = opacity,
                            isomin=isomin,
                            isomax=isomax,
                            surface_count=surface_count,   # number of isosurfaces, 2 by default: only min and max
                            colorbar_nticks=surface_count, # colorbar ticks correspond to isosurface values
                            caps=dict(x_show=False, y_show=False),
                            showlegend = False)

    layout = go.Layout( width = 1200, height = 1200, scene =  {
            'xaxis':{'range': [-1, 1],  'visible':  False,  'showbackground': False,  'autorange': False,  'showgrid': False,
                    'zeroline': False,  'showline': False,  'dtick': False,           'ticks': '',         'showticklabels': False},
            'yaxis':{'range': [-1, 1],  'visible':  False,  'showbackground': False,  'autorange': False,  'showgrid': False,
                    'zeroline': False,  'showline': False,  'dtick': False,           'ticks': '',         'showticklabels': False},
            'zaxis':{'range': [-1, 1],  'visible':  False,  'showbackground': False,  'autorange': False,  'showgrid': False,
                    'zeroline': False,  'showline': False,  'dtick': False,           'ticks': '',         'showticklabels': False},
            'aspectratio':{'x':1, 'y':1, 'z':1}})
    
    fig = go.Figure(data=data, layout=layout)
    
    for save_type in save_types:
        save_file_path = save_path + save_name + '.' + save_type
        if save_type == 'html': 
            fig.write_html(save_file_path)
        elif save_type == 'png' or 'jpg':
            fig.write_image(save_file_path, width=1920, height=1080)
        elif save_type == 'pdf': 
            fig.write_image(save_file_path)
    
    return data


def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim



def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])
    """
    xs: torch.Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: torch.Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2


def compute_iou(dataset_path, pred_path, save_path, logfile):
    num_data = 120
    dataname = 'kurbic' + dataset_path.split('/')[-2][6:]
    if 'noshadow' in dataname:
        dataname = dataname[:-9]
    gt_path = '/data12T/ykz0923/Exp/multi_view_part/TNeRF_cluster_3_6D_9_6_kubric2/data/nerf_synthetic_time/kubric_gt/' + dataname + '/'
    os.makedirs(save_path, exist_ok=True)
    all_pred_seg_mask, all_pred_image, all_gt_seg_mask, all_gt_image = [],[],[],[]
    for idx_image in range(1, num_data):
        pred_seg_mask = np.array(Image.open(pred_path + "%03d" % idx_image + '_part-1_seg_mask.png'))
        pred_seg      = np.array(Image.open(pred_path + "%03d" % idx_image + '_part-1_seg.png'))
        pred_image    = np.array(Image.open(pred_path + "%03d" % idx_image + '_part-1.png'))
        
        gt_seg_mask   = np.array(Image.open(gt_path + 'segmentation_' + "%05d" % (idx_image-1) + '.png'))
        gt_image      = np.array(Image.open(gt_path + 'rgba_' + "%05d" % (idx_image-1) + '.png'))[:,:,:3]
        
        all_pred_seg_mask += [pred_seg_mask]
        all_pred_image    += [pred_image]
        all_gt_seg_mask   += [gt_seg_mask]
        all_gt_image      += [gt_image]
        
    all_pred_seg_mask = np.stack(all_pred_seg_mask, axis=0)
    all_pred_image    = np.stack(all_pred_image,    axis=0)
    all_gt_seg_mask   = np.stack(all_gt_seg_mask,   axis=0)
    all_gt_image      = np.stack(all_gt_image,      axis=0)
    # match across all dataset
    match_list = []
    all_pred_label = np.unique(all_pred_seg_mask[:10])
    for idx_pred_label in range(all_pred_label.shape[0]):
        pred_label = all_pred_label[idx_pred_label]
        pred_label_where = np.where(all_pred_seg_mask[:10].reshape(-1)==pred_label)[0]
        correspond_gt_label = all_gt_seg_mask.reshape(-1)[pred_label_where]
        unique, count = np.unique(correspond_gt_label, return_counts=True)
        match_gt_label = unique[np.argmax(count)]
        match_list.append([pred_label, match_gt_label])
        
    mapped_seg_mask = []
    for match in match_list:
        mapped_seg_mask.append((all_pred_seg_mask == match[0])*1*match[1])
    mapped_seg_mask = np.stack(mapped_seg_mask,axis=-1).sum(-1)
    mapped_seg_mask = (mapped_seg_mask).astype(np.uint8)
    all_gt_seg_mask = (all_gt_seg_mask).astype(np.uint8)
    
    mious = []
    all_gt_label = np.unique(all_gt_seg_mask)
    with open(logfile, "a") as f:
        for idx_gt_label in range(all_gt_label.shape[0]):
            if idx_gt_label != 0:
                pred_mask = (mapped_seg_mask == idx_gt_label)
                gt_mask   = (all_gt_seg_mask == idx_gt_label)
                intersect = (pred_mask * gt_mask).sum((1,2))
                union     = (pred_mask + gt_mask).sum((1,2))
                ious      = intersect[union > 0] / union[union > 0]
                ious      = ious.mean()
                mious.append(ious)
                f.write(str(idx_gt_label) + ': ' + str(ious) + '\n')
        mious = np.array(mious).mean()
        f.write('miou' + ': ' + str(mious) + '\n')
               
    ncolors = np.array([[  0.,   0.,   0.],
                        [247.,  57.,  57.],
                        [252., 252.,  25.],
                        [ 27., 244.,  27.],
                        [ 52., 249., 249.],
                        [ 38.,  38., 244.],
                        [251.,  31., 251.]])
    
    mapped_seg_mask = ncolors[mapped_seg_mask].astype(np.uint8)
    all_gt_seg_mask = ncolors[all_gt_seg_mask].astype(np.uint8)

    for idx_image in range(mapped_seg_mask.shape[0]):
        imageio.imwrite(f'{save_path}/{idx_image:03d}.png', 
                        np.concatenate((all_gt_image[idx_image], all_gt_seg_mask[idx_image], mapped_seg_mask[idx_image]), axis=1))
        