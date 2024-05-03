from operator import le
import os, time, imageio, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from torch.nn import init
from torch.utils.cpp_extension import load
from torch_scatter import segment_coo
from turtle import forward
from PIL import Image
import cv2

from utils import *

chunk_size = 4096

class groupvit(nn.Module):
    def __init__(self, num_slots, dim_feat, tau = 1, softhard='soft', xyz = False):
        super().__init__()
        self.num_slots = num_slots
        self.xyz = xyz
        self.softhard = softhard
        self.tau = tau
        self.dim_feat = dim_feat
        dim = dim_feat
        self.slots_feat_forw = nn.Parameter(torch.randn(1, num_slots, dim_feat))
        if self.xyz:  
            dim += 3
            self.slots_xyz = nn.Parameter(torch.randn(1, num_slots, 3))

        self.gumbel_dist = torch.distributions.gumbel.Gumbel(
                            torch.tensor(0., device=self.slots_feat_forw.device, dtype=self.slots_feat_forw.dtype),
                            torch.tensor(1., device=self.slots_feat_forw.device, dtype=self.slots_feat_forw.dtype))

        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)

    def forward(self, inputs, training = True):
        query = inputs['feat_forw']
        slots = self.slots_feat_forw
        if self.xyz:
            query = torch.cat((query, inputs['xyz']) ,dim=-1)
            slots = torch.cat((slots, self.slots_xyz),dim=-1)
        query = query.unsqueeze(0)
        query = self.norm_input(query)
        slots = self.norm_slots(slots)

        q = self.to_q(slots)
        k = self.to_k(query)
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        if training:
            gumbels = self.gumbel_dist.sample(dots.shape)
            gumbels = (dots + gumbels) / self.tau  # ~Gumbel(logits,tau)
        else:
            gumbels = dots
        attn = gumbels.softmax(dim=1)

        if self.softhard == 'soft':
            feat = attn.permute(2,1,0) * inputs['feat_forw'].unsqueeze(1).repeat(1,self.num_slots,1)
            mean_slots = feat.sum(0) / (attn.sum(-1).permute(1,0) + 1)
            feat = torch.einsum('ij,jk->ik', attn[0].permute(1,0), mean_slots)
            return feat, attn, attn, mean_slots
        elif self.softhard == 'hard':
            index = attn.max(dim=1, keepdim=True)[1]
            y_hard = torch.zeros_like(attn, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
            attn_hard = y_hard - attn.detach() + attn
            feat = attn_hard.permute(2,1,0) * inputs['feat_forw'].unsqueeze(1).repeat(1,self.num_slots,1)
            mean_slots = feat.sum(0) / (attn_hard.sum(-1).permute(1,0) + 1)
            feat = torch.einsum('ij,jk->ik', attn_hard[0].permute(1,0), mean_slots)
            return feat, attn_hard, dots.softmax(dim=1), mean_slots

class time_nerf(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max, near_far, num_voxels_feat, num_voxels_motn, logfile, device, args):
        super(time_nerf, self).__init__()
        self.logfile  = logfile
        self.device   = device
        self.near_far = near_far
        self.step_ratio  = args.step_ratio
        self.alpha_init  = args.alpha_init
        self.color_thre  = args.color_thre
        self.n_iter_slot = args.n_iter_slot
        self.act_shift = np.log(1/(1-self.alpha_init) - 1)

        self.freqs_time = args.freqs_time
        self.freqs_posi = args.freqs_posi
        self.freqs_view = args.freqs_view
        self.freqs_grid = args.freqs_grid

        self.width_mlp  = args.width_mlp
        self.width_motn = args.width_motn

        self.width_backgrid = args.width_backgrid
        self.width_backnet  = args.width_backnet
        self.layer_backnet  = args.layer_backnet

        self.width_forwgrid = args.width_forwgrid
        self.width_forwnet  = args.width_forwnet
        self.layer_forwnet  = args.layer_forwnet

        self.width_featgrid = args.width_featgrid
        self.width_featnet  = args.width_featnet
        self.layer_featnet  = args.layer_featnet
        self.layer_sigmanet = args.layer_sigmanet
        self.layer_colornet = args.layer_colornet

        self.slot_name = args.slot_name
        self.slot_num  = args.slot_num
        self.slot_hard = args.slot_hard

        self.ncolors = torch.from_numpy(generate_ncolors(self.slot_num)).to(self.device) / 255

        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.set_grid_resolution(num_voxels_feat, 'feat')
        self.set_grid_resolution(num_voxels_motn, 'motn')

        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(self.freqs_time)]))
        self.register_buffer('grid_poc', torch.FloatTensor([(2**i) for i in range(self.freqs_grid)]))
        self.register_buffer('posi_poc', torch.FloatTensor([(2**i) for i in range(self.freqs_posi)]))
        self.register_buffer('view_poc', torch.FloatTensor([(2**i) for i in range(self.freqs_view)]))

        self.backgrid  = torch.nn.Parameter(torch.zeros([1, self.width_backgrid, *self.world_size_motn],dtype=torch.float32))
        self.forwgrid  = torch.nn.Parameter(torch.zeros([1, self.width_forwgrid, *self.world_size_motn],dtype=torch.float32))
        self.featgrid  = torch.nn.Parameter(torch.zeros([1, self.width_featgrid, *self.world_size_feat],dtype=torch.float32))
    
        backnet_ch = 2*self.freqs_time+1+self.width_backgrid
        self.backnet  = self.create_layer(backnet_ch, self.width_backnet, self.layer_backnet, self.width_motn)

        forwnet_ch = 2*self.freqs_time+1+self.width_forwgrid
        self.forwnet  = self.create_layer(forwnet_ch, self.width_forwnet, self.layer_forwnet, self.width_motn)

        self.groupnet = eval(self.slot_name.split('_')[0])(num_slots    = self.slot_num, 
                                                           dim_feat     = self.width_forwgrid, 
                                                           softhard     = self.slot_hard, 
                                                           xyz = 'xyz' in self.slot_name)

        self.rotatnet = self.create_layer(self.width_motn, 128, 1, 6)
        torch.nn.init.constant_(self.rotatnet[-1].weight, 0)
        self.rotatnet[-1].bias.data = torch.Tensor([1, 0, 0, 0, 1, 0])

        self.transnet = self.create_layer(self.width_motn, 128, 1, 3)
        torch.nn.init.constant_(self.transnet[-1].weight, 0)
        torch.nn.init.constant_(self.transnet[-1].bias, 0)

        featnet_ch = (2*self.freqs_posi+1)*3 + (2*self.freqs_grid+1)*self.width_featgrid*3
        self.featnet = self.create_layer(featnet_ch, self.width_mlp, self.layer_featnet, self.width_featnet)

        self.sigmanet  = self.create_layer(self.width_featnet, self.width_mlp, self.layer_sigmanet, 1)

        self.colornet1 = self.create_layer(self.width_featnet, self.width_mlp, 1, self.width_mlp)
        if self.freqs_view == 0:
            colornet2_ch = self.width_mlp
        else:
            colornet2_ch = self.width_mlp + (2*self.freqs_view+1)*3
        self.colornet2 = self.create_layer(colornet2_ch, self.width_mlp, self.layer_colornet, 3)

        with open(self.logfile, "a") as f:
            f.write('backgrid  ' + str(self.backgrid.shape) + '\n')
            f.write('forwgrid  ' + str(self.forwgrid.shape) + '\n')
            f.write('featgrid  ' + str(self.featgrid.shape) + '\n')

            f.write('backnet   ' + str(self.backnet) + '\n')
            f.write('forwnet   ' + str(self.forwnet) + '\n')
            f.write('rotatnet  ' + str(self.rotatnet)  + '\n')
            f.write('transnet  ' + str(self.transnet)  + '\n')
 
            f.write('featnet   ' + str(self.featnet)   + '\n')
            f.write('sigmanet  ' + str(self.sigmanet)  + '\n')
            f.write('colornet1 ' + str(self.colornet1) + '\n')
            f.write('colornet2 ' + str(self.colornet2) + '\n')
            f.write('groupnet  ' + str(self.groupnet)  + '\n')

    def create_layer(self, dimin, width, layers, dimout):
        if layers == 1:
            return  nn.Sequential(nn.Linear(dimin, dimout))
        else:
            return  nn.Sequential(
                    nn.Linear(dimin, width), nn.ReLU(inplace = True),
                    *[nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace = True)) for _ in range(layers-2)],
                    nn.Linear(width, dimout),) 

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        torch.save(ckpt, path)

    def load(self, ckpt):
        self.load_state_dict(ckpt['state_dict'])
        self.ncolors = torch.from_numpy(generate_ncolors(self.slot_num)).to(self.device) / 255

    def set_grid_resolution(self, num_voxels, grid_type):
        # Determine grid resolution
        if grid_type == 'feat':
            self.num_voxels_feat = num_voxels
            self.voxel_size_feat = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
            self.world_size_feat = ((self.xyz_max - self.xyz_min)/self.voxel_size_feat).long()
            with open(self.logfile, "a") as f:
                f.write('feature voxel_size      ' + str(self.voxel_size_feat) + '\n')
                f.write('feature world_size      ' + str(self.world_size_feat) + '\n')
        elif grid_type == 'motn':
            self.num_voxels_motn = num_voxels
            self.voxel_size_motn = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
            self.world_size_motn = ((self.xyz_max - self.xyz_min)/self.voxel_size_motn).long()
            with open(self.logfile, "a") as f:
                f.write('motion  voxel_size      ' + str(self.voxel_size_motn) + '\n')
                f.write('motion  world_size      ' + str(self.world_size_motn) + '\n')
    
    @torch.no_grad()
    def scale_volume_grid(self, num_voxels, grid_type):
        if grid_type == 'feat':
            self.set_grid_resolution(num_voxels, grid_type)
            self.featgrid = torch.nn.Parameter(
                F.interpolate(self.featgrid.data, size=tuple(self.world_size_feat), mode='trilinear', align_corners=True))
        if grid_type == 'motn':
            self.set_grid_resolution(num_voxels, grid_type)
            self.backgrid = torch.nn.Parameter(
                F.interpolate(self.backgrid.data, size=tuple(self.world_size_motn), mode='trilinear', align_corners=True))
            self.forwgrid = torch.nn.Parameter(
                F.interpolate(self.forwgrid.data, size=tuple(self.world_size_motn), mode='trilinear', align_corners=True))

    def get_kwargs(self):
        return {
            'logfile':     self.logfile,
            'near_far':    self.near_far,
            'step_ratio':  self.step_ratio,
            'alpha_init':  self.alpha_init,
            'color_thre':  self.color_thre,
            'n_iter_slot': self.n_iter_slot,
            'act_shift':   self.act_shift,
            
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels_feat': self.num_voxels_feat,
            'num_voxels_motn': self.num_voxels_motn,

            'freqs_time': self.freqs_time,
            'freqs_posi': self.freqs_posi,
            'freqs_view': self.freqs_view,
            'freqs_grid': self.freqs_grid,

            'width_mlp' : self.width_mlp,
            'width_motn': self.width_motn,

            'width_backgrid': self.width_backgrid,
            'width_backnet' : self.width_backnet,
            'layer_backnet' : self.layer_backnet,

            'width_forwgrid': self.width_forwgrid,
            'width_forwnet' : self.width_forwnet,
            'layer_forwnet' : self.layer_forwnet,

            'width_featgrid': self.width_featgrid,
            'width_featnet' : self.width_featnet,
            'layer_featnet' : self.layer_featnet,
            'layer_sigmanet': self.layer_sigmanet,
            'layer_colornet': self.layer_colornet,

            'slot_name' : self.slot_name,
            'slot_num'  : self.slot_num,
            'slot_hard' : self.slot_hard}

    def sample_ray(self, rays_o, rays_d):
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = self.step_ratio * self.voxel_size_feat
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, self.near_far[0], self.near_far[1], stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id, mask_inbbox

    def get_mask(self, rays_o, rays_d):
        '''Check whether the rays hit the geometry or not'''
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = self.step_ratio * self.voxel_size_feat
        ray_pts, mask_outbbox, ray_id = render_cuda.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, self.near_far[0], self.near_far[1], stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox]] = 1
        return hit

    def poc_fre(self, input_data, poc_buf):
        input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
        input_data_sin = input_data_emb.sin()
        input_data_cos = input_data_emb.cos()
        input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
        return input_data_emb

    def normalize_xyz(self, xyz_sampled):
        return ((xyz_sampled - self.xyz_min) / (self.xyz_max - self.xyz_min)) * 2.0 - 1

    def denormalize_xyz(self, xyz_sampled):
        return (xyz_sampled + 1) / 2.0 * (self.xyz_max - self.xyz_min) + self.xyz_min

    def back_deform(self, xyz_smp, times):
        timefeat = self.poc_fre(times, self.time_poc)
        feat_back = F.grid_sample(self.backgrid, xyz_smp[None,None,None,:,:], mode='bilinear', align_corners=True)
        feat_back = feat_back[0,:,0,0,:].permute(1,0)
        motn_back = nn.ReLU(inplace = True)(self.backnet(torch.cat([feat_back, timefeat], dim=-1)))

        rotat_back = rotat_from_6d(self.rotatnet(motn_back))
        trans_back = self.transnet(motn_back)

        xyz_smp = xyz_smp - trans_back
        xyz_cnc = torch.einsum('bcd,bde->bce', rotat_back, xyz_smp.unsqueeze(-1)).squeeze(-1)
        deform_back = xyz_smp - xyz_cnc

        return xyz_cnc, motn_back, rotat_back, trans_back, deform_back

    def forw_deform(self, xyz_cnc, times, iteration=-1, training=True):
        timefeat = self.poc_fre(times, self.time_poc)

        feat_forw = F.grid_sample(self.forwgrid, xyz_cnc[None,None,None,:,:], mode='bilinear', align_corners=True)
        feat_forw = feat_forw[0,:,0,0,:].permute(1,0)
        if iteration == -1 or iteration > self.n_iter_slot:
            feat_in = {'feat_forw':feat_forw, 'xyz':xyz_cnc}
            feat_forw, attn_hard, attn_soft, mean_slots = self.groupnet(feat_in, training=training)
            attn_hard = attn_hard[0].permute(1,0)
            attn_soft = attn_soft[0].permute(1,0)
        else:
            attn_hard  = None
            attn_soft  = None
            mean_slots = None

        motn_forw = nn.ReLU(inplace = True)(self.forwnet(torch.cat([feat_forw, timefeat], dim=-1)))
        rotat_forw = rotat_from_6d(self.rotatnet(motn_forw))
        trans_forw = self.transnet(motn_forw)

        xyz_smp_pred = torch.einsum('bcd,bde->bce', rotat_forw.permute(0,2,1), xyz_cnc.unsqueeze(-1)).squeeze(-1)
        xyz_smp_pred = xyz_smp_pred + trans_forw
        deform_forw = xyz_smp_pred - xyz_cnc

        return attn_hard, attn_soft, mean_slots, xyz_smp_pred, motn_forw, rotat_forw, trans_forw, deform_forw

    def feature_compute(self, xyz_cnc):
        feat_l = F.grid_sample(self.featgrid,                  xyz_cnc[None,None,None,:,:], mode='bilinear', align_corners=True)
        feat_m = F.grid_sample(self.featgrid[:,:,::2,::2,::2], xyz_cnc[None,None,None,:,:], mode='bilinear', align_corners=True)
        feat_s = F.grid_sample(self.featgrid[:,:,::4,::4,::4], xyz_cnc[None,None,None,:,:], mode='bilinear', align_corners=True)
        feat = torch.cat([feat_l,feat_m,feat_s], dim=1)[0,:,0,0,:].permute(1,0)
        net_in  = torch.cat([self.poc_fre(feat, self.grid_poc), self.poc_fre(xyz_cnc, self.posi_poc)], -1)
        feature = nn.ReLU(inplace = True)(self.featnet(net_in))
        return feature

    def sigma_compute(self, feature):
        return self.sigmanet(feature)

    def color_compute(self, feature, viewdir):
        if self.freqs_view == 0:
            net_in = nn.ReLU(inplace = True)(self.colornet1(feature))
            color = self.colornet2(net_in)
        else:
            net_in = [nn.ReLU(inplace = True)(self.colornet1(feature)), self.poc_fre(viewdir, self.view_poc)]
            color = self.colornet2(torch.cat(net_in, dim=-1))
        return color

    def add_tv_grad(self, weight_feat, weight_forw, weight_back, dense_mode):
        weight_feat = weight_feat * self.world_size_feat.max() / 128
        weight_forw = weight_forw * self.world_size_motn.max() / 128
        weight_back = weight_back * self.world_size_motn.max() / 128
        if self.featgrid.grad is not None:
            tv_cuda.total_variation_add_grad(
                    self.featgrid.float(), self.featgrid.grad.float(), weight_feat, weight_feat, weight_feat, dense_mode)
        if self.backgrid.grad is not None:
            tv_cuda.total_variation_add_grad(
                    self.backgrid.float(), self.backgrid.grad.float(), weight_back, weight_back, weight_back, dense_mode)
        if self.forwgrid.grad is not None:
            tv_cuda.total_variation_add_grad(
                self.forwgrid.float(), self.forwgrid.grad.float(), weight_forw, weight_forw, weight_forw, dense_mode)
        
    def merge_group(self, cluster_idx, group=None):
        if group != None:
            cluster_idx_new = -torch.ones_like(cluster_idx)
            for idx_group in range(len(group)):
                for cluster_ori in group[idx_group]:
                    cluster_idx_new[cluster_idx == cluster_ori] = idx_group
            cluster_idx_new = cluster_idx_new * (self.slot_num // len(group))
        else:
            cluster_idx_new = cluster_idx
        return cluster_idx_new

    def compute_depth(self, weights, rays_id, step_id, max_step, chunk_size, alphainv_last, white_bg=True):
        depth_point = step_id
        depth_map = segment_coo(src=(weights*depth_point), index=rays_id, out=torch.zeros([chunk_size]), reduce='sum')
        depth_map[alphainv_last >= 1] = 0
        return depth_map
    
    def compute_seg(self, attn_hard, weights, group, alphainv_last, rays_id, chunk_size, white_bg=True):
        cluster_idx = torch.max(attn_hard, dim=1)[1]
        cluster_idx = self.merge_group(cluster_idx, group)
        seg_color = self.ncolors[cluster_idx].to(torch.float32)
        seg_map = segment_coo(src=(weights.unsqueeze(-1)*seg_color), index=rays_id, out=torch.zeros([chunk_size, 3]), reduce='sum')
        seg_map += (alphainv_last.unsqueeze(-1) * white_bg)
        return seg_map

    def compute_seg_mask(self, attn_hard, weights, group, alphainv_last, rays_id, chunk_size, white_bg=True):
        if group is not None:
            grouped_ncolors = self.ncolors[::((self.slot_num-1) // (len(group)-1)),:]
            grouped_ncolors = torch.cat([torch.zeros(1,3), grouped_ncolors], dim=0)
            num_group = len(group) + 1
        else:
            grouped_ncolors = self.ncolors
            grouped_ncolors = torch.cat([torch.zeros(1,3), grouped_ncolors], dim=0)
            num_group = self.slot_num + 1

        cluster_idx = torch.max(attn_hard, dim=1)[1]
        cluster_idx = self.merge_group(cluster_idx, group)
        seg_onehot = torch.eye(num_group)[cluster_idx+1].to(torch.float32)
        seg_map = segment_coo(src=(weights.unsqueeze(-1)*seg_onehot), index=rays_id, out=torch.zeros([chunk_size, num_group]), reduce='sum')
        seg_map = seg_map.max(-1)[1]
        return seg_map

    @torch.no_grad()
    def evaluation(self, dataset, group=None, idx_part=-1, save_path=None, logfile=None, prtx='', N_vis=-1):
        if logfile is not None:
            with open(logfile, "a") as f:
                f.write('=' * 30 + '\n')
                f.write('start evaluation ' + '\n')
        PSNRs, SSIMs = [], []
        os.makedirs(save_path, exist_ok=True)
        img_eval_interval = 1 if N_vis < 0 else max(dataset.__len__() // N_vis,1)
        idxs = list(range(0, dataset.__len__(), img_eval_interval)) 
        for idx_image in idxs:
            rays_o, rays_d, viewdir = dataset.get_rays_one_view(idx_image)
            times = torch.ones(rays_o.shape[0],1) * dataset.all_time[idx_image]

            rgb_map, seg_map, depth_map, seg_mask = [], [], [], []
            for chunk_idx in range(rays_o.shape[0]//chunk_size + int(rays_o.shape[0] % chunk_size > 0)):
                cur_times   = times[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)
                cur_rays_o  = rays_o[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)
                cur_rays_d  = rays_d[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)
                cur_viewdir = viewdir[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)
                retdict = self.forward(cur_times, cur_rays_o, cur_rays_d, cur_viewdir, idx_part=idx_part, is_train=False, group = group)

                rgb_map   += [retdict['rgbs']]
                depth_map += [self.compute_depth(retdict['weights'], retdict['rays_id'], retdict['step_id'], retdict['max_step'], cur_times.shape[0], retdict['alphainv_last'])]
                seg_map   += [self.compute_seg(retdict['attn_hard'], retdict['weights'], group, retdict['alphainv_last'], retdict['rays_id'], cur_times.shape[0])]
                seg_mask  += [self.compute_seg_mask(retdict['attn_hard'], retdict['weights'], group, retdict['alphainv_last'], retdict['rays_id'], cur_times.shape[0])]

            rgb_map, depth_map = torch.cat(rgb_map), torch.cat(depth_map)
            rgb_map    = rgb_map.clamp(0.0, 1.0).reshape(dataset.H, dataset.W, 3).cpu()
            rgb_map_gt = dataset.all_rgbs[idx_image].reshape(dataset.H, dataset.W, 3).cpu()
            depth_map  = vis_depth(depth_map.reshape(dataset.H, dataset.W).cpu().numpy(),self.near_far)[0]
            rgbd_map   = np.concatenate(((rgb_map.numpy() * 255).astype('uint8'), depth_map), axis=1)
            imageio.imwrite(f'{save_path}/{prtx}{idx_image:03d}_part{idx_part:02d}.png',     rgbd_map)

            seg_map = torch.cat(seg_map)
            seg_map = (seg_map.clamp(0.0, 1.0).reshape(dataset.H, dataset.W,3).cpu().numpy() * 255).astype('uint8')
            imageio.imwrite(f'{save_path}/{prtx}{idx_image:03d}_part{idx_part:02d}_seg.png', seg_map)  

            seg_mask = torch.cat(seg_mask)
            seg_mask = (seg_mask.reshape(dataset.H, dataset.W,1).cpu().numpy()).astype('uint8')
            imageio.imwrite(f'{save_path}/{prtx}{idx_image:03d}_part{idx_part:02d}_seg_mask.png', seg_mask)  

            PSNRs.append(-10.0 * np.log(torch.mean((rgb_map - rgb_map_gt) ** 2).item()) / np.log(10.0))
            SSIMs.append(rgb_ssim(rgb_map, rgb_map_gt, 1))

        psnr = np.mean(np.asarray(PSNRs))
        ssim = np.mean(np.asarray(SSIMs))
        if logfile is not None:
            with open(logfile, "a") as f:
                f.write(dataset.split + ' psnr ' + str(psnr) + '\n')
                f.write(dataset.split + ' ssim ' + str(ssim) + '\n')
                f.write('end evaluation ' + '\n')
                f.write('=' * 30 + '\n')
        return psnr, ssim
        
    @torch.no_grad()
    def render_path(self, dataset, group=None, idx_part=-1, save_path=None, idxs_image=None):
        os.makedirs(save_path, exist_ok=True)
        time_list = list(np.array(list(range(0,16)))/15)
        for idx_image in idxs_image:
            all_rgbd_map, all_seg_map = [], []
            rays_o, rays_d, viewdir = dataset.get_rays_one_view(idx_image)
            for idx_time in range(len(time_list)):
                rgb_map, seg_map, depth_map = [], [], []
                times = torch.ones(rays_o.shape[0],1) * time_list[idx_time]
                for chunk_idx in range(rays_o.shape[0]//chunk_size + int(rays_o.shape[0] % chunk_size > 0)):
                    cur_times   = times[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)
                    cur_rays_o  = rays_o[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)
                    cur_rays_d  = rays_d[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)
                    cur_viewdir = viewdir[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)
                    retdict = self.forward(cur_times, cur_rays_o, cur_rays_d, cur_viewdir, idx_part=idx_part, is_train=False, group =group)
                    
                    rgb_map   += [retdict['rgbs']]
                    seg_map   += [self.compute_seg(retdict['attn_hard'], retdict['weights'], group, retdict['alphainv_last'], retdict['rays_id'], cur_times.shape[0])]
                    depth_map += [self.compute_depth(retdict['weights'], retdict['rays_id'], retdict['step_id'], retdict['max_step'], cur_times.shape[0], retdict['alphainv_last'])]

                rgb_map, seg_map, depth_map = torch.cat(rgb_map), torch.cat(seg_map), torch.cat(depth_map)
                rgb_map    = rgb_map.clamp(0.0, 1.0).reshape(dataset.H, dataset.W, 3).cpu()
                depth_map  = vis_depth(depth_map.reshape(dataset.H, dataset.W).cpu().numpy(),self.near_far)[0]
                rgbd_map   = np.concatenate(((rgb_map.numpy() * 255).astype('uint8'), depth_map), axis=1)
                seg_map    = (seg_map.clamp(0.0, 1.0).reshape(dataset.H, dataset.W,3).cpu().numpy() * 255).astype('uint8')
                imageio.imwrite(f'{save_path}/{idx_image:03d}_{time_list[idx_time]:02f}.png',     rgbd_map)
                imageio.imwrite(f'{save_path}/{idx_image:03d}_{time_list[idx_time]:02f}_seg.png', seg_map)  

                all_rgbd_map.append(rgbd_map)
                all_seg_map.append(seg_map)

            imageio.mimwrite(os.path.join(save_path, f'{idx_image:03d}_video.rgbd.mp4'), np.array(all_rgbd_map), fps=30, quality=8)
            imageio.mimwrite(os.path.join(save_path, f'{idx_image:03d}_video.seg.mp4'),  np.array(all_seg_map) , fps=30, quality=8)

    @torch.no_grad()
    def render_video(self, dataset, group=None, idx_part=-1, save_path=None, idxs_image=None):
        trans_t = lambda t : torch.Tensor([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,t],
            [0,0,0,1]]).float()

        rot_phi = lambda phi : torch.Tensor([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1]]).float()

        rot_theta = lambda th : torch.Tensor([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1]]).float()

        def pose_spherical(theta, phi, radius):
            c2w = trans_t(radius)
            c2w = rot_phi(phi/180.*np.pi) @ c2w
            c2w = rot_theta(theta/180.*np.pi) @ c2w
            c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
            return c2w

        os.makedirs(save_path, exist_ok=True)
        static_time = 90
        moving_time = 90
        if 'bouncingballs' in save_path:
            start_angle = 180
            end_angle = 30
        elif 'hellwarrior' in save_path:
            start_angle = 200
            end_angle = 50
        elif 'hook' in save_path:
            start_angle = 220
            end_angle = 120
        elif 'lego' in save_path:
            start_angle = 210
            end_angle = 60
        elif 'mutant' in save_path:
            start_angle = 210
            end_angle = 40
        elif 'trex' in save_path:
            start_angle = 240
            end_angle = 90   
        else:
            start_angle = 180
            end_angle = 30   
         
        render_poses  = [pose_spherical(start_angle, -30.0, 4.0)] * static_time
        render_poses += [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(start_angle,end_angle,moving_time+1)[:-1]]
        render_poses  = torch.stack(render_poses, 0)
        render_times = torch.linspace(0., 1., render_poses.shape[0])
        all_rgbd_map, all_seg_map = [], []
        for idx_image in range(render_poses.shape[0]):
            rgb_map, seg_map, depth_map = [], [], []
            rays_o, rays_d, viewdir = dataset.get_rays_one_view_pose(render_poses[idx_image])
            times = torch.ones(rays_o.shape[0],1) * render_times[idx_image]

            for chunk_idx in range(rays_o.shape[0]//chunk_size + int(rays_o.shape[0] % chunk_size > 0)):
                cur_times   = times[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)
                cur_rays_o  = rays_o[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)
                cur_rays_d  = rays_d[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)
                cur_viewdir = viewdir[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)
                retdict = self.forward(cur_times, cur_rays_o, cur_rays_d, cur_viewdir, idx_part=idx_part, is_train=False, group =group)
                
                rgb_map   += [retdict['rgbs']]
                seg_map   += [self.compute_seg(retdict['attn_hard'], retdict['weights'], group, retdict['alphainv_last'], retdict['rays_id'], cur_times.shape[0])]
                depth_map += [self.compute_depth(retdict['weights'], retdict['rays_id'], retdict['step_id'], retdict['max_step'], cur_times.shape[0], retdict['alphainv_last'])]

            rgb_map, seg_map, depth_map = torch.cat(rgb_map), torch.cat(seg_map), torch.cat(depth_map)
            rgb_map    = rgb_map.clamp(0.0, 1.0).reshape(dataset.H, dataset.W, 3).cpu()
            depth_map  = vis_depth(depth_map.reshape(dataset.H, dataset.W).cpu().numpy(),self.near_far)[0]
            rgbd_map   = np.concatenate(((rgb_map.numpy() * 255).astype('uint8'), depth_map), axis=1)
            seg_map    = (seg_map.clamp(0.0, 1.0).reshape(dataset.H, dataset.W,3).cpu().numpy() * 255).astype('uint8')
            imageio.imwrite(f'{save_path}/{idx_image:03d}.png',     rgbd_map)
            imageio.imwrite(f'{save_path}/{idx_image:03d}_seg.png', seg_map)  

            all_rgbd_map.append(rgbd_map)
            all_seg_map.append(seg_map)

        imageio.mimwrite(os.path.join(save_path, 'video.rgbd.mp4'), np.array(all_rgbd_map), fps=30, quality=8)
        imageio.mimwrite(os.path.join(save_path, 'video.seg.mp4'),  np.array(all_seg_map) , fps=30, quality=8)

    @torch.no_grad()
    def render_edit(self, dataset, group=None, idx_part=-1, save_path=None, idxs_image=None, \
                          edit_type=None, rotat=None, trans=None, scale=None, white_bg=True):
        if   edit_type == 'repose':
            invid_part, remove_part = True,  True
        elif edit_type == 'duplic':
            invid_part, remove_part = True,  False
        elif edit_type == 'remove':
            invid_part, remove_part = False, True
        else:
            print('no implement editing')
            return repose

        os.makedirs(save_path, exist_ok=True)
        time_list = list(np.array(list(range(0,91)))/90)
        for idx_image in idxs_image:
            rays_o, rays_d, viewdir = dataset.get_rays_one_view(idx_image)
            for idx_time in range(len(time_list)):
                rgb_map, depth_map = [], []
                times = torch.ones(rays_o.shape[0],1) * time_list[idx_time]
                for chunk_idx in range(rays_o.shape[0]//chunk_size + int(rays_o.shape[0] % chunk_size > 0)):
                    cur_times   = times[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)
                    cur_rays_o  = rays_o[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)
                    cur_rays_d  = rays_d[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)
                    cur_viewdir = viewdir[chunk_idx*chunk_size:(chunk_idx + 1)*chunk_size].to(self.device)

                    xyz_smp, rays_id, step_id, _ = self.sample_ray(cur_rays_o, cur_rays_d)
                    try:    max_step = step_id.max()
                    except: max_step = 1
                    xyz_smp = self.normalize_xyz(xyz_smp)
                    
                    if invid_part:
                        xyz_smp_part = torch.einsum('cd,bd->bc', rotat, xyz_smp)
                        xyz_smp_part = xyz_smp_part + trans
                        xyz_smp_part = xyz_smp_part / scale
                        xyz_cnc_part, _, _,_,_ = self.back_deform(xyz_smp_part, cur_times[rays_id])
                        attn_hard_part = self.forw_deform(xyz_cnc_part, cur_times[rays_id], iteration=-1, training=False)[0]
                        cluster_part   = torch.max(attn_hard_part, dim=1)[1]
                        cluster_part   = self.merge_group(cluster_part, group)
                        part_mask    =  (cluster_part == idx_part)
                        part_feature = self.feature_compute(xyz_cnc_part)
                        part_density = self.sigma_compute(part_feature)
                        part_rgb_xyz = 1.2/(1+torch.exp(-self.color_compute(part_feature, cur_viewdir[rays_id])))-0.1 
                        part_alphas  = nn.Softplus()(part_density+self.act_shift).squeeze(-1)
                        part_alphas = part_alphas * part_mask
                        part_alphas = part_alphas * (xyz_smp_part.abs().max(1)[0]<=1)
                    
                    xyz_cnc,_,_,_,_ = self.back_deform(xyz_smp, cur_times[rays_id])
                    feature = self.feature_compute(xyz_cnc)
                    rgb_xyz = 1.2/(1+torch.exp(-self.color_compute(feature, cur_viewdir[rays_id])))-0.1 
                    density = self.sigma_compute(feature)
                    alphas  = nn.Softplus()(density+self.act_shift).squeeze(-1)

                    if remove_part:
                        attn_hard = self.forw_deform(xyz_cnc, cur_times[rays_id], iteration=-1, training=False)[0]
                        cluster = torch.max(attn_hard, dim=1)[1]
                        cluster = self.merge_group(cluster, group)
                        nonpart_mask =  (cluster != idx_part)
                        alphas  = alphas * nonpart_mask
                    
                    try:
                        all_alphas  = part_alphas + alphas
                        all_rgb_xyz = (part_alphas[...,None]*part_rgb_xyz + alphas[...,None]*rgb_xyz)/(all_alphas[...,None] + 0.00001)
                    except:
                        all_alphas = alphas
                        all_rgb_xyz = rgb_xyz
                        
                    if self.color_thre > 0:
                        mask        = (all_alphas > self.color_thre)
                        all_alphas  = all_alphas[mask]
                        all_rgb_xyz = all_rgb_xyz[mask]
                        rays_id     = rays_id[mask]
                        step_id     = step_id[mask]

                    weights, alphainv_last = Alphas2Weights.apply(all_alphas, rays_id, cur_times.shape[0])
                    rgbs = segment_coo(src=(weights.unsqueeze(-1)*all_rgb_xyz), index=rays_id, out=torch.zeros([cur_times.shape[0], 3]), reduce='sum')
                    rgbs += (alphainv_last.unsqueeze(-1) * white_bg)
                    rgb_map   += [rgbs]
                    depth_map += [self.compute_depth(weights, rays_id, step_id, max_step, cur_times.shape[0], alphainv_last)]
                    
                rgb_map, depth_map = torch.cat(rgb_map), torch.cat(depth_map)
                rgb_map    = rgb_map.clamp(0.0, 1.0).reshape(dataset.H, dataset.W, 3).cpu()
                depth_map  = vis_depth(depth_map.reshape(dataset.H, dataset.W).cpu().numpy(),self.near_far)[0]
                rgbd_map   = np.concatenate(((rgb_map.numpy() * 255).astype('uint8'), depth_map), axis=1)
                imageio.imwrite(f'{save_path}/{idx_image:03d}_{times[0,0]:02f}.png', rgbd_map)

    @torch.no_grad()
    def export_mesh(self, save_path, resolution=100):
        os.makedirs(save_path, exist_ok=True)
        time_list = [0.0,0.2,0.4,0.6,0.8,1.0]
        aabb = torch.stack((self.xyz_min, self.xyz_max),dim=0)
        xyz_sampled = np.mgrid[-1:1:complex(0,resolution), -1:1:complex(0,resolution), -1:1:complex(0,resolution)]
        xyz_sampled = torch.FloatTensor(xyz_sampled).permute(1,2,3,0).to(self.device)
        # dynamic mesh
        for time in time_list:
            alphas_all = []
            cur_time = (torch.ones((resolution**2,1))*time).to(self.device)
            for idx_x in range(resolution):
                xyz_cnc, _, _, _, _ = self.back_deform(xyz_sampled[idx_x].flatten(0,1), cur_time)
                feature = self.feature_compute(xyz_cnc)
                density = self.sigma_compute(feature)
                alphas  = nn.Softplus()(density+self.act_shift).squeeze(-1)
                alphas_all.append(alphas.reshape(resolution,resolution,-1))
            alphas_all = torch.stack(alphas_all)[...,0]
            sdf2ply(alphas_all.cpu(), save_path + '/' + f'{str(time)}.ply', bbox=aabb.cpu(), level=self.color_thre)
        
        # canconical mesh
        alphas_all = []
        for idx_x in range(resolution):
            feature = self.feature_compute(xyz_sampled[idx_x].flatten(0,1))
            density = self.sigma_compute(feature)
            alphas  = nn.Softplus()(density+self.act_shift).squeeze(-1)
            alphas_all.append(alphas.reshape(resolution,resolution,-1))
        alphas_all = torch.stack(alphas_all)[...,0]
        sdf2ply(alphas_all.cpu(), save_path + '/' + 'cnc.ply', bbox=aabb.cpu(), level=self.color_thre)

        # canconical point cloud
        feature = self.feature_compute(xyz_sampled.reshape(-1,3))
        density = self.sigma_compute(feature)
        alphas  = nn.Softplus()(density+self.act_shift).squeeze(-1)
        mask = (alphas > self.color_thre).reshape(-1)
        xyz_sampled = xyz_sampled.reshape(-1,3)[mask]
        feat_forw = F.grid_sample(self.forwgrid, xyz_sampled[None,None,None,:,:], mode='bilinear', align_corners=True)
        feat_forw = feat_forw[0,:,0,0,:].permute(1,0)
        feat_in = {'feat_forw':feat_forw, 'xyz':xyz_sampled}
        feat_forw, attn_hard, attn_soft, mean_slots = self.groupnet(feat_in, training=False)
        attn_hard = attn_hard[0].permute(1,0)
        cluster_idx = torch.max(attn_hard, dim=1)[1].cpu()
        export_pc(xyz_sampled.detach().cpu().numpy(), self.ncolors[cluster_idx].cpu().numpy()*255, save_path + '/' + 'cnc_seg.ply')  

    @torch.no_grad()
    def export_segment(self, save_path):
        resolution  = 100
        time_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        xyz_cnc_all = np.mgrid[-1:1:complex(0,resolution), -1:1:complex(0,resolution), -1:1:complex(0,resolution)]
        xyz_cnc_all = torch.FloatTensor(xyz_cnc_all).permute(1,2,3,0).reshape(-1,3).to(self.backgrid.device)
        feature = self.feature_compute(xyz_cnc_all)
        density = self.sigma_compute(feature)
        alphas  = nn.Softplus()(density+self.act_shift).squeeze(-1)
        if self.color_thre > 0:
            mask = (alphas > self.color_thre).reshape(-1)
            xyz_cnc_all = xyz_cnc_all[mask]
        feat_forw = F.grid_sample(self.forwgrid, xyz_cnc_all[None,None,None,:,:], mode='bilinear', align_corners=True)
        feat_forw = feat_forw[0,:,0,0,:].permute(1,0)
        feat_in = {'feat_forw':feat_forw, 'xyz':xyz_cnc_all}
        feat_forw, attn_hard, attn_soft, mean_slots = self.groupnet(feat_in, training=False)
        all_cluster_idx = torch.max(attn_hard, dim=1)[1].cpu()[0]
        # calculate group spatial distance
        dist_map = 100*torch.ones(self.slot_num, self.slot_num)
        for idx_slot_1 in range(self.slot_num):
            for idx_slot_2 in range(self.slot_num):
                if idx_slot_1 == idx_slot_2:
                    continue
                if attn_hard[0,idx_slot_1,:].sum() == 0:
                    continue
                if attn_hard[0,idx_slot_2,:].sum() == 0:
                    continue      
                points_slot_1 = xyz_cnc_all[attn_hard[0,idx_slot_1,:] > 0.5,:]
                points_slot_2 = xyz_cnc_all[attn_hard[0,idx_slot_2,:] > 0.5,:]
                points_diff = points_slot_1.unsqueeze(1).repeat(1,points_slot_2.shape[0],1) - points_slot_2.unsqueeze(0).repeat(points_slot_1.shape[0],1,1)
                points_diff = (points_diff ** 2).sum(-1) ** 0.5
                points_diff = points_diff.min()
                dist_map[idx_slot_1, idx_slot_2] = points_diff

        all_motn_forw = []
        all_rotat_forw = []
        all_trans_forw = []
        for idx_time in range(len(time_list)):
            time = time_list[idx_time] 
            times  = (torch.ones((mean_slots.shape[0],1))*time).to(self.backgrid.device)
            timefeat = self.poc_fre(times, self.time_poc)
            motn_forw = nn.ReLU(inplace = True)(self.forwnet(torch.cat([mean_slots, timefeat], dim=-1)))
            rotat_forw = rotat_from_6d(self.rotatnet(motn_forw))
            trans_forw = self.transnet(motn_forw)
            all_motn_forw.append(motn_forw)
            all_rotat_forw.append(rotat_forw)
            all_trans_forw.append(trans_forw)


        all_motn_forw  = torch.stack(all_motn_forw, dim=0).permute(1,0,2)
        all_rotat_forw = torch.stack(all_rotat_forw, dim=0).permute(1,0,2,3)
        all_trans_forw = torch.stack(all_trans_forw, dim=0).permute(1,0,2)

        all_transform = torch.zeros(self.slot_num, len(time_list), 4, 4)
        all_transform[:,:,:3,:3] = all_rotat_forw.permute(0,1,3,2)
        all_transform[:,:,:3,-1] = all_trans_forw
        all_transform[:,:,-1,-1] = 1
        all_transform_inverse = torch.linalg.inv(all_transform).permute(1,0,2,3)
        all_transform = all_transform.permute(1,0,2,3)
        all_transform_inverse = all_transform_inverse.unsqueeze(1).repeat(1,self.slot_num,1,1,1)
        all_transform = all_transform.unsqueeze(2).repeat(1,1,self.slot_num,1,1)
        diff = torch.einsum('abcde,abcef->abcdf', all_transform_inverse, all_transform)
        diff = diff - torch.eye(4)
        diff = (diff ** 2).sum((-1,-2))
        diff = diff.sum(0)

        all_merge_traj_diff = [0]
        all_merge_spat_diff = [0]
        # all_cluster_idx = torch.cat(all_cluster_idx)
        # calculate all mean slots
        diff_slots  = (diff + torch.eye(self.slot_num)*100)
        empty_slots = torch.tensor(list(set(range(self.slot_num)).difference(set(all_cluster_idx.unique().numpy()))))
        empty_masks = torch.zeros(self.slot_num).scatter_(0, empty_slots, 1.)
        diff_slots += 100*(empty_masks[None,:].repeat(self.slot_num,1)+empty_masks[:,None].repeat(1,self.slot_num))
        merge_all   = [[[x] for x in list(range(self.slot_num))]]
        for idx_merge in range(int(self.slot_num**2/2)):
            merge_cur = copy.deepcopy(merge_all[-1])
            if diff_slots.min() >= 100: 
                break
            # calculate merge pair with the minimum distance
            merge_pair = torch.where(diff_slots == diff_slots.min())
            cur_traj_diff = copy.deepcopy(diff_slots[merge_pair[0][0],merge_pair[1][0]])
            cur_spat_diff = copy.deepcopy(dist_map[merge_pair[0][0],merge_pair[1][0]])
            for idx_merge_now in range(merge_pair[0].shape[0]):
                coord_x = merge_pair[0][idx_merge_now]
                coord_y = merge_pair[1][idx_merge_now]
                diff_slots[coord_x,coord_y] = 100
                diff_slots[coord_y,coord_x] = 100

            merge_num_1, merge_num_2 = merge_pair[0][0], merge_pair[1][0]
            # if already merge, continue to next merge route
            already_merge = False
            for group in merge_cur:
                if (merge_num_1 in group) and (merge_num_2 in group):
                    already_merge = True
            if already_merge: continue
            all_merge_traj_diff.append(cur_traj_diff)
            all_merge_spat_diff.append(cur_spat_diff)
            # merge group
            for group in merge_cur:
                if merge_num_2 in group:
                    pick_group = group
                    merge_cur.remove(pick_group)
                    break
            for idx_group in range(len(merge_cur)):
                if merge_num_1 in merge_cur[idx_group]:
                    merge_cur[idx_group] += pick_group
                    break
            merge_all.append(merge_cur)

        # all_merge_traj_diff is the APE cost
        
        # calculate all merge
        diff_merge_rotat  = torch.zeros(len(merge_all), len(time_list))
        diff_merge_trans  = torch.zeros(len(merge_all), len(time_list))
        diff_merge_deform = torch.zeros(len(merge_all), len(time_list))

        diff_merge_rotat_mod  = torch.zeros(len(merge_all), len(time_list))
        diff_merge_trans_mod  = torch.zeros(len(merge_all), len(time_list))
        diff_merge_deform_mod = torch.zeros(len(merge_all), len(time_list))  

        rotat_forw_all = []
        trans_forw_all = []
        xyz_smp_pred_all = []

        resolution  = 30
        xyz_smp_all = np.mgrid[-1:1:complex(0,resolution), -1:1:complex(0,resolution), -1:1:complex(0,resolution)]
        xyz_smp_all = torch.FloatTensor(xyz_smp_all).permute(1,2,3,0).reshape(-1,3).to(self.backgrid.device)

        for idx_merge in range(len(merge_all)):
            for idx_time in range(len(time_list)):
                merge_cur = merge_all[idx_merge]
                time = time_list[idx_time]
                times  = (torch.ones((xyz_smp_all.shape[0],1))*time).to(self.backgrid.device)
                xyz_cnc, motn_back, rotat_back, trans_back, deform_back = self.back_deform(xyz_smp_all, times)

                feature = self.feature_compute(xyz_cnc)
                density = self.sigma_compute(feature)
                alphas  = nn.Softplus()(density+self.act_shift).squeeze(-1)

                if self.color_thre > 0:
                    mask = (alphas > self.color_thre).reshape(-1)
                    times   = times[mask]
                    xyz_cnc     = xyz_cnc[mask]
                    xyz_smp     = xyz_smp_all[mask]
                    motn_back   = motn_back[mask]
                    rotat_back  = rotat_back[mask]
                    trans_back  = trans_back[mask]
                    deform_back = deform_back[mask]

                attn_hard, attn_soft, mean_slots = self.forw_deform(xyz_cnc, times, training=False)[:3]
                cluster_idx = torch.max(attn_hard, dim=1)[1].cpu()
                # convert cluster_idx and mean_slots to new cluster_idx and mean_slots 
                cluster_idx_new = -torch.ones_like(cluster_idx)
                mean_slots_new  = -torch.zeros(len(merge_cur),self.width_forwgrid)
                for idx_group in range(len(merge_cur)):
                    for cluster_ori in merge_cur[idx_group]:
                        cluster_idx_new[cluster_idx == cluster_ori] = idx_group
                    mean_slots_new[idx_group] += mean_slots[merge_cur[idx_group]].mean(0)
                # calculate new deform based on new mean slots

                timefeat_forw = self.poc_fre(times, self.time_poc)
                motn_forw_new = nn.ReLU(inplace = True)(self.forwnet(torch.cat([mean_slots_new[cluster_idx_new], timefeat_forw], dim=-1)))
                rotat_forw_new = rotat_from_6d(self.rotatnet(motn_forw_new))
                trans_forw_new = self.transnet(motn_forw_new)
                xyz_smp_pred_new = torch.einsum('bcd,bde->bce',rotat_forw_new.permute(0,2,1),xyz_cnc.unsqueeze(-1)).squeeze(-1)
                xyz_smp_pred_new = xyz_smp_pred_new + trans_forw_new

                diff_merge_rotat[idx_merge, idx_time]  += ((rotat_forw_new-rotat_back)**2).mean() *10000
                diff_merge_trans[idx_merge, idx_time]  += ((trans_forw_new-trans_back)**2).mean() *10000
                diff_merge_deform[idx_merge, idx_time] += ((xyz_smp_pred_new- xyz_smp)**2).mean() *10000

                if idx_merge == 0:
                    rotat_forw_all.append(rotat_forw_new)
                    trans_forw_all.append(trans_forw_new)
                    xyz_smp_pred_all.append(xyz_smp_pred_new)
                else:
                    for idx_group in range(len(merge_cur)):
                        if merge_cur[idx_group] not in merge_all[idx_merge-1]:
                            merge_mask = torch.zeros_like(cluster_idx)
                            for involve_cluster in merge_cur[idx_group]:
                                merge_mask[cluster_idx==involve_cluster] = 1

                    merge_mask = merge_mask.to(rotat_forw_new.device)
                    diff_merge_rotat_mod[idx_merge, idx_time]  += (((rotat_forw_new-rotat_forw_all[idx_time])**2).sum((1,2)) * merge_mask).sum() / merge_mask.sum() *10000
                    diff_merge_trans_mod[idx_merge, idx_time]  += (((trans_forw_new-trans_forw_all[idx_time])**2).sum((1)) * merge_mask).sum() / merge_mask.sum() *10000
                    diff_merge_deform_mod[idx_merge, idx_time] += (((xyz_smp_pred_new- xyz_smp_pred_all[idx_time])**2).sum((1)) * merge_mask).sum() / merge_mask.sum() *10000

        return merge_all, diff_merge_deform.mean(1), diff_merge_deform_mod.mean(1), list(empty_slots.cpu().numpy()), all_merge_traj_diff, all_merge_spat_diff

    def forward(self, times, rays_o, rays_d, viewdir, white_bg=True, iteration=-1, is_train=True, idx_part=-1, times_prev=None, times_next=None, group=None):
        ret_dict = {}
        batchsize = times.shape[0]

        xyz_smp, rays_id, step_id, _ = self.sample_ray(rays_o, rays_d)
        xyz_smp = self.normalize_xyz(xyz_smp)

        try:    max_step = step_id.max()
        except: max_step = 1

        xyz_cnc, motn_back,_ ,_ ,deform_back = self.back_deform(xyz_smp, times[rays_id])
        if times_prev is not None:
            _, _,_ ,_ ,deform_back_prev = self.back_deform(xyz_smp, times_prev[rays_id])
        if times_next is not None:
            _, _,_ ,_ ,deform_back_next = self.back_deform(xyz_smp, times_next[rays_id])

        feature = self.feature_compute(xyz_cnc)
        density = self.sigma_compute(feature)
        alphas  = nn.Softplus()(density+self.act_shift).squeeze(-1)

        if self.color_thre > 0:
            mask        = (alphas > self.color_thre)
            alphas      = alphas[mask]
            xyz_smp     = xyz_smp[mask]
            xyz_cnc     = xyz_cnc[mask]
            rays_id     = rays_id[mask]
            step_id     = step_id[mask]
            feature     = feature[mask]
            motn_back   = motn_back[mask]
            deform_back = deform_back[mask]

        if times_prev is not None:
            deform_back_prev = deform_back_prev[mask]
            ret_dict.update({'deform_back_prev':deform_back_prev})
        if times_next is not None:
            deform_back_next = deform_back_next[mask]
            ret_dict.update({'deform_back_next':deform_back_next})

        attn_hard, attn_soft, mean_slots, _, motn_forw, _, _, _ = self.forw_deform(xyz_cnc, times[rays_id], iteration, training = is_train)

        if idx_part != -1:
            cluster_idx = torch.max(attn_hard, dim=1)[1]
            cluster_idx = self.merge_group(cluster_idx, group=group)
            mask = (cluster_idx == idx_part)
            attn_hard = attn_hard[mask]
            xyz_smp   = xyz_smp[mask]
            xyz_cnc   = xyz_cnc[mask]
            rays_id   = rays_id[mask]
            step_id   = step_id[mask]
            alphas    = alphas[mask]
            feature   = feature[mask]
            
        weights, alphainv_last = Alphas2Weights.apply(alphas, rays_id, batchsize)

        if self.color_thre > 0:
            mask    = (weights > self.color_thre)
            weights = weights[mask]
            xyz_smp = xyz_smp[mask]
            xyz_cnc = xyz_cnc[mask]
            rays_id = rays_id[mask]
            step_id = step_id[mask]
            feature = feature[mask]
            if attn_hard is not None:
                attn_hard = attn_hard[mask]

        rgb_xyz = 1.2/(1+torch.exp(-self.color_compute(feature, viewdir[rays_id])))-0.1 
        rgbs = segment_coo(src=(weights.unsqueeze(-1)*rgb_xyz), index=rays_id, out=torch.zeros([batchsize, 3]), reduce='sum')
        rgbs += (alphainv_last.unsqueeze(-1) * white_bg)

        ret_dict.update({'rgbs':rgbs,       'rgb_xyz':rgb_xyz,   'rays_id':rays_id, 'alphainv_last':alphainv_last, 'xyz_smp': xyz_smp, #'xyz_cnc_ori': xyz_cnc_ori, 
                         'weights':weights, 'max_step':max_step, 'alphas':alphas,   'step_id':step_id, 'deform_back': deform_back,
                         'motn_back':motn_back, 'motn_forw':motn_forw, 'attn_soft':attn_soft, 'attn_hard':attn_hard, 'mean_slots':mean_slots})         

        return ret_dict 
