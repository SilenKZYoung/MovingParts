import os, datetime, time
import json, random
import numpy as np
import torch
import torch.nn.functional as F

from argparse import Namespace

import opt
import dloader
import model
import utils

def test(args):
    logfolder = os.path.dirname(args.ckpt)
    
    ckpt = torch.load(args.ckpt, map_location=device)
    logfile  = os.path.dirname(args.ckpt) + '/log_test.log'
    xyz_min  = ckpt['kwargs']['xyz_min']
    xyz_max  = ckpt['kwargs']['xyz_max']
    near_far = ckpt['kwargs']['near_far']
    num_voxels_feat = ckpt['kwargs']['num_voxels_feat']
    num_voxels_motn = ckpt['kwargs']['num_voxels_motn']

    with open(os.path.dirname(args.ckpt) + '/args.json','r') as f:
        data_dir = json.load(f)['data_dir']

    dataset_train = dloader.DynamicDataset(data_dir, split='train', downsample=args.downsample_train, N_vis=args.N_vis)
    dataset_test  = dloader.DynamicDataset(data_dir, split='test',  downsample=args.downsample_test,  N_vis=args.N_vis)
        
    t_nerf = model.time_nerf(xyz_min, xyz_max, near_far, num_voxels_feat, num_voxels_motn, logfile, device, Namespace(**ckpt['kwargs'])).to(device)
    t_nerf.load(ckpt)

    render_args =  {'lego'         :{'path': [2,12], 'video': [2,12], 'trail': [2,12], 'edit': [2,12], 'group': []},
                    'trex'         :{'path': [13] ,'video': [13] ,'trail': [13] ,'edit': [13] ,'group': []},
                    'hook'         :{'path': [6], 'video': [6], 'trail': [6], 'edit': [6], 'group': []},
                    'mutant'       :{'path': [7], 'video': [7], 'trail': [7], 'edit': [7], 'group': []},
                    'standup'      :{'path': [20], 'video': [20], 'trail': [20], 'edit': [20], 'group': []},
                    'jumpingjacks' :{'path': [32], 'video': [32], 'trail': [32], 'edit': [32], 'group': []},
                    'hellwarrior'  :{'path': [32], 'video': [32], 'trail': [32], 'edit': [32], 'group': []},
                    'bouncingballs':{'path': [12,25] , 'video': [12,25] , 'trail': [12,25] , 'edit': [12,25] , 'group': []}}
    
    render_arg  = render_args[data_dir.split('/')[-2]]

    # evaluation
    psnr_train, _ = t_nerf.evaluation(dataset_train, group=None, idx_part=-1, save_path=f'{logfolder}/imgs_train_all/', logfile=logfile)
    psnr_test , _ = t_nerf.evaluation(dataset_test , group=None, idx_part=-1, save_path=f'{logfolder}/imgs_test_all/' , logfile=logfile)

    t_nerf.export_mesh(f'{logfolder}/mesh/')
    t_nerf.render_video(dataset_train, group=None, idx_part=-1, save_path=f'{logfolder}/imgs_video/', idxs_image=render_arg['video'])
    
    # compute segment
    merge_all, diff_merge, diff_merge_mod, empty_slots, all_merge_traj_diff, all_merge_spat_diff = t_nerf.export_segment(f'{logfolder}/segment/')
    with open(logfile, "a") as f:
        f.write('export_segment: ' + '\n')
        f.write('merge_all  : ' + '\n')
        for idx_merge in range(len(merge_all)):
            f.write(str(merge_all[idx_merge]) + '\n')
        f.write('diff_merge     : ' + str(diff_merge) + '\n')
        f.write('diff_merge_mod : ' + str(diff_merge_mod) + '\n')
        f.write('empty_slots    : ' + str(empty_slots) + '\n')
        f.write('all_merge_traj_diff    : ' + str(all_merge_traj_diff) + '\n')
        f.write('all_merge_spat_diff    : ' + str(all_merge_spat_diff) + '\n')

    # # export merge
    if  render_arg['group'] == []:
        render_arg['group'] = list(range(len(merge_all)))
    for idx_group in render_arg['group']:
        group = merge_all[idx_group]
        t_nerf.render_path(dataset_train, group=group, idx_part=-1, save_path=f'{logfolder}/imgs_path_group/{str(idx_group)}/', idxs_image=render_arg['path'])

def train(args):
    args.expname = '(' + args.data_dir.split('/')[-2] + ')-' + args.expname
    logfolder = f'{args.savedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    os.makedirs(f'{logfolder}/imgs_vis',  exist_ok=True)
    
    with open(f'{logfolder}/args.json', 'w') as fp: 
        json.dump(args.__dict__, fp, indent=4)

    logfile = f'{logfolder}/log.log'
    with open(logfile, "a") as f:
        f.write(json.dumps(args.__dict__, indent=4))
        f.write('\n')

    if 'real' in args.data_dir:
        dataset_train = dloader_real_mono.DynamicDatasetReal(args.data_dir, split='train', downsample=args.downsample_train, N_vis=args.N_vis)
        dataset_test  = dloader_real_mono.DynamicDatasetReal(args.data_dir, split='test',  downsample=args.downsample_test,  N_vis=args.N_vis)
    else:
        dataset_train = dloader.DynamicDataset(args.data_dir, split='train', downsample=args.downsample_train, N_vis=args.N_vis)
        dataset_test  = dloader.DynamicDataset(args.data_dir, split='test',  downsample=args.downsample_test,  N_vis=args.N_vis)

    near_far = dataset_train.near_far
    xyz_min = dataset_train.scene_bbox[0]
    xyz_max = dataset_train.scene_bbox[1]
    if abs(args.bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (args.bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift

    num_voxels_feat = args.num_voxels_feat
    num_voxels_motn = args.num_voxels_motn
    if len(args.pg_scale_feat) :
        num_voxels_feat = int(num_voxels_feat / (2**len(args.pg_scale_feat))**2)
    if len(args.pg_scale_motn) :
        num_voxels_motn = int(num_voxels_motn / (2**len(args.pg_scale_motn))**3)

    t_nerf = model.time_nerf(xyz_min, xyz_max, near_far, num_voxels_feat, num_voxels_motn, logfile, device, args).to(device)

    optimizer = utils.create_optimizer_or_freeze_model(t_nerf, args, logfile, global_step=0)

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

    all_times, all_rgbs, all_rays_o, all_rays_d, all_viewdir = dataset_train.get_rays_maskcache(t_nerf, device, logfile, args.if_progress)

    index_generator = utils.batch_indices_generator(all_times.shape[0], args.batch_size)
    batch_index_sampler = lambda: next(index_generator)

    torch.cuda.synchronize()
    begin_time = time.time()
    for iteration in range(args.n_iters):
        if iteration in args.pg_scale_feat:
            n_rest_scales = len(args.pg_scale_feat)-args.pg_scale_feat.index(iteration)-1
            cur_voxels = int(args.num_voxels_feat / (2**n_rest_scales)**2)
            t_nerf.scale_volume_grid(cur_voxels, 'feat')
            optimizer = utils.create_optimizer_or_freeze_model(t_nerf, args, logfile, global_step=iteration)

        if iteration in args.pg_scale_motn:
            n_rest_scales = len(args.pg_scale_motn)-args.pg_scale_motn.index(iteration)-1
            cur_voxels = int(args.num_voxels_motn / (2**n_rest_scales)**3)
            t_nerf.scale_volume_grid(cur_voxels, 'motn')
            optimizer = utils.create_optimizer_or_freeze_model(t_nerf, args, logfile, global_step=iteration)
        
        if args.if_progress:
            if iteration < args.iters_time:
                skip_factor = iteration / float(args.iters_time+1) * dataset_train.__len__()
                max_sample = max(int(skip_factor), 6)
                pick_image = np.random.choice(np.arange(max_sample))
            else:
                pick_image = np.random.choice(dataset_train.__len__())
            pick_pixel = np.random.choice(dataset_train.H * dataset_train.W, args.batch_size)
            idx_pick = (pick_image * dataset_train.H * dataset_train.W) + pick_pixel
            times   = all_times[idx_pick].to(device)
            target  = all_rgbs[idx_pick].to(device)
            rays_o  = all_rays_o[idx_pick].to(device)
            rays_d  = all_rays_d[idx_pick].to(device)
            viewdir = all_viewdir[idx_pick].to(device)
        else:
            idx_pick = batch_index_sampler()
            times   = all_times[idx_pick].to(device)
            target  = all_rgbs[idx_pick].to(device)
            rays_o  = all_rays_o[idx_pick].to(device)
            rays_d  = all_rays_d[idx_pick].to(device)
            viewdir = all_viewdir[idx_pick].to(device)
        assert times.max() == times.min()

        return_dict = t_nerf(times, rays_o, rays_d, viewdir, is_train=True, iteration=iteration, times_prev=None, times_next=None)

        optimizer.zero_grad(set_to_none = True)
        loss = F.mse_loss(return_dict['rgbs'], target)
        psnr = utils.mse2psnr(loss.detach())

        if (iteration > args.n_iter_fbloss) and (args.w_motns > 0):
            loss_fb = torch.mean((return_dict['motn_back'] - return_dict['motn_forw'])**2)
            loss += loss_fb * args.w_motns
            loss_fb_detach = loss_fb.detach().item()
        else:
            loss_fb_detach = -1

        if args.w_rgbper > 0:
            rgbper = (return_dict['rgb_xyz'] - target[return_dict['rays_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * return_dict['weights'].detach()).sum() / len(rays_o)
            loss += rgbper_loss * args.w_rgbper
        
        if args.w_entropy > 0:
            pout = return_dict['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += args.w_entropy * entropy_last_loss
        
        loss.backward()

        if iteration<args.tv_before and iteration>args.tv_after and iteration%args.tv_every==0:
            if args.w_tv_feat>0 or args.w_tv_forw>0 or args.w_tv_back>0:
                t_nerf.add_tv_grad(args.w_tv_feat /args.batch_size, args.w_tv_forw /args.batch_size, 
                                   args.w_tv_back /args.batch_size, iteration<args.tv_feature_before)

        optimizer.step()
        PSNRs.append(psnr.item())                      

        if (iteration >= args.cnc_decay_begin) and (iteration < args.cnc_decay_end):
            decay_factor = args.cnc_decay_rate ** (1/(args.cnc_decay_end - args.cnc_decay_begin))
            for i_opt_g, param_group in enumerate(optimizer.param_groups):
                if param_group['params_name'] in ['featgrid', 'featnet', 'sigmanet', 'colornet1', 'colornet2']:
                    param_group['lr'] = param_group['lr'] * decay_factor

        if (iteration >= args.motion_decay_begin) and (iteration < args.motion_decay_end):
            decay_factor = args.motion_decay_rate ** (1/(args.motion_decay_end - args.motion_decay_begin))
            for i_opt_g, param_group in enumerate(optimizer.param_groups):
                if param_group['params_name'] in ['backgrid', 'forwgrid', 'backnet', 'forwnet', 'rotatnet', 'transnet', 'groupnet']:
                    param_group['lr'] = param_group['lr'] * decay_factor


        if iteration % args.refresh_rate == 0:
            torch.cuda.synchronize()
            with open(logfile, "a") as f:
                f.write(f'Iteration {iteration:05d}: '   + f'train_psnr={float(np.mean(PSNRs)):.2f} ' + \
                        f'forback={loss_fb_detach:.6f} ' + f'time={time.time()-begin_time:.6f}')
                f.write('\n')
            PSNRs = []
        
        if iteration % args.vis_every == 0 and iteration != 0:
            psnr_test, _ = t_nerf.evaluation(dataset_test, idx_part=-1, logfile=logfile, save_path=f'{logfolder}/imgs_vis/', prtx=f'{iteration:06d}_', N_vis=6)
            t_nerf.export_mesh(f'{logfolder}/mesh_'+ str(iteration) +'/')

    psnr_train, _ = t_nerf.evaluation(dataset_train, group=None, idx_part=-1, save_path=f'{logfolder}/imgs_train_all/', logfile=logfile, N_vis=6)
    psnr_test , _ = t_nerf.evaluation(dataset_test,  group=None, idx_part=-1, save_path=f'{logfolder}/imgs_test_all/' , logfile=logfile, N_vis=6)
    t_nerf.save(f'{logfolder}/{args.expname}_{str(psnr_test)}.th')

    t_nerf.render_path(dataset_train, group=None, idx_part=-1, save_path=f'{logfolder}/imgs_path/', idxs_image=[0])
    t_nerf.export_mesh(f'{logfolder}/mesh/')


if __name__ == '__main__':
    args = opt.config_parser()

    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.ckpt != '':
        test(args)
    else:
        train(args) 