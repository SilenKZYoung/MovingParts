import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config',        is_config_file=True)
    parser.add_argument("--expname",       type=str,  default='')
    parser.add_argument("--savedir",       type=str,  default='./log')
    parser.add_argument("--ckpt",          type=str,  default='')
    parser.add_argument("--add_timestamp", type=int,  default=0)
    parser.add_argument("--seed",          type=int,  default=0)

    parser.add_argument("--n_iters",       type=int,  default=20000)
    parser.add_argument("--if_progress",   type=bool, default=True)
    parser.add_argument("--iters_time",    type=int,  default=2000)
    parser.add_argument("--batch_size",    type=int,  default=4096)

    parser.add_argument("--refresh_rate",  type=int,  default=100)
    parser.add_argument("--vis_every",     type=int,  default=2000)
    parser.add_argument("--render_test",   type=int,  default=1)
    parser.add_argument("--render_train",  type=int,  default=1)
    parser.add_argument("--export_mesh",   type=int,  default=1)

    # data setting
    parser.add_argument("--data_type",         type=str,   default='blender')
    parser.add_argument("--data_dir",          type=str,   default='./data/bouncingballs/')
    parser.add_argument("--N_vis",             type=int,   default=-1)
    parser.add_argument('--downsample_train',  type=float, default=2.0)
    parser.add_argument('--downsample_test',   type=float, default=2.0)

    # loss weight
    parser.add_argument("--tv_every",          type=int,   default= 1)
    parser.add_argument("--tv_after",          type=int,   default=-1)
    parser.add_argument("--tv_before",         type=int,   default=1e9)
    parser.add_argument("--tv_feature_before", type=int,   default=10000)
    parser.add_argument("--w_tv_feat",         type=float, default=0)
    parser.add_argument("--w_tv_forw",         type=float, default=1e-2)
    parser.add_argument("--w_tv_back",         type=float, default=1e-2)

    parser.add_argument("--w_entropy",         type=float, default=1e-3)
    parser.add_argument("--w_rgbper",          type=float, default=1e-2) 

    parser.add_argument("--n_iter_fbloss",     type=int,   default=-1)
    parser.add_argument("--n_iter_slot",       type=int,   default=3000)
    parser.add_argument("--w_motns",           type=float, default=0.1)

    parser.add_argument('--slot_name',         type=str,   default='groupvit_xyz')
    parser.add_argument('--slot_num',          type=int,   default=12)
    parser.add_argument('--slot_hard',         type=str,   default='hard')

    # learning rate
    parser.add_argument("--motion_decay_begin", type=int, default=0)
    parser.add_argument("--motion_decay_end",   type=int, default=20000)
    parser.add_argument("--motion_decay_rate",  type=int, default=0.1)

    parser.add_argument("--cnc_decay_begin",   type=int, default=0)
    parser.add_argument("--cnc_decay_end",     type=int, default=20000)
    parser.add_argument("--cnc_decay_rate",    type=int, default=0.1)

    parser.add_argument("--lrate_backgrid",    type=float,   default=8e-2)
    parser.add_argument("--lrate_forwgrid",    type=float,   default=8e-2)
    parser.add_argument("--lrate_featgrid",    type=float,   default=1e-2)

    parser.add_argument("--lrate_backnet",     type=float,   default=6e-4)
    parser.add_argument("--lrate_forwnet",     type=float,   default=6e-4)
    parser.add_argument("--lrate_rotatnet",    type=float,   default=6e-4)
    parser.add_argument("--lrate_transnet",    type=float,   default=6e-4)
    parser.add_argument("--lrate_groupnet",    type=float,   default=8e-4)

    parser.add_argument("--lrate_featnet",     type=float,   default=8e-4)
    parser.add_argument("--lrate_sigmanet",    type=float,   default=8e-4)
    parser.add_argument("--lrate_colornet1",   type=float,   default=8e-4)
    parser.add_argument("--lrate_colornet2",   type=float,   default=8e-4)

    # ray module
    parser.add_argument('--step_ratio',  type=float, default=0.5)
    parser.add_argument('--alpha_init',  type=float, default=1e-2)
    parser.add_argument('--color_thre',  type=float, default=1e-3)
    parser.add_argument('--bound_scale', type=float, default=1.05)

    # network parameters
    parser.add_argument("--num_voxels_motn", type=int, default=50**3)
    parser.add_argument("--pg_scale_motn",   type=int, action="append", default=[])
    parser.add_argument("--num_voxels_feat", type=int, default=160**3)
    parser.add_argument("--pg_scale_feat",   type=int, action="append", default=[4000, 6000, 8000])

    parser.add_argument("--skip_zero_grad",  type=str, action="append", default=['featgrid', 'backgrid', 'forwgrid'])

    parser.add_argument('--freqs_time',   type=int,   default=4)
    parser.add_argument('--freqs_posi',   type=int,   default=10)
    parser.add_argument('--freqs_view',   type=int,   default=4)
    parser.add_argument('--freqs_grid',   type=int,   default=2)

    # backward module
    parser.add_argument('--width_motn',      type=int,   default=128)
    
    parser.add_argument('--width_backgrid',  type=int,   default=20)
    parser.add_argument('--width_backnet',   type=int,   default=128)
    parser.add_argument('--layer_backnet',   type=int,   default=2)
    # forward module
    parser.add_argument('--width_forwgrid',  type=int,   default=20)
    parser.add_argument('--width_forwnet',   type=int,   default=128)
    parser.add_argument('--layer_forwnet',   type=int,   default=2)

    # canonical module
    parser.add_argument('--width_mlp',       type=int,   default=128)
    parser.add_argument('--width_featgrid',  type=int,   default=6)
    parser.add_argument('--width_featnet',   type=int,   default=128)
    parser.add_argument('--layer_featnet',   type=int,   default=1)
    parser.add_argument('--layer_sigmanet',  type=int,   default=1)
    parser.add_argument('--layer_colornet',  type=int,   default=2)

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
