import sys, os, json
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import imageio
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

class DynamicDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, N_vis=-1):
        self.root_dir = datadir
        self.data_name = self.root_dir.split('/')[-2]
        self.split = split
        self.downsample=downsample
        self.N_vis = N_vis
        if 'kubric' in self.data_name:
            self.W, self.H = int(256/downsample), int(256/downsample)
            self.scene_bbox = torch.tensor([[-7., -10., -10.], [13., 10., 10.]])
            self.near_far = [1,40]
        else:
            self.W, self.H = int(800/downsample), int(800/downsample)
            self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
            self.near_far = [2.0,6.0]
        self.white_bg = True
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

        self.transform = T.ToTensor()
        self.read_meta()

    @torch.no_grad()
    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)
        self.focal = self.W * 0.5 / np.tan(0.5 * self.meta['camera_angle_x'])

        img_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_interval))
        self.all_rgbs = torch.zeros(len(idxs), self.H, self.W, 3)
        self.all_pose = torch.zeros(len(idxs), 4, 4)
        self.all_time = torch.zeros(len(idxs))
        self.all_ids  = torch.zeros(len(idxs))
        self.all_intrinsic = torch.zeros(len(idxs), 3, 3)

        for idx_image in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#
            frame = self.meta['frames'][idx_image]
            image_path = self.root_dir + frame['file_path'][1:] + '.png'
            
            rgbs = self.transform(Image.open(image_path).resize((self.W,self.H), Image.LANCZOS)).permute(1,2,0)
            rgbs = (rgbs[...,:3] * rgbs[...,-1:] + (1 - rgbs[...,-1:]))
            pose = torch.cuda.FloatTensor(np.array(frame['transform_matrix']))

            self.all_rgbs[idx_image] = rgbs
            self.all_pose[idx_image] = pose
            self.all_time[idx_image] = frame['time']
            self.all_ids[idx_image]  = frame['frame_id'] if 'frame_id' in frame.keys() else 0
            if 'camera_intrinsic' in frame.keys():
                self.all_intrinsic[idx_image] = torch.cuda.FloatTensor(np.array(frame['camera_intrinsic']))
        self.all_intrinsic = self.all_intrinsic / self.downsample

    @torch.no_grad()
    def get_rays(self, c2w, idx_image):
        i, j = torch.meshgrid(torch.linspace(0, self.W-1, self.W), torch.linspace(0, self.H-1, self.H)) 
        i, j = i.t(), j.t()
        i, j = i+0.5, j+0.5

        fx = self.all_intrinsic[idx_image][0, 0] if self.all_intrinsic.mean() != 0 else self.focal
        fy = self.all_intrinsic[idx_image][1, 1] if self.all_intrinsic.mean() != 0 else self.focal
        cx = self.all_intrinsic[idx_image][0,-1] if self.all_intrinsic.mean() != 0 else self.W*.5
        cy = self.all_intrinsic[idx_image][1,-1] if self.all_intrinsic.mean() != 0 else self.H*.5

        dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
        rays_o = c2w[:3,-1].expand(rays_d.shape)
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)
        return rays_o, rays_d
    
    @torch.no_grad()
    def project(self, idx_image, points, return_depth=False, use_projective_depth=True):
        c2w = self.all_pose[idx_image]
        orientation = c2w[:3,:3].T
        position    = c2w[:3, 3]
        translation = - orientation @ position
        extrins = torch.cat([orientation, translation[...,None]], dim=-1)
        extrins = torch.cat([extrins, torch.Tensor([[0,0,0,1]])], dim=0)
        if self.all_intrinsic.mean() != 0:
            intrins = self.all_intrinsic[idx_image]
        else:
            intrins = torch.zeros([3,3])
            intrins[ 0, 0] += self.focal
            intrins[ 1, 1] += self.focal
            intrins[ 0,-1] += self.W*.5
            intrins[ 1,-1] += self.H*.5
            intrins[-1,-1] += 1
            
        local_points      = (extrins[..., :3, :3] @ points[..., None])[..., 0] + extrins[..., :3, 3]
        normalized_pixels = torch.where(local_points[..., -1:] != 0, local_points[..., :2] / local_points[..., -1:], torch.Tensor([0]))
        
        pixels = intrins @ torch.cat([normalized_pixels, torch.ones_like(normalized_pixels[...,:1])],dim=-1)[..., None]
        pixels = pixels[..., 0][..., :2]
        
        if not return_depth:
            return pixels
        else:
            depths = local_points[..., 2:] if use_projective_depth else torch.norm(local_points, dim=-1)[...,None]
            return pixels, depths  
    
    @torch.no_grad()
    def get_rays_one_view(self, idx_image):
        rays_o, rays_d = self.get_rays(self.all_pose[idx_image], idx_image)
        viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
        return rays_o, rays_d, viewdirs
        
    @torch.no_grad()
    def get_rays_one_view_pose(self, c2w):
        i, j = torch.meshgrid(torch.linspace(0, self.W-1, self.W), torch.linspace(0, self.H-1, self.H)) 
        i, j = i.t(), j.t()
        i, j = i+0.5, j+0.5

        fx = self.all_intrinsic[0][0, 0] if self.all_intrinsic.mean() != 0 else self.focal
        fy = self.all_intrinsic[0][1, 1] if self.all_intrinsic.mean() != 0 else self.focal
        cx = self.all_intrinsic[0][0,-1] if self.all_intrinsic.mean() != 0 else self.W*.5
        cy = self.all_intrinsic[0][1,-1] if self.all_intrinsic.mean() != 0 else self.H*.5

        dirs = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
        rays_o = c2w[:3,-1].expand(rays_d.shape)
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)
        
        viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
        return rays_o, rays_d, viewdirs

    @torch.no_grad()
    def get_rays_maskcache(self, model, device, logfile, if_progress):
        all_times   = torch.ones([self.all_time.shape[0]*self.H*self.W, 1],  device='cpu')
        all_rgbs    = torch.zeros([self.all_time.shape[0]*self.H*self.W, 3], device='cpu')
        all_rays_o  = torch.zeros([self.all_time.shape[0]*self.H*self.W, 3], device='cpu')
        all_rays_d  = torch.zeros([self.all_time.shape[0]*self.H*self.W, 3], device='cpu')
        all_viewdir = torch.zeros([self.all_time.shape[0]*self.H*self.W, 3], device='cpu')
        top = 0
        CHUNK = 4096
        for idx_image in range(self.all_time.shape[0]):
            assert self.H == self.W
            rays_o, rays_d, viewdirs = self.get_rays_one_view(idx_image)
            if if_progress:
                mask = torch.ones((self.H * self.W), device=device)>0
            else:
                mask = torch.empty((self.H * self.W), device=device, dtype=torch.bool)
                for i in range(0, self.H*self.W, CHUNK):
                    mask[i:i+CHUNK] = model.get_mask(rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK]).to(device)
            n = mask.sum()    
            all_times[top:top+n] = self.all_time[idx_image]
            all_rgbs[top:top+n].copy_(self.all_rgbs[idx_image].flatten(0,1)[mask])
            all_rays_o[top:top+n].copy_(rays_o[mask])
            all_rays_d[top:top+n].copy_(rays_d[mask])
            all_viewdir[top:top+n].copy_(viewdirs[mask])
            top += n

        with open(logfile, "a") as f:
            f.write('get_training_rays_in_maskcache_sampling: ratio '+ str((top/all_times.shape[0]).item())+'\n')
        all_times   = all_times[:top]
        all_rgbs    = all_rgbs[:top]
        all_rays_o  = all_rays_o[:top]
        all_rays_d  = all_rays_d[:top]
        all_viewdir = all_viewdir[:top]
        
        return all_times, all_rgbs, all_rays_o, all_rays_d, all_viewdir

    @torch.no_grad()
    def compute_bbox_by_cam_frustrm(self, logfile):
        xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
        xyz_max = -xyz_min
        for idx_image in range(self.all_time.shape[0]):
            rays_o, rays_d, viewdirs = self.get_rays_one_view(idx_image)
            pts_nf = torch.stack([rays_o+viewdirs*self.near_far[0], rays_o+viewdirs*self.near_far[1]])
            xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1)))
            xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1)))
        with open(logfile, "a") as f:
            f.write('compute_bbox_by_cam_frustrm: xyz_min ' + str(xyz_min) + '\n')
            f.write('compute_bbox_by_cam_frustrm: xyz_max ' + str(xyz_max) + '\n')
            f.write('compute_bbox_by_cam_frustrm: finish \n')
        return xyz_min, xyz_max

    def __len__(self):
        return self.all_time.shape[0]

    def __getitem__(self, idx):
        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx],
                  'times': self.all_times[idx]}
        return sample

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    datadir = './data/nerf_synthetic_time/lego/'
    logfile = './a.txt'
    dataset = DynamicDataset(datadir)
    dataset.get_rays_one_view(2)
    all_times, all_rgbs, all_rays_o, all_rays_d, all_viewdir = dataset.get_rays_maskcache(None, device, logfile, True)
