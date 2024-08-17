import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('./kinpoly')

import argparse
import os
import os.path as osp
from pathlib import Path
import yaml
import numpy as np
import joblib 
import pickle 
import json 
import time 

import imageio 

import torch
import torch.nn as nn
from torch.optim import AdamW

# import pytorch3d.transforms as transforms qpos2

from collections import defaultdict

from egoego.data.ares_demo_dataset import ARESDemoDataset 

from egoego.model.head_estimation_transformer import HeadFormer 
from egoego.model.head_normal_estimation_transformer import HeadNormalFormer

from egoego.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file, run_blender_rendering_and_save2video_head_pose, BLENDER_UTILS_PATH

from utils.data_utils.process_amass_dataset import determine_floor_height_and_contacts 

import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from scipy.spatial.transform import Rotation as sRot

import pytorch3d.transforms as transforms 

# from kinpoly.relive.utils import *
# from kinpoly.relive.models.mlp import MLP
# from kinpoly.relive.models.traj_ar_smpl_net import TrajARNet
# from kinpoly.relive.data_loaders.statear_smpl_dataset import StateARDataset
# from kinpoly.relive.utils.torch_humanoid import Humanoid
# from kinpoly.relive.data_process.process_trajs import get_expert
# from kinpoly.relive.utils.torch_ext import get_scheduler
# from kinpoly.relive.utils.statear_smpl_config import Config

from kinpoly.relive.data_process.convert_amass_ego_syn_to_qpos import get_head_vel, get_obj_relative_pose 

from trainer_amass_cond_motion_diffusion import get_trainer  

def find_take_uid_from_take_name(take_json, take_name):
	for take in take_json:
		if take["take_name"] == take_name:
			return take["take_uid"]
	return None

def find_take_from_take_name(take_json, take_name):
	for take in take_json:
		if take["take_name"] == take_name:
			return take
	return None

def find_take_from_take_uid(take_json, take_uid):
	for take in take_json:
		if take["take_uid"] == take_uid:
			return take
	return None
	
EGOEXO_ROOT_DIR = "/mnt/homes/minghao/robotflow/egoego/datasets/ego_exo"
EGOEXO_DATA_ANNO_DIR = os.path.join(EGOEXO_ROOT_DIR, "annotations/")

egoexo = {
	"takes": os.path.join(EGOEXO_ROOT_DIR, "takes.json"),
	# "captures": os.path.join(EGOEXO_ROOT_DIR, "captures.json"),
	# "physical_setting": os.path.join(EGOEXO_ROOT_DIR, "physical_setting.json"),
	# "participants": os.path.join(EGOEXO_ROOT_DIR, "participants.json"),
	# "visual_objects": os.path.join(EGOEXO_ROOT_DIR, "visual_objects.json"),
	"metadata": os.path.join(EGOEXO_ROOT_DIR, "metadata.json"),
	"splits": os.path.join(EGOEXO_ROOT_DIR, "annotations", "splits.json"),
}

# egoexo_pd = {}
# for k, v in egoexo.items():
# 	egoexo_pd[k] = pd.read_json(open(v))

for k, v in egoexo.items():
	egoexo[k] = json.load(open(v))

takes = egoexo["takes"]
# captures = egoexo["captures"]
takes_by_uid = {x["take_uid"]: x for x in takes}
#! Validate whether only train_uids has gopro_calibs.csv
# ! Validate the train_uids, val_uids, test_uids, and the total nums

ALL_TRAIN_UIDS = egoexo["splits"]["split_to_take_uids"]["train"]
ALL_VAL_UIDS = egoexo["splits"]["split_to_take_uids"]["val"]
ALL_TEST_UIDS = egoexo["splits"]["split_to_take_uids"]["test"]
ALL_TAKE_UIDS = [take["take_uid"] for take in egoexo["takes"]] 
ALL_TAKE_NAMES = [take["take_name"] for take in egoexo["takes"]]

print(f"train: {len(ALL_TRAIN_UIDS)}, val: {len(ALL_VAL_UIDS)}, test: {len(ALL_TEST_UIDS)}")
print(f"total: {len(ALL_TAKE_UIDS)}")

ALL_TRAIN_TAKES = [find_take_from_take_uid(takes, take_uid) for take_uid in ALL_TRAIN_UIDS]
ALL_VAL_TAKES = [find_take_from_take_uid(takes, take_uid) for take_uid in ALL_VAL_UIDS]
ALL_TEST_TAKES = [find_take_from_take_uid(takes, take_uid) for take_uid in ALL_TEST_UIDS]

traj_dirs = [os.path.join(EGOEXO_ROOT_DIR, take["root_dir"], "trajectory") for take in egoexo["takes"]]
traj_dirs = list(filter(lambda traj_dir: os.path.exists(os.path.join(traj_dir, "gopro_calibs.csv")), traj_dirs))
print(f"Total takes with gopro_calibs.csv: {len(traj_dirs)}")
exo_traj_paths = [os.path.join(traj_dir, "gopro_calibs.csv") for traj_dir in traj_dirs]

TAKE_NAMES_TO_TAKE_UIDS_MAP = {take["take_name"]: take["take_uid"] for take in egoexo["takes"]}
TAKE_UIDS_TO_TAKE_NAMES_MAP = {take["take_uid"]: take["take_name"] for take in egoexo["takes"]}



def test(opt, device):
	weight_root_folder = "./pretrained_models"

	# Load weight for HeadNet 
	weight_path = os.path.join(weight_root_folder, "stage1_headnet_ares_250.pt")
	ckpt = torch.load(weight_path, map_location=device)
	print(f"Loaded weight for head scale estimation: {weight_path}")

	# Load weight for GravityNet 
	normal_weight_path = os.path.join(weight_root_folder, "stage1_gravitynet_2000.pt")
	normal_ckpt = torch.load(normal_weight_path, map_location=device)
	print(f"Loaded weight for head normal estimation: {normal_weight_path}")

	data_root_folder = "./test_data/ares"
	demo_save_dir = "./test_data_res"
	if not os.path.exists(demo_save_dir):
		os.makedirs(demo_save_dir)

	# Define HeadFormer model. (For predicting scale)
	head_transformer_encoder = HeadFormer(opt, device)
	head_transformer_encoder.load_state_dict(ckpt['transformer_encoder_state_dict'])
	head_transformer_encoder.to(device)
	head_transformer_encoder.eval()

	# Define normal prediction model. (For rotating SLAM trajectory)
	head_normal_transformer = HeadNormalFormer(opt, device, eval_whole_pipeline=True)
	head_normal_transformer.load_state_dict(normal_ckpt['transformer_encoder_state_dict'])
	head_normal_transformer.to(device)
	head_normal_transformer.eval()

	# Load weight for diffusion model. 
	opt.data_root_folder = "./test_data/ares"
	diffusion_trainer = get_trainer(opt, run_demo=True)
	diffusion_weight_path = os.path.join(weight_root_folder, "stage2_diffusion_4.pt")
	diffusion_trainer.load_weight_path(diffusion_weight_path)

	# Prepare head pose data loader.
	val_dataset = ARESDemoDataset(data_root_folder)
	val_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=1, shuffle=False,
		num_workers=opt.workers, pin_memory=True, drop_last=False)
   
	# train_egoexo_data_dict = joblib.load("/mnt/homes/minghao/robotflow/egoego/third_party/egoego/processed_data/egoexo_train.pkl")
	train_egoexo_data_dict = None
	# val_egoexo_data_dict = joblib.load("/mnt/homes/minghao/robotflow/egoego/third_party/egoego/processed_data/egoexo_train_first_frame.pkl")
	# test_egoexo_data_dict = joblib.load("/mnt/homes/minghao/robotflow/egoego/third_party/egoego/processed_data/egoexo_test_first_frame.pkl")
	# label = "val"
	# data_dict_in_use = val_egoexo_data_dict if label == "val" else test_egoexo_data_dict

	train_egoexo_traj_npz = np.load("/mnt/homes/minghao/robotflow/egoego/third_party/egoego/processed_data/egoexo_train.npz")
	val_egoexo_traj_npz = np.load("/mnt/homes/minghao/robotflow/egoego/third_party/egoego/processed_data/egoexo_val.npz")
	test_egoexo_traj_npz = np.load("/mnt/homes/minghao/robotflow/egoego/third_party/egoego/processed_data/egoexo_test.npz")
	label = "val"
	if label=="train":
		egoexo_poses = train_egoexo_traj_npz
	elif label=="val":
		egoexo_poses = val_egoexo_traj_npz
	elif label=="test":
		egoexo_poses = test_egoexo_traj_npz
		

	# egoexo_poses = data_dict_in_use["poses"]
	# egoexo_trajectories = data_dict_in_use["trajectories"]
	# cameras = data_dict_in_use["cameras"]
	# metadata = data_dict_in_use["metadata"]
	# poses_take_uids = data_dict_in_use["poses_takes_uids"]

	with torch.no_grad():
		# for test_it, test_input_data_dict in enumerate(val_loader):
		for take_uid in egoexo_poses.files:
		   
			this_egoexo_trajectory = egoexo_poses[take_uid]

			# this_egoexo_trajectory = [this_egoexo_trajectory[fr] for fr in this_egoexo_trajectory]
			this_egoexo_trajectory = np.stack(this_egoexo_trajectory) # T x 3 x 4
			this_egoexo_trans = this_egoexo_trajectory[:, :, 3] # T x 3
			this_egoexo_rotmat = this_egoexo_trajectory[:, :, :3] # T x 3 x 3
			this_egoexo_quat = transforms.matrix_to_quaternion(torch.from_numpy(this_egoexo_rotmat).float()) # T x 4
			this_egoexo_pose = np.concatenate([this_egoexo_trans, this_egoexo_quat.cpu().numpy()], axis=-1) # T x 7
			this_egoexo_pose = torch.from_numpy(this_egoexo_pose).float().to(device) # BS x T x 7
			seq_name = TAKE_UIDS_TO_TAKE_NAMES_MAP[take_uid]
			# print(this_egoexo_trajectory.keys(), this_egoexo_pose.keys())
			
			# seq_name = test_input_data_dict['seq_name'][0]
			
			# # Predict head pose first. 
			# test_output = head_transformer_encoder.forward_for_eval(test_input_data_dict)

			# pred_scale = test_output['pred_scale']

			# normal_input_dict = defaultdict(list) 
			
			# normal_input_head_trans = test_input_data_dict['ori_slam_trans'].to(device)
			# normal_input_dict['head_trans'] = normal_input_head_trans - normal_input_head_trans[:, 0:1, :]
			# normal_input_dict['head_rot_mat'] = test_input_data_dict['ori_slam_rot_mat'].to(device)
			
			# normal_input_dict['ori_head_pose'] = test_input_data_dict['head_pose'].to(device)

			# normal_input_dict['seq_len']= torch.tensor(normal_input_head_trans.shape[1]).float()[None] 
			# test_normal_output = head_normal_transformer.forward_for_eval(normal_input_dict, pred_scale)
		
			# s1_output = defaultdict(list) 
			
			# head_pose = test_normal_output['head_pose'] 
			
			# head_pose = torch.cat((head_pose[:, :, :3].to(device), \
			# test_output['head_pose'][:, :, 3:]), dim=-1)

			# head_vels = get_head_vel(head_pose[0].data.cpu().numpy())

			# s1_output['head_pose'] = head_pose.double() 
		
			# s1_output['head_vels'] = torch.from_numpy(head_vels)[None].to(device)
			
			# # Compute the stage 1 head pose estimation metric
			# s1_output['head_pose'][0, :, :2] -= s1_output['head_pose'][0, 0:1, :2].clone()
			# # Need to put human on floor z = 0. 
			# move_to_floor_trans = test_input_data_dict['head_pose'][0, 0:1, :3].to(s1_output['head_pose'].device) - s1_output['head_pose'][0, 0:1, :3]
			# s1_output['head_pose'][0, :, :3] += move_to_floor_trans 
			# s1_output['head_pose'][0, :, 2] -= 0.13 # This values is only for this sequence. 
			
			# s1_head_pred_trans = s1_output['head_pose'][0, :, :3].data.cpu().numpy()
			# s1_head_pred_quat = s1_output['head_pose'][0, :, 3:]
			# s1_head_pred_rot_mat = transforms.quaternion_to_matrix(s1_head_pred_quat).data.cpu().numpy()

			num_try = 1 # 3
		 
			for try_idx in range(num_try):

				sample_bs = 1 # 
				this_egoexo_pose = this_egoexo_pose.repeat(sample_bs, 1, 1) # BS X T X 7 

				ori_local_aa_rep, ori_global_root_jpos = \
				diffusion_trainer.full_body_gen_cond_head_pose_sliding_window(\
				this_egoexo_pose, seq_name) 
				# BS X T X 22 X 3, BS X T X 3 
				
				# Get global joint positions using fk 
				pred_fk_jrot, pred_fk_jpos = diffusion_trainer.ds.fk_smpl(ori_global_root_jpos.reshape(-1, 3), \
				ori_local_aa_rep.reshape(-1, 22, 3))
				# (BS*T) X 22 X 4, (BS*T) X 22 X 3 
				pred_fk_jrot = pred_fk_jrot.reshape(sample_bs, -1, 22, 4) # BS X T X 22 X 4 
				pred_fk_jpos = pred_fk_jpos.reshape(sample_bs, -1, 22, 3) # BS X T X 22 X 3 

				pred_move_trans = pred_fk_jpos[:, 0:1, 15:16, :].clone()  # BS X 1 X 1 X 3 
				pred_move_trans[:, :, :, 2] *= 0 

				pred_fk_jpos = pred_fk_jpos - pred_move_trans # BS X T X 22 X 3 
			   
				ori_global_root_jpos = pred_fk_jpos[:, :, 0, :].clone() # BS X T X 3 
			   
				for s_idx in range(sample_bs):
					# Process Predicted data to touch floor z = 0, and move thead init head translation xy to 0. 
					pred_floor_height, _, _ = determine_floor_height_and_contacts(pred_fk_jpos[s_idx].data.cpu().numpy(), fps=30)
					# print("pred floor height:{0}".format(pred_floor_height)) 

					ori_global_root_jpos[s_idx, :, 2] -= pred_floor_height 
				
					curr_head_global_jpos = pred_fk_jpos[s_idx, :, 15, :] # T X 3 

			if opt.gen_vis:
				vis_head_pose = True  

				# seq_name = test_input_data_dict['seq_name'][0]

				dest_vis_folder = os.path.join(demo_save_dir, "egoego_demo_on_egoexo_{0}".format(seq_name))

				if not os.path.exists(dest_vis_folder):
					os.makedirs(dest_vis_folder)

				vis_tag = "egoego_demo"
			   
				curr_seq_name = seq_name.replace(" ", "")
			   
				mesh_jnts, mesh_verts = diffusion_trainer.gen_full_body_vis(ori_global_root_jpos[0], \
				ori_local_aa_rep[0], dest_vis_folder, curr_seq_name)

				if vis_head_pose:
					vis_head_v_idx = 444 

					align_init_head_trans = mesh_verts[0, 0:1, vis_head_v_idx, :].detach().cpu().numpy() - this_egoexo_pose[0, 0:1, :3].detach().cpu().numpy() # 1 X 3 
					tmp_head_trans = this_egoexo_pose[0, :, :3].detach().cpu().numpy() + align_init_head_trans # T X 3

					dest_head_pose_npy_path = os.path.join(dest_vis_folder, curr_seq_name+"_head_pose.npy")
					head_save_data = np.concatenate((tmp_head_trans, \
					this_egoexo_pose[0, :, 3:].detach().cpu().numpy()), axis=-1) # T X 7 
					np.save(dest_head_pose_npy_path, head_save_data)

					dest_obj_out_folder = os.path.join(dest_vis_folder, curr_seq_name, "objs")
					# dest_out_vid_path = os.path.join(dest_vis_folder, curr_seq_name+"_human_w_head_pose.mp4")
					# run_blender_rendering_and_save2video_head_pose(dest_head_pose_npy_path, dest_obj_out_folder, \
					# dest_out_vid_path, vis_head_only=False, scene_blend_path=osp.join(BLENDER_UTILS_PATH, "floor_colorful_mat_human_w_head_pose_hres.blend"))

					dest_out_head_only_vid_path = os.path.join(dest_vis_folder, curr_seq_name+"_head_pose_only.mp4")
					run_blender_rendering_and_save2video_head_pose(dest_head_pose_npy_path, dest_obj_out_folder, \
					dest_out_head_only_vid_path, vis_head_only=True, scene_blend_path=osp.join(BLENDER_UTILS_PATH, "floor_colorful_mat_human_w_head_pose_hres.blend"))

def parse_opt():
	parser = argparse.ArgumentParser()

	parser.add_argument('--kinpoly_cfg', type=str, default="", help='Path to option JSON file.')

	parser.add_argument("--gen_vis", action="store_true")

	parser.add_argument('--data_root_folder', default='', help='folder to data')
  
	# HeadNet settings. 
	parser.add_argument('--project', default='', help='project/name')
	parser.add_argument('--exp_name', default='exp', help='save to project/name')

	parser.add_argument('--workers', type=int, default=0, help='the number of workers for data loading')
	parser.add_argument('--device', default='0', help='cuda device')
	parser.add_argument('--batch_size', type=int, default=32, help='batch size')

	parser.add_argument('--window', type=int, default=90, help='horizon')
	parser.add_argument('--n_dec_layers', type=int, default=2, help='the number of decoder layers')
	parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
	parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
	parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
	parser.add_argument('--d_model', type=int, default=256, help='the dimension of intermediate representation in transformer')

	parser.add_argument('--weight', default='latest')

	parser.add_argument('--dist_scale', type=float, default=10.0, help='scale for prediction of distance scalar')

	parser.add_argument("--freeze_of_cnn", action="store_true")
	parser.add_argument("--input_of_feats", action="store_true")    

	# GravityNet settings. 
	parser.add_argument('--normal_project', default='', help='project/name')
	parser.add_argument('--normal_exp_name', default='exp', help='save to project/name')

	parser.add_argument('--normal_window', type=int, default=90, help='horizon')
	parser.add_argument('--normal_n_dec_layers', type=int, default=4, help='the number of decoder layers')
	parser.add_argument('--normal_n_head', type=int, default=4, help='the number of heads in self-attention')
	parser.add_argument('--normal_d_k', type=int, default=256, help='the dimension of keys in transformer')
	parser.add_argument('--normal_d_v', type=int, default=256, help='the dimension of values in transformer')
	parser.add_argument('--normal_d_model', type=int, default=256, help='the dimension of intermediate representation in transformer')

	parser.add_argument('--normal_weight', default='latest')

	# Diffusion model settings
	parser.add_argument('--diffusion_window', type=int, default=80, help='horizon')

	parser.add_argument('--diffusion_batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--diffusion_learning_rate', type=float, default=2e-4, help='generator_learning_rate')

	parser.add_argument('--diffusion_n_dec_layers', type=int, default=4, help='the number of decoder layers')
	parser.add_argument('--diffusion_n_head', type=int, default=4, help='the number of heads in self-attention')
	parser.add_argument('--diffusion_d_k', type=int, default=256, help='the dimension of keys in transformer')
	parser.add_argument('--diffusion_d_v', type=int, default=256, help='the dimension of values in transformer')
	parser.add_argument('--diffusion_d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
	
	parser.add_argument('--diffusion_project', default='runs/train', help='project/name')
	parser.add_argument('--diffusion_exp_name', default='', help='save to project/name')

	# For data representation
	parser.add_argument("--canonicalize_init_head", action="store_true")
	parser.add_argument("--use_min_max", action="store_true")

	opt = parser.parse_args()
	return opt

if __name__ == "__main__":
	opt = parse_opt()
	opt.save_dir = str(Path(opt.project) / opt.exp_name) # For head pose estimation model 
	opt.normal_save_dir = str(Path(opt.normal_project) / opt.normal_exp_name)
	opt.diffusion_save_dir = str(Path(opt.diffusion_project) / opt.diffusion_exp_name)
	device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
	test(opt, device)
	