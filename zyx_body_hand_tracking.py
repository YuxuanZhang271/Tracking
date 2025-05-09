import argparse
import cv2
import json
import numpy as np
import os
from pathlib import Path
import torch
from typing import Dict, Optional

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT as hmr2_D
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset as hmr2_V, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT as hamer_D
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset as hamer_V, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel

from detectron2 import model_zoo
from detectron2.config import get_cfg


LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)


def main(args):
    # download and load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    hmr2_model, hmr2_model_cfg = load_hmr2(hmr2_D)

    download_models(CACHE_DIR_HAMER)
    hamer_model, hamer_model_cfg = load_hamer(hamer_D)

    # setup models
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    hmr2_model = hmr2_model.to(device)
    hmr2_model.eval()

    hamer_model = hamer_model.to(device)
    hamer_model.eval()

    # load detector
    detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
    detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # setup the renderer
    hmr2_renderer = Renderer(hmr2_model_cfg, faces=hmr2_model.smpl.faces)
    hamer_renderer = Renderer(hamer_model_cfg, faces=hamer_model.mano.faces)

    # make output directory
    os.makedirs(args.out_folder, exist_ok=True)
    subfolder_path = os.path.join(args.out_folder, Path(args.img_folder).parent.name)
    os.makedirs(subfolder_path, exist_ok=False)
    body = os.path.join(subfolder_path, 'body')
    os.makedirs(body, exist_ok=True)
    hand = os.path.join(subfolder_path, 'hand')
    os.makedirs(hand, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in ['*.jpg', '*.png'] for img in Path(args.img_folder).glob(end)]

    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))

        # detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # hmr2
        hmr2_dataset = hmr2_V(hmr2_model_cfg, img_cv2, pred_bboxes)
        hmr2_dataloader = torch.utils.data.DataLoader(hmr2_dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        for batch in hmr2_dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = hmr2_model(batch)

            pred_cam = out['pred_cam']
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = hmr2_model_cfg.EXTRA.FOCAL_LENGTH / hmr2_model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                regression_img = hmr2_renderer(out['pred_vertices'][n].detach().cpu().numpy(), 
                                               out['pred_cam_t'][n].detach().cpu().numpy(), 
                                               batch['img'][n], 
                                               mesh_base_color=LIGHT_BLUE, 
                                               scene_bg_color=(1, 1, 1)
                                               )
                final_img = np.concatenate([input_patch, regression_img], axis=1)
                cv2.imwrite(os.path.join(body, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                output_params = {}
                output_params["scaled_focal_length"] = float(scaled_focal_length)
                output_params["pred_cam_t_full"] = pred_cam_t_full.tolist()

                for key, val in out.items():
                    if isinstance(val, dict):
                        nested = {}
                        for subk, subv in val.items():
                            nested[subk] = subv[n].detach().cpu().numpy().tolist()
                        output_params[key] = nested
                    else:
                        output_params[key] = val[n].detach().cpu().numpy().tolist()

                json_path = os.path.join(body, f'{img_fn}_{person_id}.json')
                with open(json_path, 'w') as f:
                    json.dump(output_params, f, indent=4)
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
        
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = hmr2_renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], **misc_args)
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
            cv2.imwrite(os.path.join(body, f'{img_fn}_all.png'), 255*input_img_overlay[:, :, ::-1])

        # hamer
        vitposes_out = cpm.predict_pose(img, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)])

        bboxes = []
        is_right = []

        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        hamer_dataset = hamer_V(hamer_model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
        hamer_dataloader = torch.utils.data.DataLoader(hamer_dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in hamer_dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = hamer_model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = hamer_model_cfg.EXTRA.FOCAL_LENGTH / hamer_model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                regression_img = hamer_renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        )
                final_img = np.concatenate([input_patch, regression_img], axis=1)
                cv2.imwrite(os.path.join(hand, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                output_params = {}
                output_params['pred_cam_t_full'] = pred_cam_t_full[n].tolist()
                output_params['scaled_focal_length'] = float(scaled_focal_length)
                for key, val in out.items():
                    if isinstance(val, dict):
                        nested = {}
                        for subk, subv in val.items():
                            nested[subk] = subv[n].detach().cpu().numpy().tolist()
                        output_params[key] = nested
                    else:
                        output_params[key] = val[n].detach().cpu().numpy().tolist()
                json_path = os.path.join(hand, f'{img_fn}_{person_id}.json')
                with open(json_path, 'w') as f:
                    json.dump(output_params, f, indent=4)

                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = hamer_renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
            cv2.imwrite(os.path.join(hand, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--img_folder', type=str, default='images')
    parser.add_argument('-o', '--out_folder', type=str, default='output')
    parser.add_argument('-b', '--batch_size', type=int, default=48)
    args = parser.parse_args()
    main(args)
