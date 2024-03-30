from __future__ import absolute_import, division

import numpy as np

from .camera import world_to_camera, normalize_screen_coordinates


def create_2d_data(dataset):
    keypoints = {'S1': {},
                 'S5': {},
                 'S6': {},
                 'S7': {},
                 'S8': {}}
    cameras = dataset.cameras()

    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            keypoints[subject][action] = [None, None, None, None]
            pose_2d = np.array(dataset[subject][action]['pose_2d'])
            for cam_idx, kps in enumerate(pose_2d):
                # Normalize camera frame
                cam = cameras[subject][cam_idx]
                pose_2d[cam_idx][..., :2] = normalize_screen_coordinates(pose_2d[cam_idx][..., :2], w=cam['res_w'],
                                                                         h=cam['res_h'])
                pose_2d[cam_idx] = pose_2d[cam_idx] - (
                            pose_2d[cam_idx][:, 11:12, :] + pose_2d[cam_idx][:, 12:13, :]) / 2
                keypoints[subject][action][cam_idx] = pose_2d[cam_idx]
            # print(np.array(keypoints[subject][action]).shape)

    return keypoints


def read_3d_data(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = np.array(dataset[subject][action]['positions_3d'])
            anim = anim - (anim[:, :, 11:12, :] + anim[:, :, 12:13, :]) / 2

            anim = anim / 1000.0

            dataset[subject][action]['positions_3d'] = anim

    return dataset


def fetch(subjects, dataset, keypoints, action_filter=None, stride=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []
    out_camera_params = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action.split(' ')[0] == a:
                        found = True
                        break
                if not found:
                    continue

            cams = dataset.cameras()[subject]
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_actions.append([action.split(' ')[0]] * poses_2d[i].shape[0])
                out_camera_params.append([cams[i]['intrinsic']] * poses_2d[i].shape[0])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_poses_3d, out_poses_2d, out_actions, out_camera_params
