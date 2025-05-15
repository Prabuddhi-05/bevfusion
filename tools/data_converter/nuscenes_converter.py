import mmcv
import numpy as np
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

def create_nuscenes_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10,
                          max_radar_sweeps=10):
    """Create info file of nuScenes dataset."""
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits

    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers

    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('Unknown dataset version')

    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = [available_scenes[available_scene_names.index(s)]['token']
                    for s in train_scenes if s in available_scene_names]
    val_scenes = [available_scenes[available_scene_names.index(s)]['token']
                  for s in val_scenes if s in available_scene_names]

    test = 'test' in version
    if test:
        print(f'test scene: {len(train_scenes)}')
    else:
        print(f'train scene: {len(train_scenes)}, val scene: {len(val_scenes)}')

    train_infos, val_infos = _fill_trainval_infos(nusc, train_scenes, val_scenes,
                                                  test, max_sweeps=max_sweeps, max_radar_sweeps=max_radar_sweeps)

    metadata = dict(version=version)

    if test:
        print(f'test sample: {len(train_infos)}')
        data = dict(infos=train_infos, metadata=metadata)
        info_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        mmcv.dump(data, info_path)
    else:
        print(f'{info_prefix}')
        print(f'train sample: {len(train_infos)}, val sample: {len(val_infos)}')

        train_data = dict(infos=train_infos, metadata=metadata)
        val_data = dict(infos=val_infos, metadata=metadata)

        train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
        val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')

        mmcv.dump(train_data, train_path)
        mmcv.dump(val_data, val_path)


def get_available_scenes(nusc):
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        lidar_path, _, _ = nusc.get_sample_data(sd_rec['token'])
        lidar_path = str(lidar_path)
        if os.getcwd() in lidar_path:
            lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
        if not mmcv.is_filepath(lidar_path):
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         max_radar_sweeps=10):
    train_nusc_infos = []
    val_nusc_infos = []
    token2idx = {}

    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', lidar_token)
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        mmcv.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'radars': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'prev_token': sample['prev']
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        camera_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'][cam] = cam_info

        radar_names = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        for radar_name in radar_names:
            radar_token = sample['data'][radar_name]
            radar_rec = nusc.get('sample_data', radar_token)
            sweeps = []
            while len(sweeps) < max_radar_sweeps:
                radar_path, _, _ = nusc.get_sample_data(radar_token)
                radar_info = obtain_sensor2top(nusc, radar_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, radar_name)
                sweeps.append(radar_info)
                if radar_rec['prev'] == '':
                    break
                radar_token = radar_rec['prev']
                radar_rec = nusc.get('sample_data', radar_token)
            info['radars'][radar_name] = sweeps

        sd_rec = nusc.get('sample_data', lidar_token)
        sweeps = []
        while len(sweeps) < max_sweeps:
            if sd_rec['prev'] == '':
                break
            sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
            sweeps.append(sweep)
            sd_rec = nusc.get('sample_data', sd_rec['prev'])
        info['sweeps'] = sweeps

        if not test:
            annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
            velocity = np.array([nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array([(a['num_lidar_pts'] + a['num_radar_pts']) > 0 for a in annotations], dtype=bool).reshape(-1)
            for i in range(len(velocity)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]
            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array([a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
            token2idx[info['token']] = ('train', len(train_nusc_infos) - 1)
        else:
            val_nusc_infos.append(info)
            token2idx[info['token']] = ('val', len(val_nusc_infos) - 1)

    for info in train_nusc_infos:
        prev_token = info['prev_token']
        info['prev'] = -1 if prev_token == '' else token2idx[prev_token][1]

    for info in val_nusc_infos:
        prev_token = info['prev_token']
        info['prev'] = -1 if prev_token == '' else token2idx[prev_token][1]

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type='lidar'):
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:
        data_path = data_path.split(f'{os.getcwd()}/')[-1]
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T

    sweep['sensor2lidar_rotation'] = R.T
    sweep['sensor2lidar_translation'] = T
    return sweep


if __name__ == '__main__':
    create_nuscenes_infos('data/nuscenes/', 'nuscenes')

