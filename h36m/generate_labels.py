import pickle
import numpy as np
import os.path as osp
import json
from scipy.io import loadmat
from subprocess import call
from os import makedirs
import sys

from metadata import load_h36m_metadata
metadata = load_h36m_metadata()


def _infer_box(pose3d, camera, rootIdx):
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[0] -= 1000.0
    tl_joint[1] -= 900.0
    br_joint = root_joint.copy()
    br_joint[0] += 1000.0
    br_joint[1] += 1100.0
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])


def _weak_project(pose3d, fx, fy, cx, cy):
    pose2d = pose3d[:, :2] / pose3d[:, 2:3]
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d


if __name__ == '__main__':
    subject_list = [1, 5, 6, 7, 8, 9, 11]
    action_list = [x for x in range(2, 17)]
    subaction_list = [x for x in range(1, 3)]
    camera_list = [x for x in range(1, 5)]

    train_list = [1, 5, 6, 7, 8]
    test_list = [9, 11]

    joint_idx = [0, 1, 2, 3, 6, 7, 8, 12, 16, 14, 15, 17, 18, 19, 25, 26, 27]

    with open('camera_data.pkl', 'rb') as f:
        camera_data = pickle.load(f)

    train_db = []
    test_db = []
    cnt = 0

    json_data = None
    json_camera = None
    json_joint = None

    for s in subject_list:
        annofile_data = osp.join('annotations', 'Human36M_subject{:d}_data.json'.format(s))
        annofile_camera = osp.join('annotations', 'Human36M_subject{:d}_camera.json'.format(s))
        annofile_joint = osp.join('annotations', 'Human36M_subject{:d}_joint_3d.json'.format(s))

        with open(annofile_data) as json_file:
            json_data = json.load(json_file)

        with open(annofile_camera) as json_file:
            json_camera = json.load(json_file)

        with open(annofile_joint) as json_file:
            json_joint = json.load(json_file)

        for a in action_list:
            for sa in subaction_list:
                for c in camera_list:

                    camera = camera_data[(s, c)]
                    camera_dict = dict()
                    camera_dict['R'] = camera[0]
                    camera_dict['T'] = camera[1]
                    camera_dict['fx'] = camera[2][0]
                    camera_dict['fy'] = camera[2][1]
                    camera_dict['cx'] = camera[3][0]
                    camera_dict['cy'] = camera[3][1]
                    camera_dict['k'] = camera[4]
                    camera_dict['p'] = camera[5]

                    subdir_format = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'
                    subdir = subdir_format.format(s, a, sa, c)

                    """
                    basename = metadata.get_base_filename('S{:d}'.format(s), '{:d}'.format(a), '{:d}'.format(sa), metadata.camera_ids[c-1])
                    annotname = basename + '.cdf'

                    subject = 'S' + str(s)
                    annofile3d = osp.join('extracted', subject, 'Poses_D3_Positions_mono_universal', annotname)
                    annofile3d_camera = osp.join('extracted', subject, 'Poses_D3_Positions_mono', annotname)
                    annofile2d = osp.join('extracted', subject, 'Poses_D2_Positions', annotname)              
                    """




                    """
                    with pycdf.CDF(annofile3d) as data:
                        pose3d = np.array(data['Pose'])
                        pose3d = np.reshape(pose3d, (-1, 32, 3))

                    with pycdf.CDF(annofile3d_camera) as data:
                        pose3d_camera = np.array(data['Pose'])
                        pose3d_camera = np.reshape(pose3d_camera, (-1, 32, 3))

                    with pycdf.CDF(annofile2d) as data:
                        pose2d = np.array(data['Pose'])
                        pose2d = np.reshape(pose2d, (-1, 32, 2))
                    """

                    nposes = json_joint[str(a)][str(sa)].__len__()
                    image_format = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}_{:06d}.jpg'

                    for i in range(nposes):
                        datum = {}
                        imageid = json_data["images"][i]["id"]
                        #imagename = image_format.format(s, a, sa, c, i+1)
                        imagepath = json_data["images"][i]["file_name"] #osp.join(subdir, imagename)
                        frame_idx = json_data["images"][i]["frame_idx"]
                        if not osp.isfile(osp.join('images', imagepath)):
                            print(osp.join('images', imagepath))
                            print(nposes)
                        if osp.isfile(osp.join('images', imagepath)):
                            pose3d = np.array(json_joint[str(a)][str(sa)][str(frame_idx)])
                            pose2d = pose3d[:, :2]

                            datum['image'] = imagepath
                            datum['joints_2d'] = pose2d
                            datum['joints_3d'] = pose3d
                            datum['joints_3d_camera'] = pose3d
                            datum['joints_vis'] = np.ones((17, 3))
                            datum['video_id'] = cnt
                            datum['image_id'] = imageid
                            datum['subject'] = s
                            datum['action'] = a
                            datum['subaction'] = sa
                            datum['camera_id'] = c-1
                            datum['source'] = 'h36m'
                            datum['camera'] = camera_dict

                            box = _infer_box(datum['joints_3d_camera'], camera_dict, 0)
                            center = (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
                            scale = ((box[2] - box[0]) / 200.0, (box[3] - box[1]) / 200.0)
                            datum['center'] = center
                            datum['scale'] = scale
                            datum['box'] = box

                            if s in train_list:
                                train_db.append(datum)
                            else:
                                test_db.append(datum)
                        print(image_format.format(s, a, sa, c, frame_idx))
                    cnt += 1

    with open('h36m_train.pkl', 'wb') as f:
        pickle.dump(train_db, f)

    with open('h36m_valid.pkl', 'wb') as f:
        pickle.dump(test_db, f)






