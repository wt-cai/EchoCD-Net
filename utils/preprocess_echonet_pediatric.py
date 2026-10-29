import os
import sys
import argparse
import pandas as pd
import imageio
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils_contour import trace_to_mask


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input_dir', type=str)
    # parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-kpts', '--save_kpts', type=bool, default=True)
    parser.add_argument('-masks', '--save_masks', type=bool, default=True)
    parser.add_argument('-imgs', '--save_imgs', type=bool, default=True)
    args = parser.parse_args()

    return args


def loadvideo(filename: str) -> np.ndarray:
    """
    Function taken from the echonet repository https://github.com/echonet/dynamic
    Loads a video from a file.
    Args:
        filename (str): filename of video
    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.
    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError(f"Failed to load frame #{count} of {filename}.")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count, :, :] = frame

    v = v.transpose((3, 0, 1, 2))

    return v

def preprocess_data_A4C(input_path, output_path, save_kpts, save_masks, save_imgs):
    file_dir = '/mnt/diskb5/caiweitao_private/cwt/MemSAM/dataset/EchoNet-pediatric'

    if not os.path.exists(input_path + '/VolumeTracings.csv'):
        raise ValueError('Directory does not contain the volume tracing csv file')
    if not os.path.exists(input_path + '/FileList.csv'):
        raise ValueError('Directory does not contain the file list csv file')

    echonet_pts = pd.read_csv(input_path + '/VolumeTracings.csv')
    # print(echonet_pts)
    echonet_pts.head()
    with open(os.path.join(input_path, "FileList.csv")) as f:
        data = pd.read_csv(f)
    data['FileName'] = data['FileName'].str.replace('.avi$', '', regex=True)
    fnames = data["FileName"].tolist()
    # print(fnames)
    fnames = [fn + ".avi" for fn in fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix

    x_train = [data["FileName"][idx] for idx in data.index if data["Split"][idx] == 'TRAIN']
    x_val = [data["FileName"][idx] for idx in data.index if data["Split"][idx] == 'VAL']
    x_test = [data["FileName"][idx] for idx in data.index if data["Split"][idx] == 'TEST']

    # save two single frames per cycle
    output_list_train = []
    output_list_test = []
    output_list_val = []
    output_list_invalid = []

    # save full cycle
    output_list_test_cycle = []
    output_list_train_cycle = []
    output_list_val_cycle = []

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    nnum = 0
    for fname in fnames:
        name = fname.split('.')[0]
        # ef = data[data['FileName'] == name]['EF'].item()
        # esv = data[data['FileName'] == name]['ESV'].item()
        # edv = data[data['FileName'] == name]['EDV'].item()
        patient_df = echonet_pts[echonet_pts.FileName == fname]
        video_path = os.path.join(input_path, "Videos", fname)

        # check whether file exists
        if not os.path.exists(video_path):
            print('Not found: ', video_path)
            output_list_invalid.append(f"{name}")
            continue

        video = loadvideo(video_path).astype(np.uint8)
        print(nnum, fname)
        nnum += 1

        pts_pairs = []
        frame_pairs = {}
        frame_pairs_mask = {}
        vol_pairs = []
        frames = []
        for frame in set(patient_df.Frame):
            frame_df = patient_df[patient_df.Frame == frame]
            gt_pts = np.array(frame_df.loc[:, "X":"Y"])

            # check whether frame contains a strange number of points
            # if ((np.size(gt_pts) / 2) != 42):
            #     output_list_invalid.append(f"{name}")
            #     break

            mask = trace_to_mask(gt_pts, (112, 112))
            vol_pairs.append(np.sum(mask))
            frame = int(frame)
            frames.append(int(frame))

            pts = np.array([gt_pts[:, 0], gt_pts[:, 1]]).transpose()

            if name in x_test:
                output_list_test.append(f"{name}_{frame}.png")
                frames_folder = os.path.join(output_path, "images/test/")
                anno_folder = os.path.join(output_path, "annotations/test/")
                frames_cycle_folder = os.path.join(output_path, "echocycle/videos/test/")
                anno_cycle_folder = os.path.join(output_path, "echocycle/annotations/test/")

            if name in x_train:
                output_list_train.append(f"{name}_{frame}.png")
                frames_folder = os.path.join(output_path, "images/train/")
                anno_folder = os.path.join(output_path, "annotations/train/")
                frames_cycle_folder = os.path.join(output_path, "echocycle/videos/train/")
                anno_cycle_folder = os.path.join(output_path, "echocycle/annotations/train/")

            if name in x_val:
                output_list_val.append(f"{name}_{frame}.png")
                frames_folder = os.path.join(output_path, "images/val/")
                anno_folder = os.path.join(output_path, "annotations/val/")
                frames_cycle_folder = os.path.join(output_path, "echocycle/videos/val/")
                anno_cycle_folder = os.path.join(output_path, "echocycle/annotations/val/")

            
            if frame < 0 or frame >= video.shape[1]:
                continue
            
            frame_img = video[:, frame, :, :].transpose(1, 2, 0)
            frame_pairs[str(frame)] = frame_img
            frame_pairs_mask[str(frame)] = mask
            
            file_dir = output_path
            

            if not os.path.exists(frames_folder):
                os.makedirs(frames_folder)
            if not os.path.exists(anno_folder):
                os.makedirs(anno_folder)
            if not os.path.exists(frames_cycle_folder):
                os.makedirs(frames_cycle_folder)
            if not os.path.exists(anno_cycle_folder):
                os.makedirs(anno_cycle_folder)

            pts_pairs.append(pts)
            if save_imgs:
                imageio.imsave(frames_folder + f"{name}_{frame}.png", frame_img)
            if save_masks:
                imageio.imsave(anno_folder + f"{name}_{frame}.png", mask)
            # if save_kpts:
            #     np.savez(anno_folder + f"{name}_{frame}", mask=mask,
            #              kpts=np.asarray(pts), ef=ef)

            if len(pts_pairs) == 2:
                np.save(frames_cycle_folder + f"{name}", video)
                
                np.savez(anno_cycle_folder+ f"{name}",
                            fnum=frame_pairs,
                            fnum_mask=frame_pairs_mask,
                            # kpts=np.array(pts_pairs), 
                            # ef=ef, edv=float(edv), esv=float(esv),
                            spacing=(1.0,1.0,1.0))

                if name in x_train:
                    output_list_train_cycle.append(f"{name}.npy")
                if name in x_val:
                    output_list_val_cycle.append(f"{name}.npy")
                if name in x_test:
                    output_list_test_cycle.append(f"{name}.npy")

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    # output_file_train = open(file_dir + '/echonet_train_filenames.txt','w')
    # output_file_val = open(file_dir + '/echonet_val_filenames.txt', 'w')
    # output_file_test = open(file_dir + '/echonet_test_filenames.txt', 'w')
    # output_file_invalid = open(file_dir + '/echonet_invalid_filenames.txt','w')

    # output_file_train_cycle = open(file_dir + '/echocycle/echonet_cycle_train_filenames.txt', 'w')
    # output_file_val_cycle = open(file_dir + '/echocycle/echonet_cycle_val_filenames.txt', 'w')
    # output_file_test_cycle = open(file_dir + '/echocycle/echonet_cycle_test_filenames.txt', 'w')

    # for name in output_list_train_cycle:
    #     if name not in output_list_invalid:
    #         output_file_train_cycle.write(name + '\n')
    # for name in output_list_test_cycle:
    #     if name not in output_list_invalid:
    #         output_file_test_cycle.write(name + '\n')
    # for name in output_list_val_cycle:
    #     if name not in output_list_invalid:
    #         output_file_val_cycle.write(name + '\n')

    
    # for name in output_list_test:
    #     if name.split('_')[0] not in output_list_invalid:
    #         output_file_test.write(name + '\n')
    # for name in output_list_train:
    #     if name.split('_')[0] not in output_list_invalid:
    #         output_file_train.write(name + '\n')
    # for name in output_list_val:
    #     if name.split('_')[0] not in output_list_invalid:
    #         output_file_val.write(name + '\n')
    # for name in output_list_invalid:
    #     output_file_invalid.write(name + '\n')

    # output_file_test.close()
    # output_file_train.close()
    # output_file_val.close()
    # output_file_invalid.close()

    # output_file_test_cycle.close()
    # output_file_train_cycle.close()
    # output_file_val_cycle.close()
            



def preprocess_data_PSAX(input_path, output_path,save_imgs=True, save_masks=True):
    # 检查文件是否存在
    if not os.path.exists(os.path.join(input_path, 'VolumeTracings.csv')):
        raise ValueError('Directory does not contain the volume tracing csv file')
    if not os.path.exists(os.path.join(input_path, 'FileList.csv')):
        raise ValueError('Directory does not contain the file list csv file')

    echonet_pts = pd.read_csv(os.path.join(input_path, 'VolumeTracings.csv'))
    data = pd.read_csv(os.path.join(input_path, "FileList.csv"))
    data['FileName'] = data['FileName'].str.replace('.avi$', '', regex=True)

    # 折数转化为标签
    split_map = {
        'TRAIN': list(range(0, 7)),  # Split in 0-5
        'VAL': [7],
        'TEST': [8, 9, 10]
    }
    x_train = [data["FileName"][idx] for idx in data.index if data["Split"][idx] in split_map['TRAIN']]
    x_val = [data["FileName"][idx] for idx in data.index if data["Split"][idx] in split_map['VAL']]
    x_test = [data["FileName"][idx] for idx in data.index if data["Split"][idx] in split_map['TEST']]


    all_fnames = data["FileName"].tolist()
    fnames = [fn + ".avi" for fn in all_fnames]

    # 准备输出路径
    os.makedirs(output_path, exist_ok=True)

    output_list_train = []
    output_list_test = []
    output_list_val = []
    output_list_invalid = []

    output_list_test_cycle = []
    output_list_train_cycle = []
    output_list_val_cycle = []

    nnum = 0
    for fname in fnames:
        name = fname.replace(".avi", "")
        patient_df = echonet_pts[echonet_pts.FileName == fname]
        video_path = os.path.join(input_path, "Videos", fname)

        if not os.path.exists(video_path):
            print(f"[缺失视频] {video_path}")
            output_list_invalid.append(name)
            continue

        video = loadvideo(video_path).astype(np.uint8)
        print(f"[{nnum}] 处理视频：{fname}")
        nnum += 1

        frames_folder = anno_folder = None
        frames_cycle_folder = anno_cycle_folder = None

        if name in x_test:
            frames_folder = os.path.join(output_path, "images/test/")
            anno_folder = os.path.join(output_path, "annotations/test/")
            frames_cycle_folder = os.path.join(output_path, "echocycle/videos/test/")
            anno_cycle_folder = os.path.join(output_path, "echocycle/annotations/test/")
        elif name in x_train:
            frames_folder = os.path.join(output_path, "images/train/")
            anno_folder = os.path.join(output_path, "annotations/train/")
            frames_cycle_folder = os.path.join(output_path, "echocycle/videos/train/")
            anno_cycle_folder = os.path.join(output_path, "echocycle/annotations/train/")
        elif name in x_val:
            frames_folder = os.path.join(output_path, "images/val/")
            anno_folder = os.path.join(output_path, "annotations/val/")
            frames_cycle_folder = os.path.join(output_path, "echocycle/videos/val/")
            anno_cycle_folder = os.path.join(output_path, "echocycle/annotations/val/")
        else:
            print(f"[跳过] 未知 split 标签：{name}")
            output_list_invalid.append(name)
            continue

        # 创建输出目录
        for folder in [frames_folder, anno_folder, frames_cycle_folder, anno_cycle_folder]:
            os.makedirs(folder, exist_ok=True)

        pts_pairs = []
        frame_pairs = {}
        frame_pairs_mask = {}

        for frame in sorted(set(patient_df.Frame)):
            frame_df = patient_df[patient_df.Frame == frame]
            gt_pts = np.array(frame_df.loc[:, "X":"Y"])
            mask = trace_to_mask(gt_pts, (112, 112))
            frame = int(frame)

            if frame < 0 or frame >= video.shape[1]:
                continue

            frame_img = video[:, frame, :, :].transpose(1, 2, 0)
            frame_pairs[str(frame)] = frame_img
            frame_pairs_mask[str(frame)] = mask
            pts = np.array([gt_pts[:, 0], gt_pts[:, 1]]).transpose()
            pts_pairs.append(pts)

            fname_frame = f"{name}_{frame}.png"
            if name in x_train:
                output_list_train.append(fname_frame)
            elif name in x_val:
                output_list_val.append(fname_frame)
            elif name in x_test:
                output_list_test.append(fname_frame)

            if save_imgs:
                imageio.imsave(os.path.join(frames_folder, fname_frame), frame_img)
            if save_masks:
                imageio.imsave(os.path.join(anno_folder, fname_frame), mask)

        if len(pts_pairs) == 2:
            np.save(os.path.join(frames_cycle_folder, f"{name}.npy"), video)
            np.savez(os.path.join(anno_cycle_folder, f"{name}.npz"),
                     fnum=frame_pairs,
                     fnum_mask=frame_pairs_mask,
                     spacing=(1.0, 1.0, 1.0))

            if name in x_train:
                output_list_train_cycle.append(f"{name}.npy")
            elif name in x_val:
                output_list_val_cycle.append(f"{name}.npy")
            elif name in x_test:
                output_list_test_cycle.append(f"{name}.npy")

    print("\n✅ 处理完成")
    print(f"训练集：{len(output_list_train)} 张帧图像 / {len(output_list_train_cycle)} 个周期")
    print(f"验证集：{len(output_list_val)} 张帧图像 / {len(output_list_val_cycle)} 个周期")
    print(f"测试集：{len(output_list_test)} 张帧图像 / {len(output_list_test_cycle)} 个周期")
    print(f"跳过（无split或无视频）：{len(output_list_invalid)}")



if __name__ == '__main__':
    args = parse_args()

    # if not os.path.exists(args.input_dir):
    #     raise ValueError('Input directory does not exist.')
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    # preprocess_data(input_path=args.input_dir,
    #                 output_path=args.output_dir,
    #                 save_kpts=args.save_kpts,
    #                 save_masks=args.save_masks,
    #                 save_imgs=args.save_imgs)

    # preprocess_data_A4C(input_path='/mnt/diskb5/caiweitao_private/cwt/dataset/EchoNet-pediatric/A4C',
    #                 output_path='/mnt/diskb5/caiweitao_private/cwt/MemSAM/dataset/EchoNet-pediatric2',
    #                 save_kpts=args.save_kpts,
    #                 save_masks=args.save_masks,
    #                 save_imgs=args.save_imgs)

    preprocess_data_PSAX(input_path='/mnt/diskb5/caiweitao_private/cwt/dataset/EchoNet-pediatric/PSAX',
                    output_path='/mnt/diskb5/caiweitao_private/cwt/MemSAM/dataset/EchoNet-pediatric-PSAX',
                    save_masks=args.save_masks,
                    save_imgs=args.save_imgs)
