import cv2
import numpy as np

def process_video(input_video_path, output_video_path, target_size=(112, 112), save_frame_at=None):
    # 读取视频文件
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return
    
    # 获取视频帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 准备保存视频的参数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可以选择其他格式，如'XVID'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)
    
    frames = []

    # 读取视频帧并进行下采样
    frame_index = 0
    saved_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 使用三次插值下采样至目标大小
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)
        
        # 将下采样后的帧加入帧列表
        frames.append(resized_frame)
        
        # 写入输出视频
        out.write(resized_frame)

        # 如果需要保存某一帧
        if save_frame_at is not None and frame_index == save_frame_at:
            # 保存原图像
            cv2.imwrite(f"/mnt/diskb5/caiweitao_private/cwt/MemSAM_copy/temp/original_frame_{save_frame_at}.png", frame)  # 保存为 PNG 文件
            print(f"Original Frame {save_frame_at} saved as 'original_frame_{save_frame_at}.png'")
           
            saved_frame = resized_frame
            cv2.imwrite(f"/mnt/diskb5/caiweitao_private/cwt/MemSAM_copy/temp/saved_frame_{save_frame_at}.png", saved_frame)  # 保存为 PNG 文件
            print(f"Frame {save_frame_at} saved as 'saved_frame_{save_frame_at}.png'")
        
        frame_index += 1
    
    cap.release()
    out.release()
    
    # 将帧转换为 NumPy 数组进行进一步处理（例如用于模型输入）
    video_tensor = np.array(frames)
    
    # 标准化视频数据到 [0, 1]
    video_tensor = video_tensor / 255.0
    
    # 如果需要标准化到 [-1, 1]，可以使用下面的代码：
    # video_tensor = (video_tensor / 127.5) - 1.0
    
    print(f"Video processing complete! Processed {frame_count} frames.")
    return video_tensor

# 调用函数
input_video_path = '/mnt/diskb5/caiweitao_private/cwt/dataset/HMC-QU/HMC-QU/A4C/ES0001 _4CH_1.avi'  # 输入视频路径
output_video_path = '/mnt/diskb5/caiweitao_private/cwt/dataset/HMC-QU/HMC-QU/A4C/ES0001 _4CH_1_new.avi'  # 输出视频路径
frame_to_save = 10  # 设置要保存的帧索引，例如第 100 帧

# 调用处理函数
processed_video = process_video(input_video_path, output_video_path, save_frame_at=frame_to_save)

# 如果你需要查看处理后的某个帧，可以使用以下代码：
# cv2.imshow("Processed Frame", processed_video[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
