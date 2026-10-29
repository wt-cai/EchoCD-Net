import os

folder1 = "/mnt/diskb5/caiweitao_private/cwt/EchoSAM/result/EchoNet---EchoSAM/EchoNet-98/vis3/ReinMemSAM"
folder2 = "/mnt/diskb5/caiweitao_private/cwt/MemSAM/result/EchoNet/vis2/MemSAM"

dirs1 = {d for d in os.listdir(folder1) if os.path.isdir(os.path.join(folder1, d))}
dirs2 = {d for d in os.listdir(folder2) if os.path.isdir(os.path.join(folder2, d))}

common = dirs1 & dirs2  # 交集
for d in sorted(common):
    print(d)
    print("kong")