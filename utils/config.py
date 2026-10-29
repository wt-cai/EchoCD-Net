# This file is used to configure the training parameters for each task
class Config_US30K:
    # This dataset contain all the collected ultrasound dataset
    data_path = "../../dataset/SAMUS/"  
    save_path = "./checkpoints/SAMUS/"
    result_path = "./result/SAMUS/"
    tensorboard_path = "./tensorboard/SAMUS/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 5e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

# -------------------------------------------------------------------------------------------------
class Config_TN3K:
    data_path = "../../dataset/SAMUS/" 
    data_subpath = "../../dataset/SAMUS/ThyroidNodule-TN3K/" 
    save_path = "./checkpoints/TN3K/"
    result_path = "./result/TN3K/"
    tensorboard_path = "./tensorboard/TN3K/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train-ThyroidNodule-TN3K"  # the file name of training set
    val_split = "val-ThyroidNodule-TN3K"     # the file name of testing set
    test_split = "test-ThyroidNodule-TN3K"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_BUSI:
    # This dataset is for breast cancer segmentation
    data_path = "../../dataset/SAMUS/"
    data_subpath = "../../dataset/SAMUS/Breast-BUSI/"   
    save_path = "./checkpoints/BUSI/"
    result_path = "./result/BUSI/"
    tensorboard_path = "./tensorboard/BUSI/"
    load_path = save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-Breast-BUSI"   # the file name of training set
    val_split = "val-Breast-BUSI"       # the file name of testing set
    test_split = "test-Breast-BUSI"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_CAMUS:
    # This dataset is for breast cancer segmentation
    # data_path = "./dataset/SAMUS/CAMUS"  # 
    data_path = "./dataset/CAMUS"  #
    data_subpath = "CAMUS" 
    save_path = "./checkpoints/CAMUS/"
    result_path = "./result/CAMUS/"
    tensorboard_path = "./tensorboard/CAMUS/"
    load_path = save_path + "xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 400                        # number of total epochs to run (default: 400)
    batch_size = 8                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 4                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-EchocardiographyLA-CAMUS"   # the file name of training set
    val_split = "val-EchocardiographyLA-CAMUS"       # the file name of testing set
    test_split = "test-Echocardiography-CAMUS"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camusmulti"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_EchoNet():
    # data_path = "./dataset/SAMUS/EchoNet"  # 
    data_path = "./dataset/EchoNet"
    data_subpath = "EchoNet" 
    save_path = "./checkpoints/EchoNet/"
    result_path = "./result/EchoNet/"
    tensorboard_path = "./tensorboard/EchoNet/"
    load_path = save_path + "SAMUS_10181927_95_0.9257182998911371.pth"
    save_path_code = "_"

    workers = 8                         # number of data loading workers (default: 8)
    epochs = 100                        # number of total epochs to run (default: 400)
    batch_size = 16                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "echonet_train_filenames"   # the file name of training set
    val_split = "echonet_val_filenames"       # the file name of testing set
    test_split = "echonet_test_filenames"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "echonet"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"

class Config_EchoNet_Video():
    data_path = "/mnt/diskb5/caiweitao_private/cwt/MemSAM/dataset/EchoNet/echocycle"
    # data_path = "/mnt/diskb5/caiweitao_private/cwt/dataset/test_data/EchoNet/echocycle"
    data_subpath = "EchoNet" 
    save_path = "/mnt/diskb5/caiweitao_private/cwt/MemSAM_copy/checkpoints/EchoNet_2025_3_29/SACFBAdapter/"
    result_path = "/mnt/diskb5/caiweitao_private/cwt/EchoSAM/result/EchoNet-pediatric---EchoSAM/echonet/"
    tensorboard_path = "/mnt/diskb5/caiweitao_private/cwt/MemSAM_copy/tensorboard/EchoNet_2025_4_8/SACFBAdapter/"
    # load_path = "/mnt/diskb5/caiweitao_private/cwt/MemSAM_copy/checkpoints/EchoNet_2025_2_19/ReinMemSAM_02190233_19_0.9449509960357537.pth"
    load_path = save_path + "ReinMemSAM_04120207_19_0.9211755007656589.pth"
    save_path_code = "_"

    workers = 8                         # number of data loading workers (default: 8)
    epochs = 100                         # number of total epochs to run (default: 400)
    batch_size = 16                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"   # the file name of t33raining set
    val_split = "val"       # the file name of testing set
    test_split = "test"     # the file name of testing set 
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "echonet"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"
    semi = False
    compute_ef = False

class Config_EchoNet_pediatric():

    data_path = "/mnt/diskb5/caiweitao_private/cwt/MemSAM/dataset/EchoNet-pediatric-A4C/echocycle"
    data_subpath = "EchoNet-pediatric" 
    save_path = "/mnt/diskb5/caiweitao_private/cwt/EchoSAM/checkpoints/EchoNet-pediatric_2025_6_9/"
    result_path = "/mnt/diskb5/caiweitao_private/cwt/seg_model_EchoSAM/temp/EchoNet-pediatric_memsam/"
    tensorboard_path = "/mnt/diskb5/caiweitao_private/cwt/EchoSAM/tensorboard/EchoNet-pediatric_2025_6_9/"
    load_path = save_path + "SAMUS_10081703_24_0.9262574595178807.pth"
    save_path_code = "_"

    workers = 8                         # number of data loading workers (default: 8)
    epochs = 500                       # number of total epochs to run (default: 400)
    batch_size = 16                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"   # the file name of training set
    val_split = "val"       # the file name of testing set
    test_split = "test"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "echonet_pediatric"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"
    semi = False
    compute_ef = False



class Config_CAMUS_Video_Full():
    # data_path = "./dataset/SAMUS/CAMUS_full"  # 
    data_path = "/mnt/diskb5/caiweitao_private/cwt/MemSAM/dataset/CAMUS"  #
    data_subpath = "CAMUS_full" 
    save_path = "/mnt/diskb5/caiweitao_private/cwt/EchoSAM/result/Ablation/rein_se/CAMUS/"
    result_path = "/mnt/diskb5/caiweitao_private/cwt/EchoGD-Net/result/Temporal_Consistency/CAMUS/"
    tensorboard_path = "/mnt/diskb5/caiweitao_private/cwt/MemSAM_copy/tensorboard/CAMUS_full_2025_5_2/"
    load_path = save_path + "SAMUS_10081703_24_0.9262574595178807.pth"
    save_path_code = "_"

    workers = 8                         # number of data loading workers (default: 8)
    epochs = 1000                        # number of total epochs to run (default: 400)
    batch_size = 16                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"   # the file name of training set
    val_split = "val"       # the file name of testing set
    test_split = "test"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camus"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"
    compute_ef = False



class Config_HMCQU:
    # This dataset contain all the collected ultrasound dataset
    data_path = "/mnt/diskb5/caiweitao_private/cwt/MemSAM/dataset/HMCQU"  #
    data_subpath = "HMCQU" 
    save_path = "./checkpoints/HMCQU/"
    result_path = "/mnt/diskb5/caiweitao_private/cwt/MemSAM_copy/result/HMCQU4/"
    tensorboard_path = "./tensorboard/HMCQU/"
    load_path = save_path + "SAMUS_10081703_24_0.9262574595178807.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 100                        # number of total epochs to run (default: 400)
    batch_size = 1                    # batch size (default: 4)
    learning_rate = 5e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "hmcqu"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"
    semi = False



class Config_HMCQU_GD2H:
    # This dataset contain all the collected ultrasound dataset
    data_path = "/mnt/diskb5/caiweitao_private/cwt/MemSAM/dataset/HMCQU_GD2H"  #
    data_subpath = "HMCQU_GD2H" 
    save_path = "/mnt/diskb5/caiweitao_private/cwt/MemSAM_copy/checkpoints/HMCQU_GD2H/"
    result_path = "/mnt/diskb5/caiweitao_private/cwt/EchoSAM/result/Ablation/rein_eca/HMCQU-GD2H/"
    tensorboard_path = "./tensorboard/HMCQU_GD2H/"
    load_path = save_path + "SAMUS_10081703_24_0.9262574595178807.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 100                        # number of total epochs to run (default: 400)
    batch_size = 1                    # batch size (default: 4)
    learning_rate = 5e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "hmcqu_gd2h"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"
    semi = False



class Config_FAHGM_GD2H_LV:
    # This dataset contain all the collected ultrasound dataset
    data_path = "/mnt/diskb5/caiweitao_private/cwt/MemSAM/dataset/FAHGM-GD2H-LV"  #
    data_subpath = "FAHGM-GD2H-LV" 
    save_path = "/mnt/diskb5/caiweitao_private/cwt/MemSAM_copy/checkpoints/FAHGM-GD2H-LV/"
    result_path = "/mnt/diskb5/caiweitao_private/cwt/EchoSAM/result/Ablation/rein_eca/FAHGM-GD2H-LV/"
    tensorboard_path = "./tensorboard/FAHGM-GD2H-LV/"
    load_path = save_path + "SAMUS_10081703_24_0.9262574595178807.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 100                        # number of total epochs to run (default: 400)
    batch_size = 1                    # batch size (default: 4)
    learning_rate = 5e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    test_split = "test"                 # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "fahgm_gd2h"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"
    semi = False



# ==================================================================================================
def get_config(task="US30K"):
    if task == "CAMUS":
        return Config_CAMUS()
    elif task == "EchoNet":
        return Config_EchoNet()
    elif task == "CAMUS_Video_Full":
        return Config_CAMUS_Video_Full()
    elif task == "EchoNet_Video":
        return Config_EchoNet_Video()
    elif task == "EchoNet_pediatric":
        return Config_EchoNet_pediatric()
    elif task == "HMCQU_GD2H":
        return Config_HMCQU_GD2H()
    elif task == "FAHGM_GD2H_LV":
        return Config_FAHGM_GD2H_LV()
    

    else:
        assert("We do not have the related dataset, please choose another task.")
