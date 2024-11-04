from __future__ import division
import os
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import core.paegmm.kddcup10.kddcup10_pae_gmm as paegmm
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
print(tf.__version__)
print(tf.test.gpu_device_name())
print(tf.config.experimental.set_visible_devices)
print('GPU:', tf.config.list_physical_devices('GPU'))
print('CPU:', tf.config.list_physical_devices(device_type='CPU'))
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available())
# 输出可用的GPU数量
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# 查询GPU设备


if __name__ == '__main__':
    
    # data_path
    Coil20 = ['../Data/WebKB4.mat', 20, 'coil20']
    
    filename   = Coil20[0]
    num_clus_r = Coil20[1]
    num_clus_c = Coil20[1]

    ae_config = [1000, 500, 200, 100, 40]

    ae_col_config = [4199, 1000, 500, 200, 40]

    gmm_config = [[num_clus_r, 5], 40, 160, 80, 40, num_clus_r]

    accuracy     = []
    NMI          = []

    rounds = 10
    epochs = 3000
    epochs_pretrain = 1000
    for k in range(rounds):
        tf.reset_default_graph()
        machine = paegmm.KddcupPaeGmm(1024, num_clus_r, num_clus_c, ae_config, ae_col_config, gmm_config, 0)
        data = machine.get_data(filename)
        acc, nmi = machine.run(data, epochs, epochs_pretrain)

        accuracy      = np.append(accuracy, acc)
        NMI           = np.append(NMI, nmi)
