import os
import argparse
import numpy as np
import tensorflow.keras as keras

import onnx
import h5py
import hdf5plugin
from onnx2keras import onnx_to_keras


def decision_function(model,data):
    a = np.argmax(model.predict(data), axis=1)
    print(a.shape)
    return a

class BD_detect:
    def __init__(self, args, model_path = "saved_models/cifar10_backdoor.h5"):
        # Ex. saved_models/cifar10_backdoor.h5
        self.args = args
        self.dict = 'adv_per'

        onnx_model = onnx.load(model_path)
        k_model = onnx_to_keras(onnx_model, ['input_1'])
        keras.models.save_model(k_model, 'kerasModel.h5', overwrite=True, include_optimizer=True)
        
        self.model = keras.models.load_model(model_path)

        if not os.path.exists(self.dict): os.mkdir(self.dict)

    def get_vec(self,original_label,target_label):
        if not os.path.exists(f"{self.dict}/data_{str(original_label)}_{str(target_label)}.npy"):
            f2 = h5py.File('kerasModel.h5', 'r')
            l = list(f2.keys())
            X = np.zeros((len(l), len(f2['0'])))
            for i,k in enumerate(l):
                X[i,:] = np.array(f2[k])
            np.save(f"{self.dict}/data_{str(original_label)}_{str(target_label)}.npy", X)

    def detect(self):
        for i in range(self.args.sp, self.args.ep):
            labels = list(range(self.args.num_labels))
            labels.remove(i)
            assert len(labels) == (self.args.num_labels-1), print(f"GGGGGGGG")

            for t in labels:
                print(f"original: {i} -> {t} \n")
                self.get_vec(original_label=i, target_label=t)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type=str,
        choices=['cifar10', 'cifar100','tiny'],
        default='cifar10'
    )
    parser.add_argument('--sp', type=int)
    parser.add_argument('--ep', type=int)
    parser.add_argument('--cuda', type=str)

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    bd = BD_detect(args=args, model_path=args.model)
    bd.detect()
