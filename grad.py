# coding:utf-8

import pandas as pd
import numpy as np
import cv2
import os
import PIL as pl
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model

result_dir = "/home/toui/デスクトップ/kekka/"

K.set_learning_phase(1) #set learning phase

def Grad_Cam(input_model, pic_array, layer_name):

    # 前処理
    pic = np.expand_dims(pic_array, axis=0)
    pic = pic.astype('float32')
    preprocessed_input = pic / 255.0

    # 予測クラスの算出
    predictions = input_model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = input_model.output[:, class_idx]

    #  勾配を取得
    conv_output = input_model.get_layer(layer_name).output   # layer_nameのレイヤーのアウトプット
    grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
    gradient_function = K.function([input_model.input], [conv_output, grads])  # input_model.inputを入力すると、conv_outputとgradsを出力する関数

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    # 重みを平均化して、レイヤーのアウトプットに乗じる
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # 画像化してヒートマップにして合成
    cam = cv2.resize(cam, (200, 200), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
    jetcam = (np.float32(jetcam) + pic / 2)   # もとの画像に合成

    return jetcam


model = load_model(os.path.join(result_dir + 'model.h5'))

model.load_weights(os.path.join(result_dir, "weight.h5"))

#正常画像
pic_array = img_to_array(load_img('/home/toui/デスクトップ/ori/add_data/DATA5_test/test_50/test/correct/05-4320.png', target_size=(200, 200)))
# pic_array = img_to_array(load_img('/home/toui/デスクトップ/ori/add_data/DATA6_test/test_50/test/correct/06-2401.png', target_size=(200, 200)))
# pic_array = img_to_array(load_img('/home/toui/デスクトップ/ori/add_data/DATA7_test/test_50/test/correct/07-2401.png', target_size=(200, 200)))
pic = pic_array.reshape((1,) + pic_array.shape)
array_to_img(pic_array)

#異常画像
pic_array1 = img_to_array(load_img('/home/toui/デスクトップ/ori/add_data/DATA5_test/test_50/test/broken/05-9003.png', target_size=(200, 200)))
# pic_array1 = img_to_array(load_img('/home/toui/デスクトップ/ori/add_data/DATA6_test/test_50/test/broken/06-8033.png', target_size=(200, 200)))
# pic_array1 = img_to_array(load_img('/home/toui/デスクトップ/ori/add_data/DATA7_test/test_50/test/broken/07-8033.png', target_size=(200, 200)))
pic1 = pic_array1.reshape((1,) + pic_array1.shape)
array_to_img(pic_array1)

picture = Grad_Cam(model, pic_array, 'block5_conv3')
picture = picture[0,:,:,]
y = array_to_img(picture)
y.show()

picture1 = Grad_Cam(model, pic_array1, 'block5_conv3')
picture1 = picture1[0,:,:,]
y1 = array_to_img(picture1)
y1.show()


