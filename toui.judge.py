import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model


classes = ['correct', 'broken']
nb_classes = len(classes)
img_width, img_height = 120, 120

#重みとかモデルが保存されているファイル
result_dir = '/home/toui/デスクトップ/kekka/'

# このディレクトリにテストしたい画像を格納しておく
test_data_dir = '/home/toui/デスクトップ/研究_toui/add_data/DATA6_test/test'


if __name__ == "__main__":

    #学習したモデルの構造をロード
    vgg_model = load_model(os.path.join(result_dir + 'model.h5'))

    #学習したモデルの重みをロード
    vgg_model.load_weights(os.path.join(result_dir, 'weight.h5'))

    # テスト用画像取得
    test_imagelist = os.listdir(test_data_dir)

    # 計算用変数,配列宣言
    TP, FP, FN, TN, counter, brocheck, brorate = 0, 0, 0, 0, 0, 0, 0
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0
    filenum = len(test_imagelist)
    test_array = [0 for i in range(filenum)]
    pred_array = [0 for i in range(filenum)]
    file_nameTN = []
    file_nameFP = []
    file_nameFN = []
    file_nameTP1 = []
    file_nameFP1 = []
    file_nameFN1 = []
    i=0

    for test_image in test_imagelist:
        i = i+1

        filename = os.path.join(test_data_dir, test_image)

        img = image.load_img(filename, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # 学習時に正規化してるので、ここでも正規化
        x = x / 255
        pred = vgg_model.predict(x)[0]

        # 予測確率が高いトップを出力
        # 今回は最も似ているクラスのみ出力したいので1にしているが、上位n個を表示させることも可能。
        top = 2
        top_indices = pred.argsort()[-top:][::-1]
        result = [(classes[i], pred[i]) for i in top_indices]
        print(top_indices)

        # 正解不正解の判別
        # ファイル名を参照して4300|4400の境界で判別
        # filenum = filename[0:3]
        ft = filename.find("4300")
        ft_1 = filename.find("4400")
        ft_2 = filename.find("4500")
        ft_3 = filename.find("4600")
        ft_4 = filename.find("4700")
        # print(ft)

        if ft == -1 and ft_1 == -1 and ft_2 == -1 and ft_3 == -1 and ft_4 == -1:
            FT = 0
        else:
            FT = 1

        # F値算出.どれかに分類
        # [予測クラス,正解クラス]の配列を作成
        c_array = [top_indices[0], FT]

        if c_array == [0, 0]:
            TP = TP + 1
        elif c_array == [0, 1]:
            FP = FP + 1
            file_nameFP.append(test_image)
        elif c_array == [1, 0]:
            FN = FN + 1
            file_nameFN.append(test_image)
        elif c_array == [1, 1]:
            TN = TN + 1
            file_nameTN.append(test_image)


        #異常をTP、正常をFNにするとき
        # if ft == -1 and ft_1 == -1 and ft_2 == -1 and ft_3 == -1 and ft_4 == -1:
        #     FT = 1
        # else:
        #     FT = 0

        # F値算出.どれかに分類
        # [予測クラス,正解クラス]の配列を作成
        c_array = [top_indices[0], FT]

        if c_array == [1, 1]:
            TP1 = TP1 + 1
            file_nameTP1.append(test_image)
        elif c_array == [1, 0]:
            FP1 = FP1 + 1
            file_nameFP1.append(test_image)
        elif c_array == [0, 1]:
            FN1 = FN1 + 1
            file_nameFN1.append(test_image)
        elif c_array == [0, 0]:
            TN1 = TN1 + 1


        # if int(filenum) == 4700:
        #     brocheck = brocheck + 1
        #
        #     if c_array == [1, 1]:
        #         brorate = brorate + 1

        # ターミナルで見やすいよう予測と正解があってるかprint
        if FT == top_indices[0]:
            print('正解')
        else:
            print('間違い！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！')

        print('file name is', test_image)
        print(result)
        print('=======================================')

        # ROC曲線作成用配列に格納
        # test_array[counter] = top_indices[0]
        # pred_array[counter] = pred[top_indices[0]]
        test_array[counter] = FT
        pred_array[counter] = pred[top_indices[0]]
        counter = counter + 1

    # 正解率
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    print("全画像枚数は{name}枚".format(name=i))
    print('正解率＝', accuracy)

    print('------------------------------------------------------')
    print('正常機器(壊れてないデバイス)のうち，正常機器と正しくラベルをつけられた割合（TPR）')#Recall,真陽性率
    TPR = TP / (TP + FN)
    print('TPR=', TPR)
    print("TP=", TP)
    print("FN=", FN)
    print(file_nameFN)

    print(' ')
    print('異常機器(壊れているデバイス)のうち，正常機器と”間違えて”ラベルをつけられた割合（FPR）')#真陰性率
    FPR = FP / (FP + TN)
    print('FPR=', FPR)
    print("FP=", FP)
    print(file_nameFP)
    print("TN=", TN)
    print(file_nameTN)

    print('------------------------------------------------------')
    accuracy1 = (TP1 + TN1) / (TP1 + FP1 + TN1 + FN1)
    print('正解率1＝', accuracy1)
    print('------------------------------------------------------')
    print('異常機器(壊れてるデバイス)のうち，異常機器と正しくラベルをつけられた割合（TPR）')  # Recall,真陽性率
    TPR1 = TP1 / (TP1 + FN1)
    print('TPR1=', TPR1)
    print("TP1=", TP1)
    print(file_nameTP1)
    print("FN1=", FN1)
    print(file_nameFN1)
    print(' ')

    print('正常機器(壊れていないデバイス)のうち，異常機器と”間違えて”ラベルをつけられた割合（FPR）')  # 真陰性率
    FPR1 = FP1 / (FP1 + TN1)
    print('FPR1=', FPR1)
    print("FP1=", FP1)
    print(file_nameFP1)
    print("TN1=", TN1)
    print('------------------------------------------------------')

    # if brocheck != 0:
    #     print('4700の枚数=', brocheck)
    #     print('異常で異常だった数=', brorate)
    #     print('4700を異常機器にラベリングできた割合')
    #     LabelRate = brorate / brocheck
    #     print('4700RR=', LabelRate)
"""""
    # FPR, TPR(, しきい値) を算出
    fpr, tpr, thresholds = metrics.roc_curve(test_array, pred_array)

    # ついでにAUC
    auc = metrics.auc(tpr, fpr)

    # ROC曲線をプロット
    plt.plot(tpr, fpr, label='ROC curve (area = %.2f)' % auc)
    plt.legend()
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show()
    # plt.close('ROC curve')
"""""