import os
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
import keras.activations as activations
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import time
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import matplotlib.pyplot as plt


# 分類するクラス
classes = ['broken', 'correct']
nb_classes = len(classes)
img_width, img_height = 200, 200

# トレーニング用とバリデーション用の画像格納先
train_data_dir = '/home/toui/デスクトップ/研究_toui/add_data/DATA6_test_100'
validation_data_dir = '/home/toui/デスクトップ/ori/4/sub_300_1/validation'

#トレーニング用の画像とバリデーション用の画像枚数
nb_train_samples = 288
nb_validation_samples = 25

batch_size = 16

#結果を保存するフォルダ
result_dir = '/home/toui/デスクトップ/kekka'

#元祖 fine-tuning
def model_vgg():
    """ VGG16のモデルをFC層以外使用。FC層のみ作成して結合して用意する """

    # VGG16のロード。FC層は不要なので include_top=False
    input_tensor = Input(shape=(img_width, img_height, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # FC層の作成
    top_model = Sequential()
    # top_model.add(BatchNormalization())
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(512, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    # VGG16とFC層を結合してモデルを作成
    model = Model(input=vgg16.input, output=top_model(vgg16.output))

    return model

#最後の畳込みの前まで凍結 fine-tuning
def model_vgg2():
    """ VGG16のモデルをFC層以外使用。FC層のみ作成して結合して用意する """

    # VGG16のロード。FC層は不要なので include_top=False
    input_tensor = Input(shape=(img_width, img_height, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # FC層の作成
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(512, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    # VGG16とFC層を結合してモデルを作成
    model = Model(input=vgg16.input, output=top_model(vgg16.output))

    # 最後のconv層の直前までの層をfreeze
    for layer in model.layers[:15]:
        layer.trainable = False

    return model


#畳み込み全部凍結 転移学習
def model_vgg2_all():
    """ VGG16のモデルをFC層以外使用。FC層のみ作成して結合して用意する """

    # VGG16のロード。FC層は不要なので include_top=False
    input_tensor = Input(shape=(img_width, img_height, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # FC層の作成
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(512, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    # VGG16とFC層を結合してモデルを作成
    model = Model(input=vgg16.input, output=top_model(vgg16.output))

    # 最後のconv層までの層をfreeze
    for layer in model.layers[:19]:
        layer.trainable = False

    return model

#バッチ正規化用のモデル
def model_vgg3():
    """ VGG16のモデルをFC層以外使用。FC層のみ作成して結合して用意する """

    # VGG16のロード。FC層は不要なので include_top=False
    input_tensor = Input(shape=(img_width, img_height, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # x = MaxPooling2D((2,2), strides=(2,2))(vgg16.layers[-1].output)
    # x = Flatten()(vgg16.layers[-1].output)
    x = GlobalAveragePooling2D()(vgg16.layers[-1].output)
    x = Dense(256, activation='relu')(x)
    # top_model.add(Dropout(0.5))
    x = Dense(nb_classes, activation='softmax')(x)
    # あとでBatchNormを入れるため係数の固定はしない。初期値設定のみ転移学習とする

    return Model(vgg16.inputs, x)

#バッチ正規化用のモデル
def model_vgg3_kai():
    """ VGG16のモデルをFC層以外使用。FC層のみ作成して結合して用意する """

    # VGG16のロード。FC層は不要なので include_top=False
    input_tensor = Input(shape=(img_width, img_height, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # last = vgg16.get_layer("block4_conv3").output
    # last = vgg16.get_layer("block4_pool").output
    # last = vgg16.get_layer("block5_conv3").output
    last = vgg16.get_layer("block5_pool").output

    # x = Flatten()(vgg16.layers[-1].output)
    x = GlobalAveragePooling2D(name="GA")(last)
    # x = AveragePooling2D(pool_size=(2,2),strides=None,padding='valid')(last)
    # x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation='softmax')(x)
    # model = Model(input=vgg16.input, output=x)
    # あとでBatchNormを入れるため係数の固定はしない。初期値設定のみ転移学習とする

    # return model
    return Model(vgg16.inputs, x)

#最後の畳込みの前まで凍結 fine-tuning
def model_vgg2_ga():
    """ VGG16のモデルをFC層以外使用。FC層のみ作成して結合して用意する """

    # VGG16のロード。FC層は不要なので include_top=False
    input_tensor = Input(shape=(img_width, img_height, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    last = vgg16.get_layer("block4_conv3").output

    x = GlobalAveragePooling2D(name="GA")(last)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=vgg16.input, output=x)

    # 最後のconv層の直前までの層をfreeze
    for layer in model.layers[:15]:
        layer.trainable = False

    return model

def model_vgg2_2():
    """ VGG16のモデルをFC層以外使用。FC層のみ作成して結合して用意する """

    # VGG16のロード。FC層は不要なので include_top=False
    input_tensor = Input(shape=(img_width, img_height, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    last = vgg16.get_layer("block4_conv3").output

    x = GlobalAveragePooling2D(name="GA")(last)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=vgg16.input, output=x)

    # 最後のconv層の直前までの層をfreeze
    for layer in model.layers[:15]:
        layer.trainable = False

    return model

def model_vgg_small():#未完成
    """ VGG16のモデルをFC層以外使用。FC層のみ作成して結合して用意する """

    # VGG16のロード。FC層は不要なので include_top=False
    input_tensor = Input(shape=(img_width, img_height, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    last = vgg16.get_layer("block3_pool").output

    # FC層の作成
    x = GlobalAveragePooling2D()(last)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)

    # VGG16とFC層を結合してモデルを作成
    model = Model(input=vgg16.input, output=predictions)

    return model


#バッチ正規化
def create_batch_norm_model():
    model = model_vgg3_kai()

    for i, layer in enumerate(model.layers):
        if i==0:
            input = layer.input
            x = input
        else:
            if "conv" in layer.name:
                layer.activation = activations.linear
                x = layer(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)

            else:
                x = layer(x)

    bn_model = Model(input, x)
    return bn_model

def create_batch_norm_model_k():
    model = model_vgg3_kai()
    z_test = 0

    for i, layer in enumerate(model.layers):
        if i == 0:
            input = layer.input
            x = input
        else:
            if "block1_conv2" in layer.name:
                # z_test = z_test+1
                # if z_test%2 == 0:
                layer.activation = activations.linear
                x = layer(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
                # else:
                #     x = layer(x)
            elif "block2_conv2" in layer.name:
                layer.activation = activations.linear
                x = layer(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
            elif "block3_conv1" in layer.name:
                layer.activation = activations.linear
                x = layer(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
            elif "block3_conv3" in layer.name:
                layer.activation = activations.linear
                x = layer(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
            elif "block4_conv1" in layer.name:
                layer.activation = activations.linear
                x = layer(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
            elif "block4_conv3" in layer.name:
                layer.activation = activations.linear
                x = layer(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
            else:
                x = layer(x)

    bn_model = Model(input, x)
    return bn_model

def create_batch_norm_model_only():
    model = model_vgg3()

    for i, layer in enumerate(model.layers):
        if i == 0:
            input = layer.input
            x = input
        else:
            if "conv" in layer.name:
                layer.activation = activations.linear
                x = layer(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)

            else:
                x = layer(x)

    bn_model = Model(input, x)
    return bn_model


def image_generator():
    """ ディレクトリ内の画像を読み込んでトレーニングデータとバリデーションデータの作成 """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        vertical_flip=True,
        # zoom_range=0.2,
        # validation_split=0.2
    )

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        subset="training")

    validation_generator = validation_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        subset="validation")

    return (train_generator, validation_generator)

def plot_history(history):
    # 精度の履歴をプロット
    plt.plot(history.history['accuracy'], "o-", label="accuracy")
    # plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'], "o-", label="loss", )
    # plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()

#学習率変化
def step_decay_vgg(epoch):
    x = 0.001
    if epoch >= 20:
        x = 0.0005
    return x

#学習率変化
def step_decay_b(epoch):
    x = 0.001
    if epoch >= 15:
        x = 0.0005
    return x


if __name__ == "__main__":
    #batch_size = 8
    nb_epoch = 80

    #時間計測
    start = time.time()

    # モデル作成
    vgg_model = create_batch_norm_model()
    # vgg_model = model_vgg3_kai()
    # vgg_model = model_vgg()
    # vgg_model = model_vgg2()
    # vgg_model = model_vgg2_all()
    # vgg_model = model_vgg_small()

    #モデルを動かすための設定
    vgg_model.compile(loss='categorical_crossentropy',
                      # optimizer=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                      # optimizer=optimizers.Adam(lr=0.001, epsilon=1e-08, decay=0.0, amsgrad=False),
                      # optimizer=optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0),
                      optimizer=optimizers.SGD(lr=0.001, momentum=0.9),#lr=0.001=1e-3
                      # optimizer=optimizers.rmsprop(lr=5e-7, decay=5e-5),
                      metrics=['accuracy'])

    #アーキテクチャー表示
    vgg_model.summary()

    # 画像のジェネレータ生成
    train_generator, validation_generator = image_generator()

    lr_decay = LearningRateScheduler(step_decay_b)

    reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=3, verbose=1, mode='auto', min_lr=0.0003,)

    history = vgg_model.fit_generator(
        generator=train_generator,
        # steps_per_epoch=int(len(train_generator)/8),#これ使うならimagegeneratorの方のバッチサイズは消しておくこと
        epochs=nb_epoch,
        callbacks=[
                   ModelCheckpoint(filepath='/home/toui/デスクトップ/kekka/weight.h5', monitor="loss", verbose=1,
                                   save_best_only=True,save_weights_only=True, mode='min', period=1),
                   # lr_decay,
                   # reduce_lr,
                   # EarlyStopping(monitor="loss", patience=3),
        ],
        # validation_data=validation_generator,
        # validation_steps=16,
    )

    vgg_model.save(os.path.join(result_dir, 'model.h5'))

    # 重み保存
    # vgg_model.save_weights(os.path.join(result_dir, 'weight.h5'))

    #かかった時間表示
    process_time = (time.time() - start) / 60
    print(u'学習終了。かかった時間は', process_time, u'分です。')

    plot_history(history)