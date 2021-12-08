"""
画像から猫または犬を分類する　tf.keras.Sequentialモデルを使用して画像分類器を構築
tf.keras.preprocessing.image.ImageDataGeneratorを使ってデータをロードする。
今回学ぶのは３つ
　１、tf.keras.preprocessing.image.ImageDataGeneratorクラスを使用してデータ入力パイプラインを構築、モデルで使用するディスク上のデータを効率的に処理する
　２、過学習を識別および防止する方法
　３、データ拡張およびドロップアウト：データパイプラインと画像分類モデルに組み込むコンピュータービジョンタスクの過学習と戦うための重要なテクニック
以上である
"""
"""
疑問点
rescaleは1./255を格納しているがこれは1を255で割った数を入れている
class_mode="binary"と書いてあるが部分がある。これはsigmoidを使って1部類だからである。
"""

from keras_preprocessing.image import directory_iterator
import tensorflow as tf#TensorFlowを導入

from tensorflow.keras.models import Sequential#今回はこのモデルを使うので導入する
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D#レイヤーを導入
from tensorflow.keras.preprocessing.image import ImageDataGenerator#ImageDataGeneratorを導入

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine.sequential import clear_previously_created_nodes
from tensorflow.python.keras.utils.layer_utils import validate_string_arg
from tensorflow.python.ops.array_ops import zeros_like_impl
from tensorflow.python.saved_model.save import _AugmentedGraphView

#データを読み込む#############################################################################
_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"

path_to_zip = tf.keras.utils.get_file("cats_and_dogs.zip",origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip),"cats_and_dogs_filtered")
#############################################################################################

#データの内容を取り出し、学習および検証をするための適切なファイルパスを変数に格納。
train_dir = os.path.join(PATH,"train")
validation_dir = os.path.join(PATH,"validation")

train_cats_dir = os.path.join(train_dir,"cats")#学習用の猫画像のディレクトリ
train_dogs_dir = os.path.join(train_dir,"dogs")#学習用の犬画像のディレクトリ
validation_cats_dir = os.path.join(validation_dir,"cats")#検証用の猫画像のディレクトリ
validation_dogs_dir = os.path.join(validation_dir,"dogs")#検証用の犬画像のディレクトリ

#データの理解　学習用と検証用のディレクトリの中にある猫と犬の画像の数を見る
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr#学習用の画像の総計
total_val = num_cats_val + num_dogs_val#検証用の画像の総計
"""
print("total training cat images:", num_cats_tr)
print("total training dog images:", num_dogs_tr)

print("total validation cat images:", num_cats_val)
print("total validation dog images:", num_dogs_val)

print("--")

print("Total training images:", total_train)
print("Total validation images", total_val)
"""
print("学習用の猫の画像の数は:", num_cats_tr)
print("学習用の犬の画像の数は:", num_dogs_tr)

print("検証用の猫の画像の数は:", num_cats_val)
print("検証用の犬の画像の数は:", num_dogs_val)

print("--------------------------------------------------")

print("学習用の画像のトータル:", total_train)
print("検証用の画像のトータル:", total_val)
#############################################################################
##データセットの前処理
###ネットワークの学習中に使用する変数を設定

batch_size = 128#バッチの大きさ
epochs = 15#学習する回数
IMG_HEIGHT = 150#画像の縦の長さ
IMG_WIDTH = 150#画像の横の長さ

#データの準備#################################################################
"""
モデルにデータを送る前に、画像を適切に前処理された宇裕小数点tensorにフォーマットする
１ディスクから画像を読み取る
２これらの画像をでコードし、RGB値にしたがって適切なグリッド形式に変換する
３それらを浮動小数点tensorに変換する
４ニューラルネットワークは小さな入力値を扱う方が適しているので、tensorを0-255の値から0-1の値にリスケーリングする。
ポイント：tf.keras.ImageDataGeneratorクラスですべて実現が可能！
"""
train_image_generator = ImageDataGenerator(rescale = 1./255) #学習データのジェネレータ
validation_image_generator = ImageDataGenerator(rescale = 1./255) #検証データのジェネレータ

#"flow_from_directory"メソッドはディスクから画像をロードし、画像を必要な大きさにリサイズする
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,directory=train_dir,target_size=(IMG_HEIGHT,IMG_WIDTH),class_mode="binary")
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,directory=validation_dir,target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode="binary")

#学習用の画像を可視化#######################################################
"""
学習用のジェネレータから画像バッチを抽出して可視化する。この例では32個の画像を取り出して、そのうちの5つをmatplotlibで描画する。
"""

sample_training_images, _= next(train_data_gen)#学習用に変換した画像をすべて格納させる。

def plotImages(images_arr):#画像表示するための関数
    fig,axes = plt.subplots(1,5,figsize = (20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

#画像表示
#plotImages(sample_training_images[:5])#5つの画像を取り出し可視化する

############################################################################

#モデルの構築################################################################
"""
モデルはmax pooling層を伴う3つの畳み込みブロックからなる。
さらにrelu活性化関数によるアクティベーションを伴う512ユニットの全結合層がある。
モデルはsigmoid()による2値分類に基づいてクラスに属する確率を出力する。
"""
"""
model = Sequential([
    Conv2D(16, 3, padding="same", activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding="same", activation="relu"),
    MaxPooling2D(),
    Conv2D(64, 3, padding="same", activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation="relu"),
    Dense(1, activation="sigmoid")
])


#モデルのコンパイル
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

#モデルの概要
model.summary()

"""

#モデルの学習#################################################################
"""
ImageDataGeneratorクラスのfit_generatorメソッドを使用して、ネットワークを学習する
"""
"""
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

#ネットワークを学習したあとの結果を可視化
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Traning Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()
"""

#過学習対策で今回はデータ拡張を使用し、モデルにドロップアウトを追加する

#データの拡張　その可視化
#水平反転の適用

image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,directory=train_dir,shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

#plotImages(augmented_images)

#ズームによるデータ拡張の適用

image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

#plotImages(augmented_images)

#すべてのデータ拡張を同時に利用する

image_gen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5
)

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode="binary")

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
#plotImages(augmented_images)

###########################################################################
#検証データジェネレータの建築
####今回は検証画像に対してリスケールのみを実施してバッチに変換する。
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size, directory=validation_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode="binary")

############################################################################
#ドロップアウトを適用した新しいネットワークの構築
"""
ドロップアウトとはネットワークにおいて重みを小さくする正則化の方式である。これによって重みの値の分布がより規則的になり、少ない学習データに
対する過学習を減らすことができる。
"""
model_new = Sequential([
    Conv2D(16, 3, padding="same", activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding="same", activation="relu"),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(64, 3, padding="same", activation="relu"),
    MaxPooling2D(),
    Dropout(0.2),#ここでドロップアウトを定義
    Flatten(),
    Dense(512, activation="relu"),
    Dense(1, activation="sigmoid")
])

#モデルのコンパイル
model_new.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model_new.summary()

#モデルの学習
history = model_new.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

#ネットワークを学習したあとの結果を可視化
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Traning Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()
