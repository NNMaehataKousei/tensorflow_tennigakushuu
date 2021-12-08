"""
転移学習とファインチューニング
今回は事前にトレーニングをしたネットワークから猫や犬の画像を分類する方法である。これを転移学習と呼ぶ

これを利用するメリットは大規模なモデルで大規模なデータセットでトレーニングされているネットワークを利用するため、最初から
大量のデータセットと大規模なモデルを用意しトレーニングをする必要がないということ。いわば学習済みネットワークを利用すれば
多様なニーズに対応し、かつ大規模レベルのデータとネットワーク構築をする必要しなくなるということ。

今回はトレーニング済みモデルをカスタマイズする2つの方法を試す。
１、特徴抽出：前のネットワークで学習した表現を使用して、新しいサンプルから意味のある特徴を抽出する。事前トレーニング済みモデルの上に
　　新規にトレーニングされる新しい分類器を追加するだけで、データセットで前に学習した特徴マップを再利用できようになっている。
モデル全体を再トレーニングする必要はない。ベースとなる畳み込みネットワークには、画像分類に一般的に有用な特徴がすでに含まれている。
ただし、事前トレーニング済みモデルの最後の分類部分は元の分類タスクに固有で、その後はモデルがトレーニングされたクラスのセットに固有である。
１、ファイチューニング：凍結された基本モデルの最上位レイヤーのいくつかを解凍し、新たに追加された分類器レイヤーと解凍した基本モデルの最後の
　　レイヤーの両方を併せてトレーニングする。これにより、基本モデルの高次の特徴表現をファインチューニングして、特定のタスクにより関連性を
　　持たせることができる。
"""
#学習プロセス#
#######################################################
# 1.データを調べ理解を深める
# 2.入力パイプラインを構築する。今回はKeras ImageDataGeneratorをしようする
# 3.モデルを作成する。
#     .事前トレーニング済みの基本モデル(事前トレーニング済みの重み)を読み込む
#     .分類レイヤーを上に重ねる
# 4.モデルをトレーニングする
# 5.モデルを評価する
#######################################################

from matplotlib.colors import from_levels_and_colors
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import keras

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.ops.control_flow_ops import from_control_flow_context_def
from tensorflow.python.ops.gen_array_ops import tensor_scatter_add_eager_fallback, tensor_strided_slice_update
from tensorflow.python.ops.gen_math_ops import less_eager_fallback
from tensorflow.python.training.tracking import base

#データの前処理
#データのダウンロード

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file("cats_and_dogs.zip",origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), "cats_and_dogs_filtered")

train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "validation")

#前処理
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
#print(train_dir)
#print(validation_dir)
train_dataset = image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

#トレーニングセットの最初の9枚の画像とラベルを表示します

class_names = train_dataset.class_names

"""
plt.figure(figsize = (10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
        #plt.show()
"""
"""
元のデータセットにはテストセットが含まれていないので、テストセットを作成する。
作成するにはtf.data.experimental.cardinalityを使用して検証セットで利用可能なデータのバッチ数を調べ、そのうちの20%をテストセットに移動する。
"""
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print("Number of Validation batches: %d" % tf.data.experimental.cardinality(validation_dataset))
print("Number of test batches: %d" % tf.data.experimental.cardinality(test_dataset))

#パフォーマンスのためにデータセットを構成する

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#データ増強を使用する
#ただし、model.fitを呼び出したときのみアクティブになる。モデルがmodel.evaulateやmodel.fitなどの推論モードで使用する場合はアクティブしない
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])
"""
for image, _ in train_dataset.take(1):
    plt.figure(figsize = (10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis("off")
        #plt.show()
"""

#ピクセル値をリスケールする
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
#別の方法でRescalingレイヤーを使用して、ピクセル値を[0,255]から[-1,1]にリスケールすることも可能である
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

#事前トレーニング済み畳み込みニューラルネットワークから基本モデルを作成する。
#MoblieNet V2モデルから基本モデルを生成する。

#ImageNetでトレーニングした重みで事前に読み込んだMobileV2モデルをインスタンス化する。引数はinclude_top=False
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")

#この特徴抽出器は160x160x3の画像を5x5x1280の特徴ブロックに変換する。これで画像のバッチ例がどうなるかを見てみましょう。
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

#特徴を抽出する。
#畳み込みベースを凍結させる。
base_model.trainable = False
base_model.summary()

#分類ヘッドを追加する
#特徴ブロッカら予測値を生成する。そのためにレイヤーを使って5x5空間の空間位置を平均化し、特徴画像ごとの1280要素ベクトルに変換させる。
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

#画像ごとに単一の予測値に変換する
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

#Keras Functional APIを使用して、データ増強、リスケール、base_model、特徴抽出レイヤーを凍結してモデルを構築させる。
inputs = tf.keras.Input(shape=(160,160,3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs,outputs)

#モデルをコンパイルする
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])

#モデル概要
model.summary()

len(model.trainable_variables)

#モデルをトレーニングする
initial_epochs = 10
loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}" .format(loss0))
print("initial accuracy: {:.2f}" .format(accuracy0))

history = model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset)

#学習結果を曲線で可視化
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
#plt.ylim([min(plt.ylim()),1])
plt.ylim(0.8,1.0)
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


#######################################################################
#ファインチューニング

#パフォーマンスをさらに向上させるために、追加した分類器のトレーニングと平行して、事前トレーニング済みモデルの最上位レイヤーの重みをトレーニング
#するというもの。ただし事前トレーニング済みモデルをトレーニング不可に設定し、最上位の分類気をトレーニングした後に行うようにする。

#モデルの最上位レイヤーを解凍する
base_model.trainable = True

print("Number of layers in the bass model: ", len(base_model.layers))

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

#モデルをコンパイルする
"""
かなり大規模なモデルをトレーニングしているため、事前トレーニング済みの重みを再適用する場合は、
この段階では低い学習率を使用することが重要です。そうしなければ、モデルがすぐに過適合を起こす可能性があります。
"""
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate/10), metrics=["accuracy"])
model.summary()

len(model.trainable_variables)

##tensorboardで記録する
log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


#モデルのトレーニングを続ける
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
history_fine = model.fit(
    train_dataset,epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_dataset,
    callbacks=tensorboard_callback
    )

acc += history_fine.history["accuracy"]
val_acc += history_fine.history["val_accuracy"]

loss += history_fine.history["loss"]
val_loss += history_fine.history["val_loss"]

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#評価と予測をする
loss, accuracy = model.evaluate(test_dataset)
print("Test accuracy :", accuracy)

###############################################################################
#Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")

