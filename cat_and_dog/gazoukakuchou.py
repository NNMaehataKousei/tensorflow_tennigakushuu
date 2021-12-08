from ctypes import resize
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import stateless_random_contrast
import tensorflow_datasets as tfds

from tensorflow.keras import layers


(train_ds, val_ds, test_ds), metadata = tfds.load(
    "tf_flowers",
    split=["train[:80%]","train[80%:90%]","train[90%:]"],
    with_info=True,
    as_supervised=True,
)

num_classes = metadata.features["label"].num_classes
print(num_classes)

#花のデータセットには5つクラスがあります
get_label_name = metadata.features["label"].int2str

#データセットから画像を取得して、それを使用してデータ拡張を示す
image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))

####################################################################
#Keras前処理レイヤーを使用する
#サイズ変更と再スケーリング

IMG_SIZE = 180

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./255)
])

result = resize_and_rescale(image)
_ = plt.imshow(result)

print("Min and max pixel values:", result.numpy().min(), result.numpy().max())

#データ拡張
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),






])
# add the image to a batch
image = tf.expand_dims(image, 0)

plt.figure(figsize=(10, 10))
for i in range(0):
    augmented_image = data_augmentation(image)
    ax = plt.subplot(3, 3, i+1)
    plt.imshow(augmented_image[0])
    plt.axis("off")

#Keras前処理レイヤーを使用する２つのオプション
model = tf.keras.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
])

#オプション２　前処理レイヤーをデータセットに適用する
aug_ds = train_ds.map(
    lambda x, y: (resize_and_rescale(x, training=True),y))

batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, shuffle=False, augment=False):
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)
    
    ds = ds.batch(batch_size)

    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)

#モデルをトレーニングする
model = tf.keras.Sequential([
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes)
])

#モデルをコンパイルする
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

#いくつかのエポックのトレーニング
epochs=5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

loss, acc = model.evaluate(test_ds)
print("Accuracy", acc)


#カスタムデータの拡張
def random_invert_img(x, p=0.5):
    if tf.random.uniform([]) < p:
        x = (255-x)
    else:
        x
    return x

def random_invert(factor=0.5):
    return layers.Lambda(lambda x: random_invert_img(x, factor))

random_invert = random_invert()

plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_image = random_invert(image)
    ax = plt.subplot(3, 3, i+1)
    plt.imshow(augmented_image[0].numpy().astype("uint8"))
    plt.axis("off")

#カスタム層を実装する
class RandomInvert(layers.Layer):
    def __init__(self, factor=0.5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
    
    def call(self, x):
        return random_invert_img(x)

_ = plt.imshow(RandomInvert()(image)[0])

#tf.imageを使用する
(train_ds, val_ds, test_ds), metadata = tfds.load("tf_flowers", split=["train[:80%]", "train[80%:90%]","train[90%:]"],with_info=True, as_supervised=True,)

image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))

#次の関数を使用して、元の画像と拡張画像を並べて視覚化して比較してみる
def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title("Original image")
    plt.imshow(original)

    plt.subplot(1,2,2)
    plt.title("Augmented image")
    plt.imshow(augmented)

#データ拡張
#画像を反転する
flipped = tf.image.flip_left_right(image)
visualize(image, flipped)

#画像をグレースケール
grayscaled = tf.image.rgb_to_grayscale(image)
visualize(image, tf.squeeze(grayscaled))
_ = plt.colorbar()

#画像を飽和させる
saturated = tf.image.adjust_saturation(image, 3)
visualize(image, saturated)

#画像の明るさを変更する
bright = tf.image.adjust_brightness(image, 0.4)
visualize(image, bright)

#画像の中央でトリミング
cropped = tf.image.central_crop(image, central_fraction=0.5)
visualize(image,cropped)

#画像を回転させる
rotated = tf.image.rot90(image)
visualize(image, rotated)

#ランダム変換
######ランダムな画像操作の2セットがありますtf.image.random*とtf.image.stateless_random* 。使用tf.image.random*彼らはTF 1.xのから古いのRNGを使用して操作することを強くお勧めし代わりに、このチュートリアルで紹介したランダム画像操作を使用してください。詳細については、を参照してくださいランダム番号生成。

#画像の明るさをランダムに変更する
for i in range(3):
    seed = (i, 0) # tuple of size (2,)
    stateless_random_brightness = tf.image.stateless_random_brightness(image, max_delta=0.95, seed=seed)
    visualize(image, stateless_random_brightness)

#画像のコントラストをランダムに変更する
for i in range(3):
    seed = (i, 0)
    stateless_random_contrast = tf.image.stateless_random_contrast(image, lower=0.1, upper=0.9, seed=seed)
    visualize(image, stateless_random_contrast)

#画像をランダムに切り抜く
for i in range(3):
    seed = (i, 0)
    stateless_random_crop = tf.image.stateless_random_crop(image, size=[210, 300, 3], seed=seed)
    visualize(image,stateless_random_crop)

#データセットに拡張を適用する
(train_datasets, val_ds, test_ds), metadata = tfds.load("tf_flowers", split=["train[:80%]", "train[80%:90%]","train[90%:]"], with_info=True, as_supervised=True,)

def resize_and_rescale(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (image/255.0)
    return image, label

#ランダム変形を画像に適用することができる機能augmentを定義
def augment(image_label, seed):
    image, label = image_label
    image, label = resize_and_rescale(image, label)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    # Make a new seed
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    # Random crop back to the original size
    image = tf.image.stateless_random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
    #Random brightness
    image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=new_seed)
    image = tf.clip_by_value(image, 0, 1)
    return image, label

#作ったオブジェクトとデータセットを一緒にトレーニングするためのセット
counter = tf.data.experimental.Counter()
train_ds = tf.data.Dataset.zip((train_datasets, (counter, counter)))

#augmentをトレーニングデータセット
train_ds = (train_ds.shuffle(1000).map(augment, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE))
val_ds = (val_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE))
test_ds = (test_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE))

#tf.random.Generatorを使用する
#Create a generator
rng = tf.random.Generator.from_seed(123, alg="philox")
def f(x, y):
    seed = rng.make_seeds(2)[0]
    image, label = augment((x,y), seed)
    return image, label

#トレーニングセットにresize_and_rescale機能への検証とテスト
train_ds = (train_datasets.shuffle(1000).map(f, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE))
val_ds = (val_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE))
test_ds = (test_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE))
print("finished")
