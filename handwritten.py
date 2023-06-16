import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 

import cv2

np.random.seed(42)
tf.random.set_seed(42)

base_path = "data"
words_list = []
words = open(f"{base_path}","r").readlines()
for line in words :
  if line[0] == "#":
    continue
  if line.split(" ")[1] !="err":
    words_list.append(line)

len(words_list)
np.random.shuffle(words_list)
print(words_list[0:10])
print(len(words_list))

split_idx = int(0.9 * len(words_list))
train_samples = words_list[:split_idx]
test_sample = words_list[split_idx:]

val_plit_idx = int(0.5 * len(test_sample))
validation_sample = test_sample[:val_plit_idx]
test_sample = test_sample[val_plit_idx:]

assert len(words_list) == len(train_samples) + len(validation_sample) + len(test_sample)

print(f"total training samples :  {len(train_samples)}")
print(f"total validation samples :  {len(validation_sample)}")
print(f"total test samples :  {len(test_sample)}")

base_image_path ="/content/drive/MyDrive/words"
print(base_image_path)
def get_imgPath_and_lables(samples):
  paths = []
  corrected_sample = []
  for (i,file_line) in enumerate(samples):
    line_split = file_line.strip()
    line_split = line_split.split(" ")


    image_name = line_split[0]
    partI = image_name.split("-")[0]
    partII = image_name.split("-")[1]
   

    img_path = os.path.join(base_image_path , partI,partI+"-"+ partII ,image_name+".png")
    print(os.path.getsize(img_path))
    # if os.path.getsize(img_path):
    paths.append(img_path)
    corrected_sample.append(file_line.split("\n")[0])

  return paths , corrected_sample


train_img_path , train_lable = get_imgPath_and_lables(train_samples)
validation_img_path , validation_lable = get_imgPath_and_lables(validation_sample)
test_img_path , test_lable = get_imgPath_and_lables(test_sample)

train_lable_cleaned = []
charactres = set()
max_len = 0

for lable in train_lable:
  lable = lable.split(" ")[-1].strip()
  for char in lable:
    charactres.add(char)

  max_len = max(max_len,len(lable))
  train_lable_cleaned.append(lable)


print("min length",max_len)
print("vocab size",len(charactres))

train_lable_cleaned[:10]

def clean_lables(labels):
  cleaned_labels = []
  for label in labels:
    label = label.split(" ")[-1].strip()
    cleaned_labels.append(lable)

  return cleaned_labels


validation_lable_cleaned = clean_lables(validation_lable)
test_lable_cleaned = clean_lables(test_lable)

AUTOTUNE = tf.data.AUTOTUNE

char_to_num = StringLookup(vocabulary=list(charactres),mask_token=None)

num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(),mask_token = None , invert=True)

def distortion_free_resize(image,img_size):
  w,h = img_size
  image = tf.image.resize(image,size=(h,w),preserve_aspect_ratio=True)
  pad_hight = h - tf.shape(image)[0]
  pad_width = w - tf.shape(image)[1]


  if pad_hight % 2 != 0:
    height = pad_hight // 2
    pad_hight_top = height + 1
    pad_hight_bottom = height
  else:
    pad_hight_top = pad_hight_bottom = pad_hight // 2

  if pad_width % 2 != 0:
    width = pad_width // 2
    pad_width_left = width + 1
    pad_width_right = width
  else:
    pad_width_left = pad_width_right = pad_width // 2

  image = tf.pad(
      image,
      paddings=[
          [pad_hight_top , pad_hight_bottom],
          [ pad_width_left , pad_width_right],
          [0,0]
      ],
  )


  image = tf.transpose(image,perm=[1,0,2])
  image = tf.image.flip_left_right(image)
  return image

batch_size = 64
padding_token = 99
image_width = 128
image_height = 32

def preprocessing_image(image_path,img_size=(image_width,image_height)):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_png(image,1)
  image = distortion_free_resize(image , img_size)
  image = tf.cast(image,tf.float32) / 255.0
  return image

def vectorize_label(lable):
  lable = char_to_num(tf.strings.unicode_split(lable,input_encoding="UTF-8"))
  length = tf.shape (lable)[0]
  pad_amount = max_len - length
  lable = tf.pad(lable,paddings=[[0,pad_amount]],constant_values=padding_token)
  return lable

def porcess_image_lables(image_path,lable):
  image = preprocessing_image(image_path)
  lable = vectorize_label(lable)
  return{"image":image,"label":lable}


def prepare_dataset(image_paths,lables):
  dataset = tf.data.Dataset.from_tensor_slices((image_paths,lables)).map(porcess_image_lables,num_parallel_calls=AUTOTUNE)
  return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)

train_ds = prepare_dataset(train_img_path,train_lable_cleaned)
validation_ds = prepare_dataset(validation_sample,validation_lable_cleaned)
test_ds = prepare_dataset(test_img_path,test_lable_cleaned)
train_ds.take(1)

for data in train_ds.take(1):
  images,lables = data["image"],data["label"]
  print(images,lables)
  _,ax = plt.subplot(4,4,figsize=(15,8))
  for i in range(5):
    img = images[i]
    img = tf.image.flip_left_right(img)
    img = tf.transpose(img,perm=[1,0,2]) 
    img = (img * 255.0).numpy().clip(0,255).astype(np.uint8)
    img = img[:,:,0]
    lable = lables[i]
    indices = tf.gather(lable,tf.where(tf.math.not_equal(lable,padding_token)))
    lable = lable.numpy().decode("utf-8")
    ax[i//4,i % 4].imshow(img,cmap="gray")
    ax[i//4,i % 4].set_title(lable)
    ax[i//4,i%4].axis("off")

plt.show()

from tensorflow.python.ops.gen_array_ops import shape
class CTCLayer(keras.layers.Layer):
  def __init__(self,name=None):
    super().__init__(name=name)
    self.loss_fn = keras.backend.ctc_batch_cost
  def call(self,y_true,y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0],dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1],dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1],dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len,1),dtype="int64")
    label_length = label_length  * tf.ones(shape=(batch_len,1),dtype="int64") 

    loss = self.loss_fn(y_true,y_pred , input_length,label_length)
    self.add_loss(loss)


    return y_pred


def build_model():
  input_img = keras.Input(shape = (image_width,image_height,1),name="image")
  lables = keras.layers.Input(name="lable",shape=(None,))

  x = keras.layers.Conv2D(
      32,
      (3,3),
      activation = "relu",
      kernel_initializer = "he_normal",
      padding = "same",
      name="Conv1",

  )(input_img)

  x = keras.layers.MaxPooling2D((2,2),name = "pool1")(x)

  new_shape = ((image_width // 4),(image_height // 4) * 64)
  x = keras.layers.Reshape(new_shape )(x)
  x = keras.layers.Dense(64,activation = "relu" , name = "densel")(x)
  x = keras.layers.Dropout(0.2)(x)

  x = keras.layers.Bidirectional(
      keras.layers.LSTM(128,return_sequences=True,dropout=0.25)
  )(x)

  x = keras.layers.Bidirectional(
      keras.layers.LSTM(64,return_sequences=True,dropout=0.25)
  )(x)

  x = keras.layers.Dense(
      len(char_to_num.get_vocabulary())+2,activation="softmax",name="dense2"
  )(x)

  output = CTCLayer(name="ctc_loss")(lable,x)

  model = keras.models.Model(
      inputs=[input_img,lable],outputs = output,name="handwitten_recognizer"
  )

  opt = keras.optimizers.Adam()
  model.compile(optimizers-opt)
  return Model

model = build_model()
model.summary()

def calculate_edit_distance(labels,predictions):
  saprse_labels = tf.cast(tf.sparse.from_dense(labels,dtype = tf.int64))
  input_len = np.ones((predictions.shape[0])*predictions.shape[1])
  prediction_decoded = keras.backend.ctc_decode(
      predictions,input_length=input_len,greedy=True
  )[0][0][:,:max_len]
  
  saprse_predictions = tf.cast(
      tf.sparse.from_dense(prediction_decoded),dtype=tf.int64
      )
  
  edit_distance = tf.edit_distance(saprse_predictions,saprse_labels,normalize=False)
  return tf.reduce_mean(edit_distance)



class EditDistanceCallback(keras.callbacks.Callback):
  def __init__(self,pred_model):
    super().__init__()
    self.prediction_model = pred_model

  def on_epoch_end(self,epoch,logs=None):

    edit_distance = []

    for i in range(len(validation_image)):
      lable = validation_lable[i]
      predictions = self.prediction_model.predict(validation_image[i])
      edit_distance.append(calculate_edit_distance(lable,predictions).numpy())

    print(f"mean edit distance fro epoch {epoch + 1}:{np.mean(edit_distance):.4f}")

epoch = 10
model = build_model()
prediction_model = keras.models.Model(
    model.get_layer(name="image"),input,model.get_layer(name="dense2").output
)
edit_distance_callback = EditDistanceCallback(prediction_model)

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs = epoch,
    callback = [edit_distance_callback]
    
)

def decode_batch_prediction(pred):
  input_len = np.ones(pred.shape[0])*pred.shape[1]
  result = keras.backend.ctc_decode(pred,input_length=input_len,greedy=True)[0][0][:,:max_len]
  output_text = []
  for res in result:
    res = tf.gather(res,tf.where(tf.math.not_equal(res,-1)))
    res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
    output_text.append(res)

  return output_text


for batch in test_ds.take(1):
  batch_images = batch["image"]
  _,ax = plt.subplot(4,4,figsize=(15,8))

  preds = prediction_model.predict((batch_images))
  pred_text = decode_batch_prediction(preds)



for i in range(16):
  img = batch_images[i]
  img = tf.image.flip_left_right(img)
  img = tf.transpose(img,perm=[1,0,2]) 
  img = (img * 255.0).numpy().clip(0,255).astype(np.uint8)
  img = img[:,:,0]
  lable = lables[i]
  title = f"prediction: {pred_texts[i]}"
  ax[i//4,i % 4].imshow(img,cmap="gray")
  ax[i//4,i % 4].set_title(lable)
  ax[i//4,i%4].axis("off")

plt.show()

