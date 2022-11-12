#loding required models 

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename

#for loading, spliting and decoding the data 
  
# load numpy 
import numpy as np
# load tensorflow 
import tensorflow as tf  

# make load decoder dictionary 
decoder=dict()
# read labels 
with open('labels.txt',mode='r') as label_file:# read file 
  for line in label_file.readlines():# read line in file
    if line:# if line is not empty 
      clean_line_list=line.replace('\n','').split(' ')# make remove '\n' and split data by space
      decoder.update({int(clean_line_list[0]):clean_line_list[1]})# add class index as key and class name as value
# see labels 
print('Labels:',decoder)

# model path 
model_path=r'keras_model.h5'
# load model - pretrained model
face_detector=tf.keras.models.load_model(model_path)
# see model summary 
face_detector.summary()



#giving numy array of the image according to the tf model 

#from IPython.display import Image
# load pil module 
from PIL import Image,ImageOps

# define image size 
image_traget_size=(224,224)

try:
  filename=take_photo()# get image path 
  print('Saved to {}'.format(filename))
  # Show the image which was just taken.
  #display(Image(filename))
  # load image as numpy array 
  #image=tf.keras.preprocessing.image.load_img(filename,color_mode='rgb',target_size=None,interpolation='nearest',
  #                                            keep_aspect_ratio=False)# read image 
  #input_array=tf.keras.preprocessing.image.img_to_array(image)# image to numpy array
  # or 
  image=Image.open(filename).convert('RGB')# load image as color image 
  image_processed=ImageOps.fit(image,image_traget_size,Image.LANCZOS)
  image_array=np.asarray(image_processed)# make numpy array 
  input_array=(image_array.astype(np.float32)/127.0)-1# pre process image 
  input_batch=tf.expand_dims(input_array,axis=0)# Convert single image to a batch.
  label=np.argmax(face_detector(input_batch))# get labels from probabilites 
  print('I think you are:',decoder[label])
except Exception as error:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print('Error! fail to capture image - ',str(error))
  
