import numpy as np
import tensorflow as tf
import os
from inception import inception_model


IMAGE_SIZE = 299
def preprocess_image(image_buffer):
  """Preprocess JPEG encoded bytes to 3D float Tensor."""

  # Decode the string as an RGB JPEG.
  # Note that the resulting image contains an unknown height and width
  # that is set dynamically by decode_jpeg. In other words, the height
  # and width of image is unknown at compile-time.
  image = tf.image.decode_jpeg(image_buffer, channels=3)
  # After this point, all image pixels reside in [0,1)
  # until the very end, when they're rescaled to (-1, 1).  The various
  # adjust_* ops all require this range for dtype float.
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  # Crop the central region of the image with an area containing 87.5% of
  # the original image.
  #image = tf.image.central_crop(image, central_fraction=0.875)
  # Resize the image to the original height and width.
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image,
                                   [IMAGE_SIZE, IMAGE_SIZE],
                                   align_corners=False)
  image = tf.squeeze(image, [0])
  # Finally, rescale to [-1,1] instead of [0, 1)
  image = tf.sub(image, 0.5)
  image = tf.mul(image, 2.0)
  return image


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
  

NUM_CLASSES = 7
NUM_TOP_CLASSES = 8

MODEL_CHECKPOINT_PATH = 'dk-finetune/model.ckpt-45000'
#with tf.Graph().as_default():    
jpegs = tf.placeholder(tf.string)
images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)
# Run inference.
logits, _ = inception_model.inference(images, NUM_CLASSES + 1)
# Transform output to topK result.
values, indices = tf.nn.top_k(logits, NUM_TOP_CLASSES)
# Restore the moving average version of the learned variables for eval.
variable_averages = tf.train.ExponentialMovingAverage(
    inception_model.MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

sess = tf.Session()
# Restore variables from training checkpoints.  
saver.restore(sess, MODEL_CHECKPOINT_PATH)
# Assuming model_checkpoint_path looks something like:
#   /my-favorite-path/imagenet_train/model.ckpt-0,
# extract global_step from it.
global_step = MODEL_CHECKPOINT_PATH.split('/')[-1].split('-')[-1]
print('Successfully loaded model from %s at step=%s.' %
      (MODEL_CHECKPOINT_PATH, global_step))

folder = 'test_release'
test_csv = 'test_release.csv'

if __name__ == '__main__':
  print 'Running...'

  ## list of test files
  with open(test_csv) as csvTest:
    list_test = csvTest.readlines()[1:]
  classes = ['unused background', 'astronaut', 'aurora', 'black', 'city', 'none', 'stars', 'unknown']
  # result file
  with open("result_test.csv", "w") as csvResult:
    csvResult.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'
    .format('image_url', 'city', 'none', 'unknown', 'aurora', 'astronaut', 'stars', 'black'))
    # process one image a time
    for line in list_test:    
      url = line.strip()
      # get filename
      f = os.path.basename(url)     
      file_local = os.path.join(folder, f) 
      # send request
      with open(file_local, 'rb') as fp:      
	data = fp.read()
	scores = []
	result = sess.run([values, indices], {jpegs: [data]})
	logts = result[0][0]
	idxs = result[1][0]
	# convert log to softmax prob
	probs = softmax(logts)
	# write to result file
	result_dict = {'city':0, 'none':0, 'unknown':0, 'aurora':0, 'astronaut':0, 'stars':0, 'black':0, 
		      'unused background':0}
	for c,p in zip(classes, probs):
	  result_dict[c] = p
	csvResult.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'
	.format(url, result_dict['city'], result_dict['none'], result_dict['unknown'], result_dict['aurora'], 
		result_dict['astronaut'], result_dict['stars'], result_dict['black']))    
      #break     

  print 'Done!'