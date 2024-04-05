from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
import pyrealsense2 as rs
import numpy as np
import cv2
import imutils

classes=['r','o']
# classes=  ['Aluminium', 'Carton', 'Glass', 'Organic Waste', 'Other Plastics', 'Paper and Cardboard', 'Plastic', 'Textiles', 'Wood']
# print("Here")
model = keras.models.load_model("./waste.h5")
# print("There")
class GradCAM:
  def __init__(self, model, classIdx, layerName=None):
    self.model = model
    self.classIdx = classIdx
    self.layerName = layerName
    if self.layerName is None:
      self.layerName = self.find_target_layer()
   
  def find_target_layer(self):
    for layer in reversed(self.model.layers):
      if len(layer.output_shape) == 4:
        return layer.name
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

  def compute_heatmap(self, image, eps=1e-8):
    gradModel = Model(
			inputs=[self.model.inputs],
			outputs=[self.model.get_layer(self.layerName).output,
				self.model.output])
    with tf.GradientTape() as tape:
      inputs = tf.cast(image, tf.float32)
      (convOutputs, predictions) = gradModel(inputs)
      loss = predictions[:, self.classIdx]
    grads = tape.gradient(loss, convOutputs)
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
    (w, h) = (image.shape[2], image.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")
    return heatmap

  def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS):
    heatmap = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    return (heatmap, output)

path = "./organic.jpeg"
orig = cv2.imread(path)


# pipeline = rs.pipeline()
# config = rs.config()

# Get device product line for setting a supporting resolution
# pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# pipeline_profile = config.resolve(pipeline_wrapper)
# device = pipeline_profile.get_device()
# device_product_line = str(device.get_info(rs.camera_info.product_line))


# found_rgb = False
# for s in device.sensors:
#     if s.get_info(rs.camera_info.name) == 'RGB Camera':
#         found_rgb = True
#         break
# if not found_rgb:
#     print("The demo requires Depth camera with Color sensor")
#     exit(0)

# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# if device_product_line == 'L500':
#     config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
# else:
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # Start streaming
# pipeline.start(config)

print("Done Initializing")
# #image comes here   
# import time
# st=time.time()
# try:
#     while True:

  # Wait for a coherent pair of frames: depth and color
#  ? frames = pipeline.wait_for_frames()
#  ? color_frame = frames.get_color_frame()
#  ? if  not color_frame:
#  ?     continue
# ?
# # Convert images to numpy arrays
# color_image = np.asanyarray(color_frame.get_data())

# # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
# color_colormap_dim = color_image.shape

resized = cv2.resize(orig, (64,64))

# image = tf.keras.preprocessing.image.load_img(path, target_size=(256, 256))
# image = tf.keras.preprocessing.image.img_to_array(image)
image = np.expand_dims(resized, axis=0)

predictions = model.predict(image)
# cam = GradCAM(model, np.argmax(predictions[0]), "expanded_conv_6/expand")
# heatmap = cv2.resize(cam.compute_heatmap(image), (orig.shape[1], orig.shape[0]))

#heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
# (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

#cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
# print(st-time.time())
cv2.putText(resized, classes[np.argmax(predictions)], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
print(np.argmax(predictions))
# output = np.vstack([orig, heatmap, output])
# output = imutils.resize(output, height=700)
# cv2.imshow("Output",image)
# cv2.waitKey(0)
# cv2.imwrite("output.jpg",output)


# If depth and color resolutions are different, resize color image to match depth image for display
# if depth_colormap_dim != color_colormap_dim:
#     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
#     images = np.hstack((resized_color_image, depth_colormap))
# else:
#     images = np.hstack((color_image, depth_colormap))

# Show images
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
cv2.imshow('RealSense', resized)
cv2.waitKey(0)

# finally:

#     # Stop streaming
#     pipeline.stop()


# resized = cv2.resize(orig, (256, 256))

# image = tf.keras.preprocessing.image.load_img(path, target_size=(256, 256))
# image = tf.keras.preprocessing.image.img_to_array(image)
# image = np.expand_dims(image, axis=0)

# predictions = model.predict(image)
# cam = GradCAM(model, np.argmax(predictions[0]), "expanded_conv_6/expand")
# heatmap = cv2.resize(cam.compute_heatmap(image), (orig.shape[1], orig.shape[0]))

# #heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
# (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

# #cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
# print(st-time.time())
# cv2.putText(output, classes[np.argmax(predictions)], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# output = np.vstack([orig, heatmap, output])
# output = imutils.resize(output, height=700)
# cv2.imshow("Output",output)
# cv2.waitKey(0)
# cv2.imwrite("output.jpg",output)
