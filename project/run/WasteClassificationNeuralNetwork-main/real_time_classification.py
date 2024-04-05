from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
import pyrealsense2 as rs
import numpy as np
import cv2
import imutils



fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
classes=  ['Aluminium', 'Carton', 'Glass', 'Organic Waste', 'Other Plastics', 'Paper and Cardboard', 'Plastic', 'Textiles', 'Wood']
# classes=["Recycle", "Organic"]

# model = keras.models.load_model("./tf_model_2.h5")
# model = keras.models.load_model("./WasteClassificationModel.h5")
# model = keras.models.load_model("./Final_Model.h5")
model = keras.models.load_model("./waste.h5")



pipeline = rs.pipeline()
config = rs.config()    

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))


found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

print("Done Initializing")
# #image comes here   
# import time
# st=time.time() 
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if  not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        color_colormap_dim = color_image.shape
        
        # resized = color_image
        # out.write(color_image) 
        resized = cv2.resize(color_image, (64, 64 ))
    
        # image = tf.keras.preprocessing.image.load_img(path, target_size=(256, 256))
        # image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(resized, axis=0)
        # print
        predictions = model.predict(image)
        print(predictions)
        # predictions=np.argmax(predictions)
        # print(predictions)
      
        # res=""
        # if predictions <= 0.45:
        #     res="Recycle"
        # else:
        #     res="Organic"
    
        # cam = GradCAM(model, np.argmax(predictions[0]), "expanded_conv_6/expand")
        # heatmap = cv2.resize(cam.compute_heatmap(image), (orig.shape[1], orig.shape[0]))

        #heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        # (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

        #cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
        # print(st-time.time())


        # cv2.putText(color_image, res, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(color_image, classes[np.argmax(predictions)], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


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
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()


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
