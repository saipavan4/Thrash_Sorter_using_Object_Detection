# import yolov5


from ultralytics import YOLO
import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
import pyrealsense2 as rs
import numpy as np
import cv2
import imutils

# load model
# model = yolov5.load('keremberke/yolov5m-garbage')
model = YOLO("./best.pt")
  
# # set model parameters
# model.conf = 0.25  # NMS confidence threshold
# model.iou = 0.45  # NMS IoU threshold
# model.agnostic = False  # NMS class-agnostic
# model.multi_label = False  # NMS multiple labels per box
# model.max_det = 1000  # maximum number of detections per

# img = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
# path = "./img.jpg"
# orig = cv2.imread(path)
# classes=['biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic']
#classes=[ 'glass','cardboard',  'metal', 'paper', 'plastic','biodegradable',]
 
# perform inference
# results = model(orig, size=128)

# inference with test time augmentation

# for i in 
# cv2.imshow("Spud",results)

# colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
# img= np.expand_dims(orig, axis=0)
# print(boxes[0])
# print(scores)
# print(categories)


pipeline = rs.pipeline()
config = rs.config()    

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
def plot_boxes(labels,cord, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """
    # labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.3:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, classes(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame

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

# for i in boxes:
#     box=i.reshape([1, 1, 4])

#     tf.image.draw_bounding_boxes(
#     img, box, colors, name="anat" 
#     )
# cv2.imshow("img",np.squeeze(orig))
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
        
        # resized = cv2.resize(color_image, (256, 256 ))
        # resized = cv2.resize(color_image, (128, 128 ))
    
        # image = tf.keras.preprocessing.image.load_img(path, target_size=(256, 256))
        # image = tf.keras.preprocessing.image.img_to_array(image)
        # image = np.expand_dims(resized, axis=0)
        # print
        results = model.predict(color_image ,show=True)

        # parse results
        # predictions = results.pred[0]
        # boxes = predictions[:, :4] # x1, y1, x2, y2
        # scores = predictions[:, 4]
        # categories = predictions[:, 5]
        # cv2.imshow("Sup",results.render())
        # print(results.render()[0])
        # cv2.imshow("Sup",results.render()[0])
        print("*********************---------------------------+++++++++++++++++++++++++")
        # predictions = model.pred)ict(color_image)
        # print(predictions)
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
        # cv2.putText(color_image, classes[np.argmax(predictions)], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


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
        # img=plot_boxes(categories,boxes,color_image)
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', img)
        # cv2.waitKey(1)
        # results.show()


finally:

    # Stop streaming
    pipeline.stop()


# show detection bounding boxes on image
results.show()
# 