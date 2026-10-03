

import backbone
import tensorflow as tf
import cv2
import numpy as np


class model:
    def __init__(self):
       
        self.detection_graph, self.category_index = backbone.set_model(
            "ssd_mobilenet_v1_coco_2018_01_28", "mscoco_label_map.pbtxt"
        )
        

    def get_category_index(self):
        return self.category_index

    def detect_pedestrians(self, frame):
      
        image = np.asarray(frame)
       
        input_tensor = tf.convert_to_tensor(image)
        
        input_tensor = input_tensor[tf.newaxis,...]

   
        output_dict = self.detection_graph(input_tensor)

        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections


        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        (boxes, scores, classes, num) = (output_dict['detection_boxes'], output_dict['detection_scores'],
            output_dict['detection_classes'], output_dict['num_detections'])

       
        pedestrian_score_threshold = 0.35
        pedestrian_boxes = []
        total_pedestrians = 0
        for i in range(int(num)):
            if classes[i] in self.category_index.keys():
                class_name = self.category_index[classes[i]]["name"]
                
                if class_name == "person" and scores[i] > pedestrian_score_threshold:
                    total_pedestrians += 1
                    score_pedestrian = scores[i]
                    pedestrian_boxes.append(boxes[i])

        return pedestrian_boxes, total_pedestrians
