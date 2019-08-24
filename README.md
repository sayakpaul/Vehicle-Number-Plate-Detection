# Vehicle-Number-Plate-Detection
This project demonstrates the use of [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to automatically number plates (Indian) from vehicles.

Dataset used: https://www.kaggle.com/dataturks/vehicle-number-plate-detection

## File description:

- Data-Images.zip: Contains the images of the cars, number plates and annotations in `.txt` files (YOLO format)
- Data_prep_and_visualization.ipynb: A notebook demonstrating the process of preparing the dataset (`.csv` files) for creating TFRecords (otherwise TensorFlow Object Detection API won't work)
- Indian_Number_plates.json: Configuration file which contains image download paths and annotations
- frozen_inference_graph.pb: Inference graph in `.pb` format which can be used to run inference
- label_map.pbtxt: Contains the encodings of the dataset classes which,in this case, is 1: **license_plate**
- ssd_mobilenet_v1_pets.config: Training and evaluation pipeline configuration file as needed by TensorFlow Object Detection API
- test.record & train.record: `TFRecords` files of testing and training sets respectively
- test_labels.csv & train_labels.csv: `.csv` files as required by the `generate_tfrecord.py` script

## Demo inference (as collected from TensorBoard):

![](https://github.com/sayakpaul/Vehicle-Number-Plate-Detection/blob/master/demo_images/WhatsApp%20Image%202019-08-24%20at%2016.46.34.jpeg?raw=true)

![](https://github.com/sayakpaul/Vehicle-Number-Plate-Detection/blob/master/demo_images/WhatsApp%20Image%202019-08-24%20at%2016.49.13.jpeg?raw=true)
