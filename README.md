# Vehicle-Number-Plate-Detection
This project demonstrates the use of [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to automatically number plates (Indian) from vehicles.

Dataset used: https://www.kaggle.com/dataturks/vehicle-number-plate-detection

## File description:

- Data-Images.zip: Contains the images of the cars, number plates and annotations in `.txt` files (YOLO format)
- Data_prep_and_visualization.ipynb: A notebook demonstrating the process of preparing the dataset (`.csv` files) for creating TFRecords (otherwise TensorFlow Object Detection API won't work)
- Indian_Number_plates.json: Configuration file which contains image download paths and annotations
- exported_graph: Contains the inference graph in `.pb` and `.tflite` formats which can be used to run inference on both CPU platforms and on-device platforms
- label_map.pbtxt: Contains the encodings of the dataset classes which,in this case, is 1: **license_plate**
- ssd_mobilenet_v1_pets.config: Training and evaluation pipeline configuration file as needed by TensorFlow Object Detection API
- test.record & train.record: `TFRecords` files of testing and training sets respectively
- test_labels.csv & train_labels.csv: `.csv` files as required by the `generate_tfrecord.py` script

## Demo inference (as collected from TensorBoard):

![](https://github.com/sayakpaul/Vehicle-Number-Plate-Detection/blob/master/demo_images/WhatsApp%20Image%202019-08-24%20at%2016.46.34.jpeg?raw=true)

![](https://github.com/sayakpaul/Vehicle-Number-Plate-Detection/blob/master/demo_images/WhatsApp%20Image%202019-08-24%20at%2016.49.13.jpeg?raw=true)


To kick-start the model training process and to export the trained model's inference graph (using the `export_tflite_ssd_graph.py` script), I followed:
- [TensorFlow Object Detection API's official documentation](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [Training and serving a realtime mobile object detector in 30 minutes with Cloud TPUs](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193)

I used **SSD_MobileNet_V1** architecture which was pretrained on the COCO dataset. 

To convert the frozen inference graph, I ran the following command:
```
tflite_convert \
    --output_file=detect.tflite \
    --graph_def_file=frozen_inference_graph.pb \
    --input_shapes=1,300,300,3 \
    --input_arrays=normalized_input_image_tensor \
    --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
    --inference_type=QUANTIZED_UINT8 \
    --mean_values=128 \
    --std_dev_values=128 \
    --change_concat_input_ranges=false \
    --allow_custom_ops
```
    
**Note**: To be able convert an inference graph to its `.tflite` variant you need to enable _quantization aware training_ and you can specify that in the `.config` file itself. 
