# object_detection_uno

This notebook is primarly oriented to be used on Google Colab and to use models from modelzoo.

# Usage
To use the notebook, you will first need to install the necessary libraries, which are listed below and train the model, which you can do in the notebook itself.

You will then need to adjust some paths and the notebook will be ready to use.


# Requirements
Python 3.7.13

libcudnn8=8.1.0.77-1

cuda11.2

object detection api (https://github.com/tensorflow/models).

# Examples and Metrics
![alt-text-1](example1.jpg "title-1") ![alt-text-2](example2.jpg "title-2")







| Metric | Area & Maximum detections | Result |
| :---         |     :---       |        :---   |
| AP IoU=0.50    | all   maxDets = 100  | 0.788    |
| AP IoU=0.75      | all  maxDets = 100     | 0.984      |
| AP IoU=0.50:0.95    | small  maxDets = 100     |  0.760      |


| Metric | Area & Maximum detections | Result |
| :---         |     :---       |        :---   |
| AR IoU=0.50:0.95      | all  maxDets = 100     |  0.827     |
| AR IoU=0.50:0.95      | small  maxDets = 100     | 0.803      |
| AR IoU=0.50:0.95      | medium  maxDets = 100     | 0.878      |

