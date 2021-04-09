# Face-Mask-Detection-Counter-and-Tracker-using-DeepSORT-and-SSD-

## Introduction /n

This repository contains a moded version of PyTorch YOLOv5. It filters out every detection that is not a person. The/n detections of persons are then passed to a Deep Sort algorithm which tracks the persons. 

## Face Mask Detection Model /n


We used the structure of SSD. However,  in order to  make it run quickly in the browser, the backbone network is lite. The total model only has 1.01M parametes.

Input size of the model is 360x360, the backbone network only has 8 conv layers. The total model has only 24 layers with the  location and classification layers counted.

SSD anchor configurtion is show below:

| multibox layers | feature map size | anchor size | aspect ratio）|
| ---- | ---- | ---- | ---- |
|First|33x33|0.04,0.056|1,0.62,0.42|
Second ||17x17|0.08,0.11|1,0.62,0.42|
|Third|9x9|0.16,0.22|1,0.62,0.42|
|Forth |5x5|0.32,0.45|1,0.62,0.42|
|Fifth|3x3|0.64,0.72|1,0.62,0.42|

## How to run /n

python3 track.py --source video-path
