#!/usr/bin/env bash

#!/usr/bin/env bash
wget --no-check-certificate https://pjreddie.com/media/files/yolov3.weights
wget --no-check-certificate https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true -O ./yolov3.cfg
wget --no-check-certificate https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true -O ./coco.names

