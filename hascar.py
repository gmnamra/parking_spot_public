import cv2 as cv
import sys
import numpy as np
from rectangle import intersection_over_union
from common import __this_spot__
import match
from fetchandextract import fetch_first_frame


# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
iouThreshold = 0.4
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

# Load names of classes
classesFile = "coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
carClassIndex = classes.index('car')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg";
modelWeights = "yolov3.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


## Get the names of the output layers
#
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

## Documentation for drawPred
#  draw the predicted bounding box

def drawPred(frame, classId, conf, iou, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)

    label = '%.2f:%.3f' % (conf, iou)

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_TRIPLEX, 0.6, 1)
    top = max(top, labelSize[1])

    #
    # cv.rectangle(frame, (left, top - round(1.25 * labelSize[1])),
    #             (left + round(1.25 * labelSize[0]), top + baseLine),
    #             (128,128,128), cv.FILLED)

    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)

## Documentation for postprocess
#
#  postprocess takes in outputs of a network run.
#  removed bounding boxes with low confidence using non-maxima suppression
#  returns assessment, image and roi
#


def postprocess(frame, outs, parkingSpot, show):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    parkingSpot_width = parkingSpot[3] - parkingSpot[1]
    parkingSpot_height = parkingSpot[2] - parkingSpot[0]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    ious = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            if not classId == carClassIndex:
                continue
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)

                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                bbox = [left, top, left + width, top + height]
                iou = intersection_over_union(bbox, parkingSpot)
                if iou < iouThreshold:
                    continue

                ious.append(iou)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    # Return the first pass or failure if none
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        if show:
            drawPred(frame, classIds[i], confidences[i], ious[i], left, top, left + width, top + height)
        return (True, (left, top, left + width, top + height))
    return (False, None)

## Documentation for process
#
#  process runs yolov3 and returns assessment and corresponding bounding box
#  it calls postprocess to appraise results of the network run
#
#

def process(filename, show):
    # Process inputs
    if show:
        winName = 'Parking Lot Monitoring'
        cv.namedWindow(winName, cv.WINDOW_NORMAL)

    frame = cv.imread(filename)

    msg = 'running yolo3 on ' + filename
    print(msg)

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    parkingSpot = __this_spot__
    result = postprocess(frame, outs, parkingSpot, show)

    # @todo: Put efficiency information.
    # The function getPerfProfile returns the overall time for inference(t) and
    # the timings for each of the layers(in layersTimes)
    if show:
        display = frame.copy()
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(display, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv.rectangle(display, (parkingSpot[0], parkingSpot[1]), (parkingSpot[2], parkingSpot[3]), (50, 255, 50), 3)
        if result[0]:
            box = result[1]
            cv.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 25, 255), 3)
        while cv.waitKey(1) < 0:
            cv.imshow(winName, display)
    return (result[0], frame, result[1])

## Documentation for compare
#
#  compares result of two process results and returns a validated / verified result
#  assumes two process results are sequential in time
#  return similarity assessment between the two time points
#

def compare(out_a, out_b, show=False):
    parkingSpot = __this_spot__
    yolo3_a = out_a[0]
    yolo3_b = out_b[0]
    imga = out_a[1]
    imgb = out_b[1]
    box_a = out_a[2]
    box_b = out_b[2]
    has_box_a = box_a != None
    has_box_b = box_b != None
    if not has_box_a and not has_box_a:
        box_a = parkingSpot
        box_b = parkingSpot
    elif has_box_a and not has_box_b:
        box_b = box_a
    elif has_box_b and not has_box_a:
        box_a = box_b
    else:
        left = min(box_a[0], box_b[0])
        top = min(box_a[1], box_b[1])
        right = max(box_a[2], box_b[2])
        bottom = max(box_a[3], box_b[3])
        box_a = (left, top, right, bottom)
        box_b = box_a

    # Crop at the spot
    roi_a = imga[box_a[0]:box_a[2], box_a[1]:box_a[3]]
    roi_b = imgb[box_b[0]:box_b[2], box_b[1]:box_b[3]]
    mi_a, h_a = match.MutualInformation(roi_a, roi_b)

    if show:
        match.mi_lum(roi_a, roi_b)

    roi_a_wider = imga[box_a[0] - 5:box_a[2] + 5, box_a[1] - 5: box_a[3] + 5]
    res = match.find_template(roi_a_wider, roi_b)

    return (res[1] > 0.5 and mi_a > 0.55) or mi_a > 1.0



if __name__ == '__main__':

    # compare(image_filename_a, image_filename_b)

    show = False
    files = []
    options = []
    exec_filename = sys.argv[0]

    for idx, arg in enumerate(sys.argv):
        if arg == exec_filename: continue
        if arg == 'show':
            options.append("show")
            continue
        ok = fetch_first_frame(arg)
        if ok[0]:
            files.append(ok[1])

    show = len(options) == 1
    if len(files) == 1:
        found = process(files[0], show)
        if not found:
            print('no car detected')
        else:
            print('car detected')
    elif len(files) == 2:
        out_a = process(files[0], show)
        out_b = process(files[1], show)

        is_same = compare(out_a, out_b, show)
        if not is_same:
            print('Different Cars')
        else:
            print('Same Cars')
