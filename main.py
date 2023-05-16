import cv2 as cv
import numpy as np

cap = cv.VideoCapture(1)


coco_file="coco.names"
coco_classes=[]
net_config = "yolov3.cfg"
net_weights="yolov3.weights"
blob_size = 320
confidence_threshold = 0.25
nms_threshold = 0.3

with open(coco_file,"rt") as f:
    coco_classes = f.read().rstrip("\n").split("\n")

net = cv.dnn.readNetFromDarknet(net_config,net_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def findObjects(output, img):
    img_h, img_w, img_c = img.shape
    bboxes = []
    class_ids = []
    confidences = []

    for cell in output:
        for detect_vector in cell:
            scores = detect_vector[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                w,h = int(detect_vector[2] * img_w), int(detect_vector[3] * img_h)
                x,y = int((detect_vector[0] * img_w) - w/2), int((detect_vector[1] * img_h) - h/2)
                bboxes.append([x,y,w,h])
                class_ids.append(class_id)
                confidences.append(float(confidence))
    indices = cv.dnn.NMSBoxes(bboxes, confidences, confidence_threshold, nms_threshold)
    # print(indices)
    for i in indices:
        i = i[0]
        bbox = bboxes[i]
        x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
        cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv.putText(img, f'{coco_classes[class_ids[i]].upper()} {int(confidences[i] * 100)}%',
                    (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


while True:
    success, frame = cap.read()
    blob = cv.dnn.blobFromImage(frame,1/255,(blob_size,blob_size),(0,0,0), True,False)

   # for image in blob:
        #for k,b in enumerate(image):
           # cv.imshow(str(k),b)

    net.setInput(blob)
    out_name = net.getUnconnectedOutLayersNames()
    output = net.forward(out_name)
    #print(len(output))

    findObjects(output, frame)
    cv.imshow("ilia",frame)
    if cv.waitKey(1) == ord('q'):
        break