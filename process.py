import cv2
import torch
import easyocr
import numpy as np
import os
import matplotlib.pyplot as plt

laplas_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])


def pixels_cals(image):
    height, width = image.shape[:2]
    white = 0
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            pixel_val = image[y, x]
            if pixel_val != 0:
                white += 1
    return white


def drawRect(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.rectangle(image, (x, y), (x + w, y + h), (192, 89, 71), 2)
    cv2.drawContours(image, [box], 0, (120, 255, 90), 1)
    return (x, y, w, h), angle


def image_preproccessing(image):
    results = model(image)

    try:
        bbox = results.xyxy[0][:, :4].cpu().numpy()[0]
        xmin, ymin, xmax, ymax = bbox.astype(int)
        obj = image[ymin:ymax, xmin:xmax]

        im = cv2.filter2D(obj, -1, laplas_kernel)
        ret_, im_ = cv2.threshold(im, 90, 255, cv2.THRESH_BINARY)
        gray = cv2.cvtColor(im_, cv2.COLOR_BGR2HLS)
        gray[:, :, 1] = 0
        gray[:, :, 0] = 0
        gray[:, :, 2] = 255 - gray[:, :, 2]
        blurred = cv2.GaussianBlur(im, (3, 3), 0)
        canny = cv2.Canny(blurred, 20, 180, 1)

        return canny
    except Exception as e:
        return None


def is_it_square(canny):
    cnts, hc = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = -1
    max_area_cnt = -1
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        approx = cv2.approxPolyDP(c, 0.1 * cv2.arcLength(c, True), True)
        print(approx)
        if len(approx) == 4:
            if w * h < max_area:
                continue
            max_area = w * h
            max_area_cnt = c
    if max_area != -1:
        return max_area, max_area_cnt
    else:
        return False


def image_proccessing(image):
    results = model(image)

    bbox = results.xyxy[0][:, :4].cpu().numpy()[0]
    xmin, ymin, xmax, ymax = bbox.astype(int)
    obj = image[ymin:ymax, xmin:xmax]

    gray = cv2.cvtColor(obj, cv2.COLOR_BGR2HLS)
    gray[:, :, 1] = 0
    gray[:, :, 0] = 0
    gray[:, :, 2] = 255 - gray[:, :, 2]
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 20, 180, 1)
    debug_img = obj.copy()

    cv2.imshow('r', debug_img)
    cv2.waitKey(0)

    cnts, hc = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = -1
    max_area_cnt = -1

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        approx = cv2.approxPolyDP(c, 0.1 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            if abs(w / h) - 1 > 0.5 or w * h < max_area:
                continue
            max_area = w * h
            max_area_cnt = c

    (x, y, w, h), angle = drawRect(debug_img, max_area_cnt)

    rotM = cv2.getRotationMatrix2D((debug_img.shape[1] * 0.5, debug_img.shape[0] * 0.5), angle, 1)
    rotated = cv2.warpAffine(debug_img, rotM, (debug_img.shape[1], debug_img.shape[0]))
    cv2.imshow('r', rotated)
    cv2.waitKey(0)

    reader = easyocr.Reader(['en'], gpu=True)

    text = None

    for_text = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    for_text = cv2.adaptiveThreshold(for_text, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    reader_results = reader.readtext(for_text)
    for detection in reader_results:
        _, text, _ = detection
        print(text)

    cv2.putText(debug_img, f'angle: {round(angle, 0)}', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(192, 89, 71))
    cv2.imshow('r', debug_img)
    cv2.waitKey(0)

    side = 0

    vars_of_text = {}
    rotM = cv2.getRotationMatrix2D((best_img.shape[1] * 0.5, best_img.shape[0] * 0.5), angle, 1)
    rotated = cv2.warpAffine(best_img, rotM, (best_img.shape[1], best_img.shape[0]))

    while side != 4:
        rotM = cv2.getRotationMatrix2D((best_img.shape[1] * 0.5, best_img.shape[0] * 0.5), 90*side, 1)
        rotated = cv2.warpAffine(rotated, rotM, (best_img.shape[1], best_img.shape[0]))

        cv2.imshow('1', rotated)
        cv2.waitKey(0)
        reader_results = reader.readtext(rotated)
        for detection in reader_results:
            _, text, _ = detection
        print(text)

        vars_of_text[str(text)] = vars_of_text.setdefault(str(text), 0) + 1
        side += 1

    try:
        name_of_file = max(vars_of_text)
        cv2.imwrite(f'{name_of_file}.jpg', debug_img)
    except:
        cv2.imwrite('1.jpg', debug_img)



parent_dir = 'C:/Users/javas/PycharmProjects/TextOnMetal'

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5/runs/train/exp4/weights/last.pt')

cap = cv2.VideoCapture('video1.mp4')

ret, frame = cap.read()


best_frame_processed = image_preproccessing(frame)
best_img = frame
best_max_area = -1
best_max_area_cnt = None


best_frame_p = 0

k = 0

best_frame_angle = 0

while ret:
    ret, frame = cap.read()
    if ret:
        img = image_preproccessing(frame)
        if img is not None:
            new_ans = is_it_square(img)
            if new_ans:
                max_area, max_area_cnt = new_ans
                if max_area >= best_max_area:
                    p = pixels_cals(img)
                    print(k, max_area, p, best_frame_p)
                    if p >= best_frame_p+100:
                        print(k)
                        best_frame_p = p
                        best_max_area = max_area
                        best_max_area_cnt = max_area_cnt
                        best_img = frame
                        best_frame_processed = img
                        cv2.imwrite('best_frame.jpg', best_img)
            cv2.imwrite(os.path.join('C:/Users/javas/PycharmProjects/TextOnMetal/test/', f'file{k}.jpg'), img)
            k += 1
        else:
            break

image_proccessing(best_img)


