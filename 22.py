import cv2

cap = cv2.VideoCapture(1)


# print('width: {0}, height: {1}'.format(cap.get(3),cap.get(4)))
# cap.set(3, 1280)
# cap.set(4, 720)
# print('width: {0}, height: {1}'.format(cap.get(3),cap.get(4)))
i = 0
# frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
while True:
    ret, frame = cap.read()

    if cv2.waitKey(1) == 27:
        filename = 'xxx/nothing0' + str(i + 1) + '.jpg'
        i = i + 1
        cv2.imwrite(filename, frame)
    # result_img = process_image(frame)
    cv2.imshow('Cam', frame)
    if cv2.waitKey(1) == 53:
        break
