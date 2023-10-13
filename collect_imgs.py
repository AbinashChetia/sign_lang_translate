import os
import cv2
import time

DATA_dir = './data'
if not os.path.exists(DATA_dir):
    os.mkdir(DATA_dir)

n_class = 3
dataset_size = 100

cap = cv2.VideoCapture(0)
for i in range(n_class):
    if not os.path.exists(os.path.join(DATA_dir, str(i))):
        os.mkdir(os.path.join(DATA_dir, str(i)))

    time.sleep(5)
    print(f'Collecting class {i} images...')
    
    for j in range(dataset_size):
        ret, frame = cap.read()
        cv2.imwrite(os.path.join(os.path.join(DATA_dir, str(i)), str(j) + '.jpg'), frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    print('class {} done'.format(i))