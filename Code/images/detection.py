import cv2
import numpy as np
import os


def object_detection(img, rects, model, fruits_db):
    id_to_labels = {i: v for i, v in
                    enumerate(os.listdir(fruits_db.train_dir))}

    images = [cv2.resize(img[y:y + h, x:x + w], (45, 45)) for x, y, w, h in rects]
    results = model.predict(np.array(images))

    objects, types = np.where(results > 0.9)
    for i, object in enumerate(objects):
        x, y, w, h = rects[object]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))
        cv2.putText(img, id_to_labels[types[i]], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0))
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    return results
