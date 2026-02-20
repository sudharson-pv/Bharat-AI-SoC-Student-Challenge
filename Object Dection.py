import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import time


def non_max_suppression(boxes, overlapThresh=0.4):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes).astype("float")
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:-1]]

        idxs = np.delete(
            idxs,
            np.concatenate(([len(idxs) - 1],
                            np.where(overlap > overlapThresh)[0]))
        )

    return boxes[pick].astype("int")


def run_hog_people_detection(img_dir):
    image_paths = sorted(
        glob.glob(img_dir + "/*.jpg") +
        glob.glob(img_dir + "/*.png") +
        glob.glob(img_dir + "/*.jpeg")
    )

    print("Total images:", len(image_paths))

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    latencies = []
    total_start = time.time()

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = 640.0 / w
        img = cv2.resize(img, (640, int(h * scale)))

        start = time.time()

        rects, weights = hog.detectMultiScale(
            img,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.03
        )

        boxes = []
        for (x, y, w, h), weight in zip(rects, weights):
            if weight > 0.6:
                boxes.append([x, y, x + w, y + h])

        final_boxes = non_max_suppression(boxes, overlapThresh=0.4)

        end = time.time()
        latency = end - start
        latencies.append(latency)

        # Draw detections
        for (x1, y1, x2, y2) in final_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "Person", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 4))
        plt.imshow(img_rgb)
        plt.title(f"{img_path.split('/')[-1]} | Detected: {len(final_boxes)}")
        plt.axis("off")
        plt.show()

    total_end = time.time()

    # Performance Summary
    latencies = np.array(latencies)

    if len(latencies) == 0:
        print("No images processed.")
        return

    avg_latency = latencies.mean()
    min_latency = latencies.min()
    max_latency = latencies.max()
    fps = 1.0 / avg_latency
    total_time = total_end - total_start

    print("\n===== CPU-ONLY PERFORMANCE =====")
    print(f"Total images processed : {len(latencies)}")
    print(f"Average latency        : {avg_latency*1000:.2f} ms")
    print(f"Min latency            : {min_latency*1000:.2f} ms")
    print(f"Max latency            : {max_latency*1000:.2f} ms")
    print(f"Throughput (FPS)       : {fps:.2f}")
    print(f"Total execution time   : {total_time:.2f} s")


# ---- CALL FUNCTION ----
run_hog_people_detection("/home/xilinx/jupyter_notebooks/data_sets")