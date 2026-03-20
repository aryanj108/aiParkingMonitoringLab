from utils import compute_iou

def classify_spaces(space_boxes, car_boxes, iou_thresh=0.3):
    occupied, vacant = [], []

    for space in space_boxes:
        if any(compute_iou(space, car) > iou_thresh for car in car_boxes):
            occupied.append(space)
        else:
            vacant.append(space)

    return occupied, vacant
