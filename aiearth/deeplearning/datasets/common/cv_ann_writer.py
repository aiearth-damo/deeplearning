import abc
import json
import cv2
import numpy as np


class CVAnnFileWriter(metaclass=abc.ABCMeta):
    def set_canvas(self, canvas):
        self.canvas = canvas

    def set_save_path(self, save_path):
        self.save_path = save_path

    @abc.abstractmethod
    def draw_polygons(self, image_polygons, objcode_id):
        pass

    @abc.abstractmethod
    def save(self):
        pass

class BlankCVAnnFileWriter(CVAnnFileWriter):
    def draw_polygons(self, image_polygons, objcode_id):
        pass

    def save(self):
        pass


class UniversalCVAnnFileWriter(CVAnnFileWriter):
    def __init__(self, canvas=None, save_path=None):
        self.save_path = save_path
        self.canvas = canvas
        pass

    def set_canvas(self, canvas):
        self.canvas = canvas

    def set_save_path(self, save_path):
        self.save_path = save_path

    def draw_polygons(self, image_polygons, objcode_id):
        cv2.drawContours(self.canvas, image_polygons, -1, (objcode_id,), cv2.FILLED)

    def save(self):
        cv2.imwrite(self.save_path, self.canvas)


class UniversalCVAnnFileColorWriter(UniversalCVAnnFileWriter):
    def __init__(self, canvas=None, save_path=None, colormap=None):
        self.save_path = save_path
        self.canvas = canvas
        self.colormap = colormap

    def save(self):
        lab = self.colormap[self.canvas]
        cv2.imwrite(self.save_path, np.concatenate([lab], axis=1))


class ChangeDetAnnFileWriter(UniversalCVAnnFileWriter):
    pass


class LandcoverAnnFileWriter(UniversalCVAnnFileWriter):
    pass


class TargetExtractionAnnFileWriter(UniversalCVAnnFileWriter):
    pass


class DetectionAnnFileWriter(CVAnnFileWriter):
    def __init__(self, canvas=None, save_path=None):
        super().__init__(canvas, save_path)
        self.content = ""
        self.save_path = save_path

    def points_2_minRect(self, points):
        #print(points)
        #points = np.array(points)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        print("box", box)
        #cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        return box

    def draw_polygons(self, image_polygons, objcode_id):
        #print(image_polygons, objcode_id)
        for polygon in image_polygons:
            line = []
            for point in self.points_2_minRect(polygon):
                line += [ str(i) for i in point]
            line.append(str(objcode_id))
            self.content += (" ".join(line) + '\n')

    def save(self):
        with open(self.save_path, 'w') as f:
            print(self.content)
            f.write(self.content)


class DetectionAnnFileColorWriter(DetectionAnnFileWriter):
    def __init__(self, canvas=None, save_path=None):
        self.canvas = canvas
        self.save_path = save_path
        self.content = ""

    def save(self):
        for li in self.content.split('\n'):
            li = li.strip().split(' ')
            if len(li) != 9:
                continue
            x1, y1, x2, y2, x3, y3, x4, y4 = [int(float(i)) for i in li[:8]]
            cnt = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape(-1, 2)
            rect = cv2.minAreaRect(cnt) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            box = cv2.boxPoints(rect) # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
            box = np.int0(box)
            cv2.drawContours(self.canvas, [box], -1, (0, 0, 255), 2)
        cv2.imwrite(self.save_path, self.canvas)


class InstanceSegAnnFileWrite(CVAnnFileWriter):
    def __init__(self, canvas=None, save_path=None):
        super().__init__(canvas, save_path)
        self.ann_dict = []

    def cal_bbox(self, polygon):
        xs, ys = np.squeeze(polygon.transpose()).tolist()
        min_xs = min(xs)
        max_xs = max(xs)
        min_ys = min(ys)
        max_ys = max(ys)
        bx = min_xs
        by = min_ys
        bh = max_xs - min_xs
        bw = max_ys - min_ys
        return [bx, by, bh, bw]

    def draw_polygons(self, image_polygons, objcode_id):
        ann_temp = {'segmentation': [[510.45, 423.01, ...]], 'area': 702.1057499999998, 'iscrowd': 0, 'image_id': 289343, 'bbox': [473.07, 395.93, 38.65, 28.67], 'category_id': 18, 'id': 1768}
        for polygon in image_polygons:
            ann_temp['segmentation'] = polygon.reshape(-1).tolist()
            bx, by, bh, bw = self.cal_bbox(polygon)
            ann_temp['bbox'] = [bx, by, bh, bw]
            ann_temp['area'] = bh * bw
            ann_temp['category_id'] = objcode_id
            self.ann_dict.append(ann_temp)

    def save(self):
        print(json.dumps(self.ann_dict, indent=4))



if __name__ == '__init__':
    import sys
    from .geo_transfer import trans_annfile_to_image_coordinate
    ann_json_file = sys.argv[1]
    transform_file = sys.argv[2]
    save_path = sys.argv[3]
    changedet_ann_writer = ChangeDetAnnFileWriter(save_path=save_path)
    trans_annfile_to_image_coordinate(ann_json_file, transform_file, changedet_ann_writer)