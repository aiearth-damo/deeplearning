import sys
import json

from rasterio.transform import rowcol
import rasterio
import cv2
import numpy as np

from .cv_ann_writer import CVAnnFileWriter, BlankCVAnnFileWriter 


def trans_geo_coordinate_to_image_coordinate(coordinates, transform):
    coordinates = np.array(coordinates)
    xs = coordinates[:, 0]
    ys = coordinates[:, 1]
    if transform is not None:
        [xs, ys] = rowcol(transform, xs=xs, ys=ys)
    if xs[0] != xs[-1] or ys[0] != ys[-1]:
        raise Exception("coordinates error")
    polygon = [xs, ys]
    contour = np.array(polygon, dtype=np.int32).transpose(1, 0)[:, np.newaxis, (1, 0)]
    #print(contour.shape)
    return contour

def trans_geometry_to_image_coordinate(geometry, transform):
    coordinates = geometry['coordinates']
    polygons = []
    if 'Polygon' == geometry['type']:
        for coordinate in coordinates:
            po = trans_geo_coordinate_to_image_coordinate(coordinate, transform)
            polygons.append(po)
    elif 'MultiPolygon' == geometry['type']:
        for polygon_index, polygon in enumerate(coordinates):
            for coordinate in polygon:
                po = trans_geo_coordinate_to_image_coordinate(coordinate, transform)
                polygons.append(po)
    return polygons


def trans_annfile_to_image_coordinate(annotations_json_path, transform_path, cv_ann_file_writer=BlankCVAnnFileWriter(), classes_filter=[]):
    assert isinstance(cv_ann_file_writer, CVAnnFileWriter)
    assert type(classes_filter) == list

    with open(annotations_json_path) as ann_f:
        records = json.load(ann_f)
        #print(records)

    with open(transform_path) as tran_f:
        transform_info = json.load(tran_f)
        tm = transform_info['transform']
        tm = rasterio.Affine(tm[0], tm[1], tm[2], tm[3], tm[4], tm[5])
        height = transform_info['height']
        width = transform_info['width']
        #print(tm)

    polygons = []
    if cv_ann_file_writer.canvas is None:
        canvas = np.zeros((height, width), dtype=np.uint8)
        cv_ann_file_writer.set_canvas(canvas)

    for feature in records['features']:
        objcode = feature['properties']['objcode']
        objcode_id = int(feature['properties']['objcode_id'])
        if classes_filter:
            if objcode not in classes_filter:
                continue
            objcode_id = classes_filter.index(objcode)
        # canvas default value is 0, skip
        if objcode_id == 0:
            continue
        geometry = feature['geometry']
        image_coor = trans_geometry_to_image_coordinate(geometry, tm)
        cv_ann_file_writer.draw_polygons(image_coor, objcode_id)
        polygons += image_coor
    cv_ann_file_writer.save()
    return polygons


def draw_to_image(src_image_path, polygons, save_image_path):
    image_data = cv2.imread(src_image_path)
    cv2.polylines(image_data, polygons, isClosed=True, color=(255, 125, 125), thickness=2)
    cv2.imwrite(save_image_path, image_data)


