#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np

from utils import CvFpsCalc
from face_mesh.face_mesh import FaceMesh
from iris_landmark.iris_landmark import IrisLandmark


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--max_num_faces", type=int, default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.7)

    args = parser.parse_args()

    return args


def main():
    # 引数 #####################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    max_num_faces = args.max_num_faces
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    face_mesh = FaceMesh(
        max_num_faces,
        min_detection_confidence,
        min_tracking_confidence,
    )
    iris_detector = IrisLandmark()

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        # Face Mesh検出
        face_results = face_mesh(image)
        for face_result in face_results:
            # 目周辺のバウンディングボックス計算
            left_eye, right_eye = face_mesh.calc_around_eye_bbox(face_result)

            # 虹彩検出
            left_iris, right_iris = detect_iris(image, iris_detector, left_eye,
                                                right_eye)

            # 虹彩の外接円を計算
            left_center, left_radius = calc_min_enc_losingCircle(left_iris)
            right_center, right_radius = calc_min_enc_losingCircle(right_iris)

            # デバッグ描画
            debug_image = draw_debug_image(
                debug_image,
                left_iris,
                right_iris,
                left_center,
                left_radius,
                right_center,
                right_radius,
            )

        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('Iris(tflite) Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()

    return


def detect_iris(image, iris_detector, left_eye, right_eye):
    image_width, image_height = image.shape[1], image.shape[0]
    input_shape = iris_detector.get_input_shape()

    # 左目
    # 目の周辺の画像を切り抜き
    left_eye_x1 = max(left_eye[0], 0)
    left_eye_y1 = max(left_eye[1], 0)
    left_eye_x2 = min(left_eye[2], image_width)
    left_eye_y2 = min(left_eye[3], image_height)
    left_eye_image = copy.deepcopy(image[left_eye_y1:left_eye_y2,
                                         left_eye_x1:left_eye_x2])
    # 虹彩検出
    eye_contour, iris = iris_detector(left_eye_image)
    # 座標を相対座標から絶対座標に変換
    left_iris = calc_iris_point(left_eye, eye_contour, iris, input_shape)

    # 右目
    # 目の周辺の画像を切り抜き
    right_eye_x1 = max(right_eye[0], 0)
    right_eye_y1 = max(right_eye[1], 0)
    right_eye_x2 = min(right_eye[2], image_width)
    right_eye_y2 = min(right_eye[3], image_height)
    right_eye_image = copy.deepcopy(image[right_eye_y1:right_eye_y2,
                                          right_eye_x1:right_eye_x2])
    # 虹彩検出
    eye_contour, iris = iris_detector(right_eye_image)
    # 座標を相対座標から絶対座標に変換
    right_iris = calc_iris_point(right_eye, eye_contour, iris, input_shape)

    return left_iris, right_iris


def calc_iris_point(eye_bbox, eye_contour, iris, input_shape):
    iris_list = []
    for index in range(5):
        point_x = int(iris[index * 3] *
                      ((eye_bbox[2] - eye_bbox[0]) / input_shape[0]))
        point_y = int(iris[index * 3 + 1] *
                      ((eye_bbox[3] - eye_bbox[1]) / input_shape[1]))
        point_x += eye_bbox[0]
        point_y += eye_bbox[1]

        iris_list.append((point_x, point_y))

    return iris_list


def calc_min_enc_losingCircle(landmark_list):
    center, radius = cv.minEnclosingCircle(np.array(landmark_list))
    center = (int(center[0]), int(center[1]))
    radius = int(radius)

    return center, radius


def draw_debug_image(
    debug_image,
    left_iris,
    right_iris,
    left_center,
    left_radius,
    right_center,
    right_radius,
):
    # 虹彩：外接円
    cv.circle(debug_image, left_center, left_radius, (0, 255, 0), 2)
    cv.circle(debug_image, right_center, right_radius, (0, 255, 0), 2)

    # 虹彩：ランドマーク
    for point in left_iris:
        cv.circle(debug_image, (point[0], point[1]), 1, (0, 0, 255), 2)
    for point in right_iris:
        cv.circle(debug_image, (point[0], point[1]), 1, (0, 0, 255), 2)

    # 虹彩：半径
    cv.putText(debug_image, 'r:' + str(left_radius) + 'px',
               (left_center[0] + int(left_radius * 1.5),
                left_center[1] + int(left_radius * 0.5)),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    cv.putText(debug_image, 'r:' + str(right_radius) + 'px',
               (right_center[0] + int(right_radius * 1.5),
                right_center[1] + int(right_radius * 0.5)),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    return debug_image


if __name__ == '__main__':
    main()
