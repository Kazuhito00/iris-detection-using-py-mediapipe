# iris-detection-using-py-mediapipe
MediaPipeのIris(虹彩検出)をPythonで動作させるデモです。<br>
MediaPipeのFace Meshで顔のランドマークを検出した後に、「[iris_landmark.tflite](https://github.com/google/mediapipe/blob/master/mediapipe/modules/iris_landmark/iris_landmark.tflite)」を用いて虹彩の検出をしています。<br>

![suwkm-avmbx](https://user-images.githubusercontent.com/37477845/101514487-a59d8500-39c0-11eb-8346-d3c9ab917ea6.gif)

# Requirement 
* mediapipe 0.8.1 or later
* OpenCV 3.4.2 or later

mediapipeはpipでインストールできます。
```bash
pip install mediapipe
```

# Demo
デモの実行方法は以下です。
```bash
python demo.py
```
デモ実行時には、以下のオプションが指定可能です。

* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --max_num_faces<br>
顔の検出最大数<br>
デフォルト：1
* --min_detection_confidence<br>
検出信頼値の閾値<br>
デフォルト：0.7
* --min_tracking_confidence<br>
トラッキング信頼値の閾値<br>
デフォルト：0.7

# ToDo
- [ ] 焦点距離から深度を推定するオプションを追加

# Reference
* [MediaPipe](https://github.com/google/mediapipe)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
mediapipe-python-sample is under [Apache-2.0 License](LICENSE).

また、女性の画像は[フリー素材ぱくたそ](https://www.pakutaso.com)様の写真を利用しています。
