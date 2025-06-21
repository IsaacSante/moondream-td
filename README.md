# moondream-td
Realtime object detection with no data training using moondream vlm, then pipe the data into TouchDesigner.
Requires python and Touchdesigner.
A good laptop or desktop to get low latency. (Im using a 2020 M1 MacBook Pro)

# How to run
1. install moondream via transformers
2. install project dependencies ... pip install -r requirements.txt
3. define your objects of interest.
4. define your camera to run detection from.
3. run project via python main.py
4. run td project, make sure your server port is the same on the td webserver dat.
5. import the json into a table via the webserver dat callback.
6. have fun!

# Notes
Not sure this runs on windows but i imagine it could and even faster with cuda ?
