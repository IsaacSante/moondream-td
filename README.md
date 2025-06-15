# moondream-td
realtime obj detection with no training using moondream and touchdesigner.

runs with td, moondream station and td.

windows not supported yet.

1. install moondream station for mac os.
2. install project dependencies ... pip install -r requirements.txt
3. run project.
4. run td project.
5. activate process from td.



# What it do ?
Basically we take the camera from td.
save it to this repo every second
then run moondream obj detection on that frame
finally we pipe the obj bounding boxes back into td
Addtional: we can use torins mediapipe to check intersection with objects.

# todo
1. move objs to config


# reminders
# native cam 0
# obs cam 1