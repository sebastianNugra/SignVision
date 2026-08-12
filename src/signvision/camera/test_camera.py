from signvision.camera import Camera

camera = Camera(device_index=0)

try:
    camera.open()
    print("Camera opened successfully")

    frame = camera.read()

    print(type(frame))
    print(frame.shape)
    print(frame.dtype)

finally:
    camera.close()
