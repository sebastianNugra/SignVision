from signvision.camera.camera import Camera

camera = Camera(0)

try:
    camera.open()
    print("Camera opened successfully")
except RuntimeError as e:
    print(f"Camera failed: {e}")
