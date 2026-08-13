from signvision.camera import Camera


def test_camera_initial_state() -> None:
    camera = Camera(device_index=0)

    assert camera._device_index == 0
    assert camera._capture is None
