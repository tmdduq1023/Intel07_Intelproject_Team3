"""
if 이 코드가 qt
    qt에서 웹소켓 구현 가능.
    with 문제
        qt에서 사진 찍는 법을 모르겠음.
"""

from picamera2 import Picamera2

picam2 = Picamera2()
picam2.start()
picam2.capture_file("test.jpg")
picam2.stop()
