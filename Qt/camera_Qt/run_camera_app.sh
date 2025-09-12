#!/bin/bash

# 라즈베리파이 카메라 앱 실행 스크립트
# Qt 멀티미디어와 라즈베리파이 카메라의 호환성을 위한 환경 설정

echo "라즈베리파이 카메라 앱을 시작합니다..."

# 카메라 상태 확인
echo "카메라 상태 확인 중..."
if command -v vcgencmd &> /dev/null; then
    echo "vcgencmd 결과:"
    vcgencmd get_camera
fi

# 비디오 디바이스 확인
echo -e "\n사용 가능한 비디오 디바이스:"
ls -la /dev/video* 2>/dev/null || echo "비디오 디바이스를 찾을 수 없습니다."

# V4L2 디바이스 정보
if command -v v4l2-ctl &> /dev/null; then
    echo -e "\nV4L2 디바이스 목록:"
    v4l2-ctl --list-devices 2>/dev/null
fi

# libcamera 지원 확인 (최신 라즈베리파이 OS)
if command -v libcamera-hello &> /dev/null; then
    echo -e "\nlibcamera 카메라 목록:"
    timeout 3s libcamera-hello --list-cameras 2>/dev/null || echo "libcamera 타임아웃"
fi

# Qt 환경 변수 설정 - 라즈베리파이에 최적화
# 디스플레이 감지 및 적절한 플랫폼 선택
if [ -n "$DISPLAY" ] && xset q >/dev/null 2>&1; then
    echo "X11 디스플레이 사용: $DISPLAY"
    export QT_QPA_PLATFORM=xcb
elif [ -n "$WAYLAND_DISPLAY" ]; then
    echo "Wayland 디스플레이 사용: $WAYLAND_DISPLAY"
    export QT_QPA_PLATFORM=wayland
else
    echo "GUI 디스플레이를 찾을 수 없습니다. EGL 프레임버퍼 사용"
    export QT_QPA_PLATFORM=eglfs
    # eglfs에서 마우스 지원
    export QT_QPA_EGLFS_CURSOR=1
fi

export QT_LOGGING_RULES="qt.multimedia.debug=true"

# GStreamer 백엔드 우선 사용 (라즈베리파이에서 더 안정적)
export QT_MULTIMEDIA_PREFERRED_PLUGINS=gstreamer

# GStreamer 디버그 및 라즈베리파이 최적화
export GST_DEBUG=2
export GST_PLUGIN_PATH=/usr/lib/arm-linux-gnueabihf/gstreamer-1.0

# libcamera와 Qt 호환성을 위한 V4L2 설정
# 다양한 포맷과 해상도를 시도하는 GStreamer 파이프라인
export QT_GSTREAMER_CAMERABIN_VIDEOSRC="v4l2src device=/dev/video0 ! videoconvert ! videoscale"
export QT_GSTREAMER_CAMERABIN_AUDIOSRC="alsasrc"

# GStreamer 파이프라인 디버그
export GST_DEBUG_DUMP_DOT_DIR=/tmp
export GST_DEBUG="*:3,v4l2:5,gstqtvideosink:5"

echo -e "\n환경 변수 설정 완료"
echo "QT_QPA_PLATFORM: $QT_QPA_PLATFORM"
echo "QT_MULTIMEDIA_PREFERRED_PLUGINS: $QT_MULTIMEDIA_PREFERRED_PLUGINS"

# 앱 실행
echo -e "\nQt 카메라 앱을 실행합니다..."
if [ -f "./camera_Qt" ]; then
    ./camera_Qt
else
    echo "오류: camera_Qt 실행 파일을 찾을 수 없습니다."
    echo "먼저 다음 명령어로 빌드하세요:"
    echo "  qmake camera_Qt.pro"
    echo "  make"
fi