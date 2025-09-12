#!/bin/bash

echo "========================================="
echo "라즈베리파이 카메라 정보 수집 스크립트"
echo "========================================="
echo ""

echo "1. 라즈베리파이 모델 및 OS 정보"
echo "-----------------------------------"
cat /proc/cpuinfo | grep "Model" || echo "Model info not found"
cat /etc/os-release | grep PRETTY_NAME || echo "OS info not found"
echo ""

echo "2. 카메라 모듈 활성화 상태"
echo "-----------------------------------"
if command -v vcgencmd &> /dev/null; then
    vcgencmd get_camera
else
    echo "vcgencmd not available"
fi
echo ""

echo "3. rpicam 카메라 목록 (최신 방식)"
echo "-----------------------------------"
if command -v rpicam-hello &> /dev/null; then
    echo "rpicam-hello 사용 가능"
    timeout 5s rpicam-hello --list-cameras 2>&1 || echo "rpicam-hello 실행 실패 또는 타임아웃"
else
    echo "rpicam-hello 명령어를 찾을 수 없음"
fi
echo ""

echo "4. libcamera 카메라 목록 (이전 방식)"
echo "-----------------------------------"
if command -v libcamera-hello &> /dev/null; then
    echo "libcamera-hello 사용 가능"
    timeout 5s libcamera-hello --list-cameras 2>&1 || echo "libcamera-hello 실행 실패 또는 타임아웃"
else
    echo "libcamera-hello 명령어를 찾을 수 없음"
fi
echo ""

echo "5. 비디오 디바이스 목록"
echo "-----------------------------------"
if ls /dev/video* >/dev/null 2>&1; then
    ls -la /dev/video*
else
    echo "비디오 디바이스(/dev/video*)를 찾을 수 없음"
fi
echo ""

echo "6. V4L2 디바이스 상세 정보"
echo "-----------------------------------"
if command -v v4l2-ctl &> /dev/null; then
    v4l2-ctl --list-devices 2>&1 || echo "v4l2-ctl 실행 실패"
    echo ""
    
    # 각 비디오 디바이스의 상세 정보
    for dev in /dev/video*; do
        if [ -e "$dev" ]; then
            echo "=== $dev 상세 정보 ==="
            v4l2-ctl --device=$dev --info 2>&1 | head -20
            echo "지원 형식:"
            v4l2-ctl --device=$dev --list-formats 2>&1 | head -10
            echo ""
        fi
    done
else
    echo "v4l2-ctl 명령어를 찾을 수 없음 (apt install v4l-utils 필요)"
fi

echo "7. 커널 모듈 확인"
echo "-----------------------------------"
echo "카메라 관련 모듈:"
lsmod | grep -i -E "(camera|bcm2835|v4l2)" || echo "카메라 관련 모듈을 찾을 수 없음"
echo ""

echo "8. rpicam 테스트 (5초간)"
echo "-----------------------------------"
if command -v rpicam-still &> /dev/null; then
    echo "rpicam-still로 테스트 이미지 촬영 시도..."
    timeout 10s rpicam-still -t 1 -o /tmp/test_camera.jpg --width 640 --height 480 2>&1
    if [ -f "/tmp/test_camera.jpg" ]; then
        ls -la /tmp/test_camera.jpg
        echo "테스트 이미지 촬영 성공!"
        rm -f /tmp/test_camera.jpg
    else
        echo "테스트 이미지 촬영 실패"
    fi
else
    echo "rpicam-still 명령어를 찾을 수 없음"
fi
echo ""

echo "9. 기존 raspistill 테스트 (호환성)"
echo "-----------------------------------"
if command -v raspistill &> /dev/null; then
    echo "raspistill로 테스트 이미지 촬영 시도..."
    timeout 10s raspistill -t 1 -o /tmp/test_camera_old.jpg -w 640 -h 480 2>&1
    if [ -f "/tmp/test_camera_old.jpg" ]; then
        ls -la /tmp/test_camera_old.jpg
        echo "기존 방식 테스트 이미지 촬영 성공!"
        rm -f /tmp/test_camera_old.jpg
    else
        echo "기존 방식 테스트 이미지 촬영 실패"
    fi
else
    echo "raspistill 명령어를 찾을 수 없음"
fi

echo ""
echo "========================================="
echo "카메라 정보 수집 완료"
echo "이 결과를 복사해서 Claude에게 보여주세요!"
echo "========================================="