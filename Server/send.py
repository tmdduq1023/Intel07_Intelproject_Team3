"""
연산 노드 측 코드야

라즈베리파이한테 발신.
이 코드를 ai 모델이 호출해서 딕셔너리를 인자로 주면 그걸 json으로 변경해서 발신함.


* 추가 사항
사진을 입력받아서 리턴하는 함수.
이 함수를 ai 모델에서 import해서 입력받으면 사진을 리턴받을 수 있어.
""" 

import requests

# 라즈베리파이의 실제 IP 주소로 변경
DESTINATION_URL = "http://192.168.0.90:5000/receive"  # http:// 추가 및 포트 5000


def send_data(data: dict) -> None:
    try:
        response = requests.post(
            DESTINATION_URL,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        print("전송 완료!")
        print(f"상태코드: {response.status_code}")
        print(f"응답: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"전송 실패: {e}")
    except Exception as e:
        print(f"오류 발생: {e}")


# 테스트 데이터
test_data = {
    "forehead": {"moisture": 70, "elasticity": 80, "pigmentation": 60},
    "l_check": {"moisture": 75, "elasticity": 85, "pigmentation": 65, "pore": 70},
    "r_check": {"moisture": 72, "elasticity": 82, "pigmentation": 62, "pore": 68},
    "chin": {"moisture": 68, "elasticity": 78},
    "lib": {"elasticity": 90},
}

if __name__ == "__main__":
    send_data(test_data)
