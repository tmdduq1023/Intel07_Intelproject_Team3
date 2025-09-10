import requests
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import random

# 라즈베리파이의 실제 IP 주소로 변경
DESTINATION_URL = "http://192.168.0.90:5000/receive"  # http:// 추가 및 포트 5000
app = Flask(__name__)

test = True


def send_json(data: dict) -> None:
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


@app.route("/upload", methods=["POST"])
def recieve_image():
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        if not test:
            # uploads 디렉터리 생성
            upload_dir = "uploads"
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            filename = secure_filename(file.filename)
            file.save(os.path.join(upload_dir, filename))
        elif test:
            random_test_data: dict = {
                "forehead": {
                    "moisture": random.randint(0, 100),
                    "elasticity": random.randint(0, 100),
                    "pigmentation": random.randint(0, 100),
                },
                "l_check": {
                    "moisture": random.randint(0, 100),
                    "elasticity": random.randint(0, 100),
                    "pigmentation": random.randint(0, 100),
                    "pore": random.randint(0, 100),
                },
                "r_check": {
                    "moisture": random.randint(0, 100),
                    "elasticity": random.randint(0, 100),
                    "pigmentation": random.randint(0, 100),
                    "pore": random.randint(0, 100),
                },
                "chin": {
                    "moisture": random.randint(0, 100),
                    "elasticity": random.randint(0, 100),
                },
                "lib": {"elasticity": random.randint(0, 100)},
            }
            send_json(random_test_data)

        # Qt에서 기대하는 JSON 응답
        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Image uploaded successfully!",
                    "filename": "filename",  # filename
                }
            ),
            200,
        )

    return jsonify({"error": "Upload failed"}), 500


# 테스트 데이터
test_data = {
    "forehead": {"moisture": 70, "elasticity": 80, "pigmentation": 60},
    "l_check": {"moisture": 75, "elasticity": 85, "pigmentation": 65, "pore": 70},
    "r_check": {"moisture": 72, "elasticity": 82, "pigmentation": 62, "pore": 68},
    "chin": {"moisture": 68, "elasticity": 78},
    "lib": {"elasticity": 90},
}

if __name__ == "__main__":
    # send_json(test_data)
    app.run(host="0.0.0.0", port=5000, debug=True)
