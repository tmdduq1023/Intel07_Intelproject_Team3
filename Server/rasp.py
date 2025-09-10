from flask import Flask, request, jsonify
import serial
import threading
import requests

ser = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)
ser_lock = threading.Lock()
app = Flask(__name__)

# 최신 분석 결과 저장
latest_analysis_data = None


@app.route("/receive", methods=["POST"])
def receive_json():
    global latest_analysis_data

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "invalid json"}), 400

    try:
        parts = [
            payload["forehead"]["moisture"],
            payload["forehead"]["elasticity"],
            payload["forehead"]["pigmentation"],
            payload["l_check"]["moisture"],
            payload["l_check"]["elasticity"],
            payload["l_check"]["pigmentation"],
            payload["l_check"]["pore"],
            payload["r_check"]["moisture"],
            payload["r_check"]["elasticity"],
            payload["r_check"]["pigmentation"],
            payload["r_check"]["pore"],
            payload["chin"]["moisture"],
            payload["chin"]["elasticity"],
            payload["lib"]["elasticity"],
        ]
        print(f"분석 결과 수신: {parts}")

        # 최신 분석 데이터 저장
        latest_analysis_data = payload

        # UART로 하드웨어에 전송
        data = "@".join(str(p) for p in parts)
        uart_write(data)

        print(f"하드웨어로 전송 완료: {data}")

        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Data received and sent to hardware",
                    "analysis_data": payload,
                }
            ),
            200,
        )

    except (TypeError, KeyError) as e:
        return jsonify({"error": f"missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------- 테스트용 코드---------------
@app.route("/get_analysis", methods=["GET"])
def get_analysis_data():
    """Qt 클라이언트가 최신 분석 결과를 요청하는 엔드포인트"""
    global latest_analysis_data

    print(f"Qt client requesting analysis data. Current data: {latest_analysis_data}")

    if latest_analysis_data:
        response_data = {"status": "success", "analysis_data": latest_analysis_data}
        print(f"Sending success response: {response_data}")
        return jsonify(response_data), 200
    else:
        response_data = {"status": "no_data", "message": "No analysis data available"}
        print(f"Sending no_data response: {response_data}")
        return jsonify(response_data), 200


@app.route("/test_data", methods=["GET"])
def send_test_data():
    """테스트용: 샘플 분석 데이터를 생성하고 저장"""
    global latest_analysis_data

    # 테스트 데이터 생성
    test_analysis_data = {
        "forehead": {"moisture": 75, "elasticity": 82, "pigmentation": 58},
        "l_check": {"moisture": 78, "elasticity": 88, "pigmentation": 62, "pore": 65},
        "r_check": {"moisture": 76, "elasticity": 85, "pigmentation": 60, "pore": 68},
        "chin": {"moisture": 72, "elasticity": 80},
        "lib": {"elasticity": 92},
    }

    # 분석 데이터 저장
    latest_analysis_data = test_analysis_data

    print(f"테스트 데이터 생성 및 저장: {test_analysis_data}")

    return (
        jsonify(
            {
                "status": "success",
                "message": "Test analysis data created and saved",
                "test_data": test_analysis_data,
            }
        ),
        200,
    )


@app.route("/clear_data", methods=["GET"])
def clear_analysis_data():
    """테스트용: 저장된 분석 데이터 초기화"""
    global latest_analysis_data

    latest_analysis_data = None
    print("분석 데이터 초기화됨")

    return jsonify({"status": "success", "message": "Analysis data cleared"}), 200


# -------------------


def uart_write(data: str):
    if not isinstance(data, str):
        data = str(data)
    with ser_lock:
        ser.write(data.encode("utf-8"))


if __name__ == "__main__":
    # 모든 인터페이스에서 접근 가능하도록 설정
    app.run(host="0.0.0.0", port=5000, debug=True)
