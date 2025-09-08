from flask import Flask, request, jsonify
import serial
import threading
import os
import base64
from werkzeug.utils import secure_filename

ser = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)
ser_lock = threading.Lock()
app = Flask(__name__)


@app.route("/receive", methods=["POST"])
def receive():
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
        print(parts)

        data = "@".join(str(p) for p in parts)
        uart_write(data)
        return jsonify({"status": "sent"}), 200

    except (TypeError, KeyError) as e:
        return jsonify({"error": f"missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def uart_write(data: str):
    if not isinstance(data, str):
        data = str(data)
    with ser_lock:
        ser.write(data.encode("utf-8"))


@app.route("/upload", methods=["POST"])
def image_recieve():
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        # uploads 디렉터리 생성
        upload_dir = "uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        filename = secure_filename(file.filename)
        file.save(os.path.join(upload_dir, filename))

        # Qt에서 기대하는 JSON 응답
        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Image uploaded successfully!",
                    "filename": filename,
                }
            ),
            200,
        )

    return jsonify({"error": "Upload failed"}), 500


if __name__ == "__main__":
    # 모든 인터페이스에서 접근 가능하도록 설정
    app.run(host="0.0.0.0", port=5000, debug=True)
