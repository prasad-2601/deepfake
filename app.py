from flask import Flask, render_template, request
import os
from predict import predict_image
from video_predict import predict_video

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # check file type
        if file.filename.endswith((".jpg", ".png", ".jpeg")):
            result = predict_image(filepath)
        elif file.filename.endswith((".mp4", ".avi")):
            result = predict_video(filepath)
        else:
            result = "Unsupported file"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)