import requests
import cv2
import os

def download_video(https://www.kaggle.com/code/xdxd003/download-ff-mega, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Downloaded:", save_path)


def extract_frames(video_path, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % 10 == 0:
            cv2.imwrite(f"{output_folder}/frame_{count}.jpg", frame)

        count += 1

    cap.release()


def process_video(url, label):

    video_path = "temp.mp4"

    download_video(url, video_path)

    output_folder = f"frames/{label}"
    extract_frames(video_path, output_folder)

    os.remove(video_path)
    print("Processed and deleted video")