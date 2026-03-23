import cv2
import os

# 🔥 SETTINGS (FAST TEST MODE)
MAX_VIDEOS = 10       # max videos per class
FRAME_SKIP = 10       # take 1 frame every 10
MAX_FRAMES = 50       # max frames per video


def extract_frames(video_path, output_folder, label):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    video_name = os.path.basename(video_path).split('.')[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % FRAME_SKIP == 0:
            frame = cv2.resize(frame, (224, 224))

            filename = f"{label}_{video_name}_{saved}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), frame)

            saved += 1

        count += 1

        if saved >= MAX_FRAMES:
            break

    cap.release()


def process_videos(input_folder, output_folder, label):
    os.makedirs(output_folder, exist_ok=True)

    videos = os.listdir(input_folder)

    # 🔥 limit videos
    videos = videos[:MAX_VIDEOS]

    for i, video in enumerate(videos):
        video_path = os.path.join(input_folder, video)

        print(f"[{label.upper()}] {i+1}/{len(videos)} → {video}")

        extract_frames(video_path, output_folder, label)


# ✅ YOUR CURRENT STRUCTURE
process_videos("dataset/real", "dataset_frames/real", "real")
process_videos("dataset/fake", "dataset_frames/fake", "fake")