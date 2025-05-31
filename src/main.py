import os
import typing

import tkinter as tk
import cv2 as cv

from ultralytics import YOLO
import ultralytics as ult

from deepface import DeepFace
from tkinter import filedialog
from PIL import Image, ImageTk

# ------------------ Constants ------------------ #

_BIT_MASK = 0xFF

_BG_COLOR: str = "#121212"
_FG_COLOR: str = "#FFFFFF"

_ROOT_RES:   tuple = (1900, 600)
_FRAME_RES:  tuple = (1000, 800)
_OUTPUT_RES: tuple = (500, 400)
_ICON_RES:   tuple = (170, 200)

_DATA_DIR:  tuple = ("dataset", "acne04yolov11")
_MODEL_DIR: str = "models"
_IMG_DIR:   str = "faces"
_ICON_DIR:  str = "icon"

_DATA_CONFIG_NAME:  str = "data.yaml"
_MODEL_NAME:        str = "yolo11n.pt"
_SAVE_MODEL_NAME:   str = "acne.pt"
_ICON_NAME:         str = "logo.png"
_IMG_NAME:          str = "saved.jpg"
_STREAM_DIR:        str = "video"


_DATA_CONFIG_PATH: str = os.path.realpath(
    os.path.join("..", *_DATA_DIR, _DATA_CONFIG_NAME)
)
_SAVE_MODEL_PATH:  str = os.path.realpath(
    os.path.join("..", _MODEL_DIR, _SAVE_MODEL_NAME)
)
_MODEL_PATH:       str = os.path.realpath(
    os.path.join("..", _MODEL_DIR, _MODEL_NAME)
)
_ICON_PATH:        str = os.path.realpath(
    os.path.join("..", _ICON_DIR, _ICON_NAME)
)
_IMG_PATH:         str = os.path.realpath(
    os.path.join("..", _IMG_DIR, _IMG_NAME)
)
_STREAM_PATH:      str = os.path.realpath(
    os.path.join("..", _STREAM_DIR)
)

# ------------------ Models ------------------ #


def train_save_YOLO() -> None:
    model = YOLO(_MODEL_PATH)
    trained_model = model.train(data=_DATA_CONFIG_PATH, epochs=100, imgsz=640)
    trained_model.save(_SAVE_MODEL_PATH)


def deep_face_video():
    DeepFace.stream(db_path=_STREAM_PATH)


def deep_face_analyze(file_path: str) -> list[dict[str, typing.Any]]:
    return DeepFace.analyze(
        file_path,
        actions=['age', 'gender', 'race', 'emotion'],
        enforce_detection=False
    )

# ------------------ Displays ------------------ #


def display_result(
        file_path: str,
        image_label: tk.Label,
        result_label: tk.Label
):

    panel = Image.open(file_path)
    image = panel.resize(_OUTPUT_RES)
    photo = ImageTk.PhotoImage(image)

    image_label.config(image=photo)
    image_label.image = photo

    result = deep_face_analyze(file_path)[0]

    result_label.config(
        text=f"Age: {result['age']}\n"
            f"Gender: { {k: float(v) for k, v in result['gender'].items()} }\n"
        f"Race: {result['dominant_race']}\n"
        f"Emotion: {result['dominant_emotion']}"
    )


def screenshot_video(
        image_label: tk.Label,
        result_label: tk.Label
) -> None:
    try:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Video capture is not initialized!")

        cap.set(cv.CAP_PROP_FRAME_WIDTH, _FRAME_RES[0])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, _FRAME_RES[1])

        while True:
            ret, frame = cap.read()

            if not ret:
                raise ValueError("Unable to capture frame!")
                break

            cv.imshow("Recording", frame)

            key_input = cv.waitKey(1) & _BIT_MASK

            if key_input == ord('s'):
                cv.imwrite(filename=_IMG_PATH, img=frame)
                print(
                    f"\n\n -------- "
                    f"\x1b[1mImage saved at {_IMG_PATH}.\x1b[0m"
                    f"--------\n"
                )
                cv.destroyAllWindows()
                display_result(_IMG_PATH, image_label, result_label)
                break
            elif key_input == ord('q'):
                break

        cv.destroyAllWindows()
        cap.release()

    except ValueError as e:
        print(f"error! {e}!")


def choose_file(image_label: tk.Label, result_label: tk.Label):
    chosen_file_path = filedialog.askopenfilename()
    if chosen_file_path:
        display_result(chosen_file_path, image_label, result_label)


def display_window() -> None:
    """ ------------------ Initializing window ------------------ """
    root = tk.Tk()
    root.title("Image Analyzer using DeepFace")
    root.geometry("x".join(str(val) for val in _ROOT_RES))
    root.configure(bg=_BG_COLOR)

    canvas = tk.Canvas(root, bg=_BG_COLOR, highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True)

    scroll_bar = tk.Scrollbar(root, command=canvas.yview)
    scroll_bar.pack(side="left", fill='y')
    canvas.configure(yscrollcommand=scroll_bar.set)

    canvas.bind(
        '<Configure>',
        lambda x: canvas.configure(scrollregion=canvas.bbox('all'))
    )

    x_center = _ROOT_RES[0] / 2
    y_center = _ROOT_RES[1] / 2

    frame = tk.Frame(canvas, bg=_BG_COLOR)

    canvas.create_window((x_center, y_center), window=frame, anchor="center")

    """ ------------------ Initializing Labels ------------------ """

    title_label = tk.Label(
        frame,
        text="Image Analyzer using DeepFace",
        fg=_FG_COLOR, bg=_BG_COLOR,
        font=("Arial", 20)
    )
    title_label.pack(pady=10)

    icon = Image.open(_ICON_PATH)
    icon = icon.resize(_ICON_RES)
    load_icon = ImageTk.PhotoImage(icon)
    icon_label = tk.Label(frame, image=load_icon)
    icon_label.pack(pady=50, side="top")

    root.iconphoto(False, load_icon)

    image_label = tk.Label(frame, bg=_BG_COLOR)
    image_label.pack()

    result_label = tk.Label(
        frame,
        fg=_FG_COLOR,
        bg=_BG_COLOR,
        font=("Arial", 14)
    )
    result_label.pack(pady=50)

    """ ------------------ Initializing Buttons ------------------ """

    button = tk.Button(
        frame, text="Choose Image",
        command=lambda: choose_file(image_label, result_label),
        bg="#4B4E5C", fg=_FG_COLOR, font=("Arial", 14)
    )
    button.pack(pady=20)

    button = tk.Button(
        frame, text="Screenshot Real-Time Video",
        command=lambda: screenshot_video(image_label, result_label),

        bg="#4B4E5C", fg=_FG_COLOR, font=("Arial", 14)
    )
    button.pack(pady=20)

    button = tk.Button(
        frame, text="Real-Time Facial Detection",
        command=lambda: deep_face_video(),
        bg="#4B4E5C", fg=_FG_COLOR, font=("Arial", 14)
    )
    button.pack(pady=20)

    """ ------------------ Run window ------------------ """

    root.mainloop()


def image_analyzer() -> None:
    try:
        train_save_YOLO()
    except KeyboardInterrupt as e:
        print(f"error! Video capture has exited abruptly! {e}")


def main() -> None:
    image_analyzer()


if __name__ == "__main__":
    main()
