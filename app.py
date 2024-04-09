from time import sleep
import cv2

from CaptureManager import FrameStream
from WindowManager import Window
from BSManager import Detector

### VIDEO CAPTURE SOURCE
#   Input a directory path to feed in a series of frame images,
#   named as an integer denoting the frame number and must be
#   in PNG format.
#
#   Input a file path to feed in a video file.
#
#   Input integer '0' to use Picamera2 capture_array() to capture
#   feed frame-by-frame.
VIDEO_CAPTURE_SOURCE = "./export_04-08-2024-14-35-53/"

### VIDEO CAPTURE RESOLUTION
#   These are the recording parameters which dictate capture
#   resolution.
#
#   When wanting to use the frame-by-frame output from the
#   bubble-backscatter-simulation program, set these values
#   to the same as the ones input to that program (800x600).
#
#   When wanting to use a pre-recorded video source, these
#   values will be updated to match the correct resolution
#   of the video. Ensure they are similar to avoid confusion.
#
#   Want wanting to use the Pi Camera feed, these values will
#   be used when configuring the camera resolution parameters,
#   however, the camera will align the stream size to force
#   optimal alignment, so the resolution may be slightly
#   different.
VIDEO_CAPTURE_WIDTH = 800
VIDEO_CAPTURE_HEIGHT = 600

### PREVIEW WINDOW NAMES
#   These constants store the names for each GUI window
INPUT_PREVIEW_WINDOW_NAME = "Input Feed"





CANNY_THRESHOLD_SIGMA = 0.5








if __name__ == "__main__":

    stream = FrameStream(VIDEO_CAPTURE_SOURCE)
    
    input_feed_window = Window(INPUT_PREVIEW_WINDOW_NAME)

    detector = Detector(
        canny_threshold=CANNY_THRESHOLD_SIGMA,
        histogram_equalisation=True,
        debug_windows=True
    )

    while True:

        frame = stream.read()

        keypress = cv2.waitKey(1)

        if keypress == ord('e'):
            break

        if frame is not None:
            input_feed_window.update(frame)

            canny = detector.detect(frame)

            cv2.waitKey(0)

    input_feed_window.destroy()

