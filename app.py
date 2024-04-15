from time import sleep
import numpy as np
import cv2
import pprint

from CaptureManager import FrameStream
from WindowManager import Window
from BSManager import Detector
from TimeManager import Timer

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
SEGMENTATION_PREVIEW_WINDOW_NAME = "Backscatter Segmentation"
PROJECTOR_PREVIEW_WINDOW_NAME = "Projected Light Pattern"

### BSMANAGER PARAMETERS
#   CANNY_THRESHOLD_SIGMA: Threshold for the zero-parameter
#   Canny implementation - (https://pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/)
#
#   BS_MANAGER_HISTOGRAM_EQUALISATION: Whether or not to carry
#   out the histogram equalisation step.
#
#   BS_MANAGER_DEBUG_WINDOWS: Whether or not to display the intermediate
#   step visualisation.
CANNY_THRESHOLD_SIGMA = 0.33
BS_MANAGER_HISTOGRAM_EQUALISATION = True
BS_MANAGER_DEBUG_WINDOWS = False





if __name__ == "__main__":
    # Initialise capture stream
    stream = FrameStream(VIDEO_CAPTURE_SOURCE)
    
    # Initialise window to display the input
    input_feed_window = Window(INPUT_PREVIEW_WINDOW_NAME)

    # Initialise window to display the particle segmentation
    segmentation_window = Window(SEGMENTATION_PREVIEW_WINDOW_NAME)

    # Initialise window to display the projector output
    projector_window = Window(PROJECTOR_PREVIEW_WINDOW_NAME)

    # Initialise the backscatter detector
    detector = Detector(
        canny_threshold=CANNY_THRESHOLD_SIGMA,
        histogram_equalisation=BS_MANAGER_HISTOGRAM_EQUALISATION,
        debug_windows=BS_MANAGER_DEBUG_WINDOWS
    )

    # Initialise variable to track frame processing duration (sec)
    total_frame_processing_time = 0

    
    
    while True:
        # Read a frame
        frame = stream.read()

        # Detect keypress
        keypress = cv2.waitKey(1)

        # Exit if the 'e' key is pressed
        if keypress == ord('e'):
            break

        # While there are frames...
        if frame is not None:
            # Start timer for the total frame processing duration
            timer = Timer()

            # Update the input visualisation window
            input_feed_window.update(frame)

            # Detect the particles and retrieve real-time metrics
            particles, metrics = detector.detect(frame)

            # Create a black mask for the segmentation preview
            particle_mask = np.copy(frame)

            # Draw white circles on the black mask for each MEC
            for particle in particles:
                cv2.circle(
                    particle_mask,
                    particle[0],
                    particle[1],
                    (0, 0, 255),
                    1
                )

            # Display the black mask with white circles
            segmentation_window.update(particle_mask)

            # Create a white mask for the projector preview
            projector_mask = np.ones_like(frame) * 255

            # Draw black circles on the black mask for each MEC
            for particle in particles:
                cv2.circle(
                    projector_mask,
                    particle[0],
                    particle[1],
                    (0, 0, 0),
                    -1
                )

            # Display the white mask with black circles
            projector_window.update(projector_mask)

            # Stop the total frame processing duration timer
            total_frame_processing_time = timer.stop()

            # Append the total frame processing time to the metrics list
            metrics.append(total_frame_processing_time)

            # Print the metrics
            pprint.pprint(metrics)



    cv2.destroyAllWindows()

