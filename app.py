from multiprocessing import Process, Queue
from datetime import datetime
import numpy as np
import pandas as pd
import logging
import cv2
import os

from CaptureManager import FrameStream, VideoStream, PicameraStream
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
VIDEO_CAPTURE_SOURCE = "./import_04-08-2024-14-35-53/"

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


PROCESS_QUEUE_QUIT_SIGNAL = "QUIT"



class S1_Capture(Process):
    """ Acquires and enqueues frames from the capture source. """

    def __init__(self, output_q):
        """
        output_q: Queue to store captured frames.\n
        All queues' maxsize must be 1 for sequential stage processing (S1->S2->S3).
        """
        super().__init__()
        # mp.Event that stores queue of captured frames
        self.output_q = output_q

        # Variable to store process durations
        self.capture_duration = 0

    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,S1_Capture,%(message)s',
            filename='export_log-s1.csv',
            filemode='w'
        )

        # Log the start
        logging.info("Process started")

        # Initialise capture stream
        match VIDEO_CAPTURE_SOURCE:
            # If int 0 then set up Pi camera stream
            case 0:
                # Log
                logging.debug("Initialising PicameraStream")
                # Initialise FrameStream
                stream = PicameraStream(VIDEO_CAPTURE_WIDTH, VIDEO_CAPTURE_HEIGHT)
                # Log
                logging.debug("PicameraStream initialised")
            # If not int 0 then check if it is a valid path
            case _:
                # If the path is a directory, then FrameStream
                if os.path.isdir(VIDEO_CAPTURE_SOURCE):
                    # Log
                    logging.debug("Initialising FrameStream")
                    # Initialise FrameStream
                    stream = FrameStream(VIDEO_CAPTURE_SOURCE)
                    # Log
                    logging.debug("FrameStream initialised")
                # If the path is a file, then VideoStream
                elif os.path.isfile(VIDEO_CAPTURE_SOURCE):
                    # Log
                    logging.debug("Initialising VideoStream")
                    # Initialise VideoStream
                    stream = VideoStream(VIDEO_CAPTURE_SOURCE)
                    # Log
                    logging.debug("VideoStream initialised")

        # Initialise frame counter to 0
        frame_count = 0

        # Initialise window to display the input
        input_feed_window = Window(INPUT_PREVIEW_WINDOW_NAME)

        while True:
            # Keep capturing frames until the end of the file
            if not stream.empty():
                # Log the frame retrieval 
                logging.debug("Retrieving frame %d", frame_count)
                # Capture the frame
                frame, capture_metrics = stream.read()
                # Enqueue the frame (block until queue's size<maxsize)
                self.output_q.put((frame, [capture_metrics]), block=True, timeout=None)
                # Update the input visualisation window
                input_feed_window.update(frame)
                # Log the frame enqueue 
                logging.debug("Retrieved and enqueued frame %d", frame_count)
                # Increment frame count
                frame_count += 1
            else:
                # Handle event where capture has finished!
                logging.info("No frames left to capture - sending quit signal")
                self.output_q.put((PROCESS_QUEUE_QUIT_SIGNAL, None), block=True, timeout=None)
                break



class S2_Process(Process):
    """ Processes frames to detect backscatter particles. """

    def __init__(self, input_q, output_q):
        """
        input_q: Queue that stores captured frames.\n
        output_q: Queue to store computed backscatter particles.\n
        All queues' maxsize must be 1 for sequential stage processing (S1->S2->S3).
        """
        super().__init__()
        # mp.Event that stores queue of captured frames ready for processing
        self.input_q = input_q
        # mp.Event that stores queue of detected backscatter particle positions
        self.output_q = output_q

    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,S2_Process,%(message)s',
            filename='export_log-s2.csv',
            filemode='w'
        )

        # Log the start
        logging.info("Process started")

        # Log
        logging.debug("Initialising BSDetector")

        # Initialise the backscatter detector
        detector = Detector(
            canny_threshold=CANNY_THRESHOLD_SIGMA,
            histogram_equalisation=BS_MANAGER_HISTOGRAM_EQUALISATION,
            debug_windows=BS_MANAGER_DEBUG_WINDOWS
        )

        # Log
        logging.debug("BSDetector initialised")

        # Initialise window to display the particle segmentation
        segmentation_window = Window(SEGMENTATION_PREVIEW_WINDOW_NAME)

        # Initialise frame counter to 0
        frame_count = 0

        while True:
            # Log the frame retrieval 
            logging.debug("Retrieving frame %d", frame_count)
            # Retrieve the frame (block until queue has something to dequeue)
            frame, metrics = self.input_q.get(block=True, timeout=None)
            # Check if the frame is a quit signal (check whether it's a string first!)
            if type(frame) == str:
                if frame == PROCESS_QUEUE_QUIT_SIGNAL:
                    # If it is then send quit signal to S3 and break
                    logging.info("Frame %d was actually a quit signal - I am now quitting", frame_count)
                    self.output_q.put((PROCESS_QUEUE_QUIT_SIGNAL, None, None), block=True, timeout=None)
                    break
            # Log the frame retrieval 
            logging.debug("Processing frame %d", frame_count)
            # Process the frame
            particles, detector_metrics = detector.detect(frame)
            # Join frame's detect metrics to frame's capture metrics
            metrics = metrics + detector_metrics
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
            # Log the frame retrieval and processing 
            logging.debug("Processed frame %d", frame_count)
            # Enqueue output queue for the next stage (block until queue's size<maxsize)
            self.output_q.put((frame, particles, metrics), block=True, timeout=None)
            # Increment frame count
            frame_count += 1






class S3_Project(Process):
    """ Project the backscatter-cancelling light patterns. """

    def __init__(self, input_q, output_q):
        """
        input_q: Queue that stores computed backscatter particles.\n
        input_q's maxsize must be 1 for sequential stage processing (S1->S2->S3).\n
        output_q: Queue that stores metrics across all stages to be logged.
        """
        super().__init__()
        # mp.Event that stores queue of backscatter particles in each frame
        self.input_q = input_q
        self.output_q = output_q

    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,S3_Project,%(message)s',
            filename='export_log-s3.csv',
            filemode='w'
        )

        # Log the start
        logging.info("Process started")

        # Initialise window to display the projector output
        projector_window = Window(PROJECTOR_PREVIEW_WINDOW_NAME)

        # Initialise frame counter to 0
        frame_count = 0

        while True:
            # Log the particles retrieval 
            logging.debug("Retrieving particles in frame %d", frame_count)
            # Retrieve the particles (block until queue has something to dequeue)
            frame, particles, metrics = self.input_q.get(block=True, timeout=None)
            # Check if the frame is a quit signal (check whether it's a string first!)
            if type(frame) == str:
                if frame == PROCESS_QUEUE_QUIT_SIGNAL:
                    # If it is then send quit signal to Logging and break
                    logging.info("Frame %d was actually a quit signal - I am now quitting!", frame_count)
                    self.output_q.put(PROCESS_QUEUE_QUIT_SIGNAL)
                    break
            # Log the particles retrieval 
            logging.debug("Projecting frame %d", frame_count)
            # Create a white mask for the projector preview
            projector_mask = np.ones_like(frame) * 255
            # Process the particles:
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
            # Log the particle retrieval and processing 
            logging.debug("Projected frame %d", frame_count)
            # Send metrics to logger
            self.output_q.put(metrics)
            # Increment frame count
            frame_count += 1






class Logging(Process):
    """ Log frame-by-frame real time metrics. """

    def __init__(self, input_q):
        """
        input_q: Queue that stores each frame's metrics.\n
        """
        super().__init__()
        self.input_q = input_q

    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,Logging,%(message)s',
            filename='export_log-logging.csv',
            filemode='w'
        )

        # Log the start
        logging.info("Process started")


        # Generate CSV export filename
        export_filename_csv = "export_" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + ".csv"

        # Initialise a Pandas DataFrame to log real-time metrics
        rt_metrics_df = pd.DataFrame(
            columns=[
                'Capture Duration (s)',
                'Greyscale Conversion Duration (s)',
                'Histogram Equalisation Duration (s)',
                'Gaussian Blur Duration (s)',
                'Canny Algorithm Duration (s)',
                'CV2 findContours() Duration (s)',
                'CV2 minEnclosingCircle() Duration (s)',
                'Number of MECs on screen',
            ]
        )

        # Initialise frame counter to 0
        frame_count = 0

        while True:
            # Log the particles retrieval 
            logging.debug("Retrieving metrics for frame %d", frame_count)
            # Retrieve the particles (block until queue has something to dequeue)
            metrics = self.input_q.get(block=True, timeout=None)
            # Check if the frame is a quit signal (check whether it's a string first!)
            if type(metrics) == str:
                if metrics == PROCESS_QUEUE_QUIT_SIGNAL:
                    # If it is then export to CSV and break
                    logging.info("Frame %d data was actually a quit signal - starting to export", frame_count)
                    # Export dataframe as CSV
                    rt_metrics_df.to_csv(
                        path_or_buf=export_filename_csv,
                        encoding='utf-8'
                    )
                    logging.info("Successfully exported - now exiting")
                    break
            # Log the particles retrieval 
            logging.debug("Logging data for frame %d", frame_count)
            # Add this frame's metrics to the end of the dataframe
            rt_metrics_df.loc[len(rt_metrics_df)] = metrics
            # Log the particle retrieval and processing 
            logging.debug("Logged data for frame %d", frame_count)
            # Increment frame count
            frame_count += 1








if __name__ == "__main__":
    q1_2 = Queue(1) # queue between stages 1 and 2
    q2_3 = Queue(1) # queue between stages 2 and 3
    q3_logging = Queue() # queue between stages 3 and Logging

    # Create Processes for stages of pipeline
    stages = []
    stages.append(S1_Capture(output_q=q1_2))
    stages.append(S2_Process(input_q=q1_2, output_q=q2_3))
    stages.append(S3_Project(input_q=q2_3, output_q=q3_logging))
    stages.append(Logging(input_q=q3_logging))


    # Start the stages
    for stage in stages:
        stage.start()

    # Wait for stages to finish
    for stage in stages:
        stage.join()

    cv2.destroyAllWindows()




# if __name__ == "__main__":


    
#     while True:
#         # Read a frame
#         frame = stream.read()

#         # Detect keypress
#         keypress = cv2.waitKey(1)

#         # Exit if the 'e' key is pressed
#         if keypress == ord('e'):
#             break

#         # While there are frames...
#         if frame is not None:
#             # Start timer for the total frame processing duration
#             timer = Timer()

#             # Update the input visualisation window
#             input_feed_window.update(frame)

#             # Detect the particles and retrieve real-time metrics
#             particles, metrics = detector.detect(frame)

#             # Create a black mask for the segmentation preview
#             particle_mask = np.copy(frame)

#             # Draw white circles on the black mask for each MEC
#             for particle in particles:
#                 cv2.circle(
#                     particle_mask,
#                     particle[0],
#                     particle[1],
#                     (0, 0, 255),
#                     1
#                 )

#             # Display the black mask with white circles
#             segmentation_window.update(particle_mask)

#             # Create a white mask for the projector preview
#             projector_mask = np.ones_like(frame) * 255

#             # Draw black circles on the black mask for each MEC
#             for particle in particles:
#                 cv2.circle(
#                     projector_mask,
#                     particle[0],
#                     particle[1],
#                     (0, 0, 0),
#                     -1
#                 )

#             # Display the white mask with black circles
#             projector_window.update(projector_mask)

#             # Stop the total frame processing duration timer
#             total_frame_processing_time = timer.stop()

#             # Append the total frame processing time to the metrics list
#             metrics.append(total_frame_processing_time)

#             # Append metrics list to end of dataframe
#             rt_metrics_df.loc[len(rt_metrics_df)] = metrics
    
#     # Export dataframe as CSV
#     rt_metrics_df.to_csv(
#         path_or_buf=export_filename_csv,
#         encoding='utf-8'
#     )

#     cv2.destroyAllWindows()

