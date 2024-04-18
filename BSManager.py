from multiprocessing import Process, Queue
import numpy as np
import logging
import cv2

from WindowManager import Window
from TimeManager import Timer

### DEBUG WINDOW NAMES
#   These constants store the names for each GUI window
GREYSCALE_DEBUG_WINDOW_NAME = "BSDetector Debug: Greyscale"
GAUSBLUR_DEBUG_WINDOW_NAME = "BSDetector Debug: Gaussian Blur"
CANNY_DEBUG_WINDOW_NAME = "BSDetector Debug: Canny Algorithm"
CONTOUR_DEBUG_WINDOW_NAME = "BSDetector Debug: Detected Contours"
HISTEQU_DEBUG_WINDOW_NAME = "BSDetector Debug: Histogram Equalisation"

PROCESS_QUEUE_QUIT_SIGNAL = "QUIT"
PROCESS_QUEUE_FQUIT_SIGNAL = "FQUIT"







class S1_Greyscale(Process):
    """ Process that applies a greyscale filter to an input. """

    def __init__(self, debug_windows, input_q, output_q):
        """
        debug_windows: Generates preview window for debugging if True.\n
        input_q: Queue that stores frames that require greyscaling.\n
        input_q's maxsize must be 1 for sequential stage processing (S1->S2->etc...).\n
        output_q: Queue that stores greyscaled frames.
        """
        super().__init__()
        self.debug_window = debug_windows
        self.input_q = input_q
        self.output_q = output_q

    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,BS-S1_Greyscale,%(message)s',
            filename='export_log-BSDetect-s1.csv',
            filemode='w'
        )

        # Log the start
        logging.info("Process started with PID=%d", self.pid)

        # Initialise frame counter to 0
        frame_count = 0

        if self.debug_window:
            self.greyscale_window = Window(GREYSCALE_DEBUG_WINDOW_NAME)

        while True:
            # Log the frame retrieval 
            logging.debug("Retrieving frame %d", frame_count)
            # Retrieve the frame (block until queue has something to dequeue)
            frame, metrics = self.input_q.get(block=True, timeout=None)
            # Check if the frame is a quit signal (check whether it's a string first!)
            if type(frame) == str:
                if frame == PROCESS_QUEUE_QUIT_SIGNAL or frame == PROCESS_QUEUE_FQUIT_SIGNAL:
                    # If it is then send quit signal to next stage and break
                    logging.info("Quit signal received - I am now quitting")
                    self.output_q.put((frame, None))
                    break
            # Log
            logging.debug("Greyscaling frame %d", frame_count)
            # Log start timestamp
            timer = Timer()
            # apply the single-channel conversion with greyscale filter
            greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Calculate process duration
            duration = timer.stop()
            if self.debug_window:
                self.greyscale_window.update(greyscale)
            # Log
            logging.debug("Greyscaled frame %d", frame_count)
            # Compile metrics
            metrics.append(duration)
            # Send everything to the next stage
            self.output_q.put((greyscale, metrics))
            # Increment frame count
            frame_count += 1





class S2_HistogramEqualisation(Process):
    """ Process that applies a histogram equalisation to an input. """

    def __init__(self, debug_windows, input_q, output_q):
        """
        debug_windows: Generates preview window for debugging if True.\n
        input_q: Queue that stores frames that require hist. equ.\n
        input_q's maxsize must be 1 for sequential stage processing (S1->S2->etc...).\n
        output_q: Queue that stores hist. equalised frames.
        """
        super().__init__()
        self.debug_window = debug_windows
        self.input_q = input_q
        self.output_q = output_q

    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,BS-S2_HistogramEqualisation,%(message)s',
            filename='export_log-BSDetect-s2.csv',
            filemode='w'
        )

        # Log the start
        logging.info("Process started with PID=%d", self.pid)

        # Initialise frame counter to 0
        frame_count = 0

        if self.debug_window:
            self.histequ_window = Window(HISTEQU_DEBUG_WINDOW_NAME)

        while True:
            # Log the frame retrieval 
            logging.debug("Retrieving frame %d", frame_count)
            # Retrieve the frame (block until queue has something to dequeue)
            frame, metrics = self.input_q.get(block=True, timeout=None)
            # Check if the frame is a quit signal (check whether it's a string first!)
            if type(frame) == str:
                if frame == PROCESS_QUEUE_QUIT_SIGNAL or frame == PROCESS_QUEUE_FQUIT_SIGNAL:
                    # If it is then send quit signal to next stage and break
                    logging.info("Quit signal received - I am now quitting")
                    self.output_q.put((frame, None))
                    break
            # Log
            logging.debug("Histogram equalising frame %d", frame_count)
            # Log start timestamp
            timer = Timer()
            # apply the histogram equalisation
            histequ = cv2.equalizeHist(frame)
            # Calculate process duration
            duration = timer.stop()
            # Display preview if debug True
            if self.debug_window:
                self.histequ_window.update(histequ)
            # Compile metrics
            metrics.append(duration)
            # Log
            logging.debug("Histogram equalised frame %d", frame_count)
            # Send everything to the next stage
            self.output_q.put((histequ, metrics))
            # Increment frame count
            frame_count += 1







class S3_GaussianBlur(Process):
    """ Process that applies a Gaussian blur to an input. """

    def __init__(self, debug_windows, input_q, output_q):
        """
        debug_windows: Generates preview window for debugging if True.\n
        input_q: Queue that stores frames that require Gaus. blurring.\n
        input_q's maxsize must be 1 for sequential stage processing (S1->S2->etc...).\n
        output_q: Queue that stores blurred frames.
        """
        super().__init__()
        self.debug_window = debug_windows
        self.input_q = input_q
        self.output_q = output_q

    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,BS-S3_GaussianBlur,%(message)s',
            filename='export_log-BSDetect-s3.csv',
            filemode='w'
        )

        # Log the start
        logging.info("Process started with PID=%d", self.pid)

        # Initialise frame counter to 0
        frame_count = 0

        if self.debug_window:
            self.gausblur_window = Window(GAUSBLUR_DEBUG_WINDOW_NAME)

        while True:
            # Log the frame retrieval 
            logging.debug("Retrieving frame %d", frame_count)
            # Retrieve the frame (block until queue has something to dequeue)
            frame, metrics = self.input_q.get(block=True, timeout=None)
            # Check if the frame is a quit signal (check whether it's a string first!)
            if type(frame) == str:
                if frame == PROCESS_QUEUE_QUIT_SIGNAL or frame == PROCESS_QUEUE_FQUIT_SIGNAL:
                    # If it is then send quit signal to next stage and break
                    logging.info("Quit signal received - I am now quitting")
                    self.output_q.put((frame, None))
                    break
            # Log 
            logging.debug("Gaussian blurring frame %d", frame_count)
            # Log start timestamp
            timer = Timer()
            # apply the gaussian blur
            gausblur = cv2.GaussianBlur(frame, (5,5), 0)
            # Calculate process duration
            duration = timer.stop()
            # Display preview if debug True
            if self.debug_window:
                self.gausblur_window.update(gausblur)
            # Compile metrics
            metrics.append(duration)
            # Log
            logging.debug("Gaussian blurred frame %d", frame_count)
            # Send everything to the next stage
            self.output_q.put((gausblur, metrics))
            # Increment frame count
            frame_count += 1







class S4_Canny(Process):
    """ Process that applies the Canny algorithm to an input. """

    def __init__(self, debug_windows, canny_threshold, input_q, output_q):
        """
        debug_windows: Generates preview window for debugging if True.\n
        input_q: Queue that stores frames to apply Canny with.\n
        input_q's maxsize must be 1 for sequential stage processing (S1->S2->etc...).\n
        output_q: Queue that stores blurred frames.
        """
        super().__init__()
        self.debug_window = debug_windows
        self.input_q = input_q
        self.output_q = output_q
        self.canny_threshold = canny_threshold

    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,BS-S4_Canny,%(message)s',
            filename='export_log-BSDetect-s4.csv',
            filemode='w'
        )

        # Log the start
        logging.info("Process started with PID=%d", self.pid)

        # Initialise frame counter to 0
        frame_count = 0

        if self.debug_window:
            self.canny_window = Window(CANNY_DEBUG_WINDOW_NAME)

        while True:
            # Log the frame retrieval 
            logging.debug("Retrieving frame %d", frame_count)
            # Retrieve the frame (block until queue has something to dequeue)
            frame, metrics = self.input_q.get(block=True, timeout=None)
            # Check if the frame is a quit signal (check whether it's a string first!)
            if type(frame) == str:
                if frame == PROCESS_QUEUE_QUIT_SIGNAL or frame == PROCESS_QUEUE_FQUIT_SIGNAL:
                    # If it is then send quit signal to next stage and break
                    logging.info("Quit signal received - I am now quitting")
                    self.output_q.put((frame, None, None))
                    break
            # Log 
            logging.debug("Applying Canny to frame %d", frame_count)
            # Log start timestamp
            timer = Timer()
            # compute the median single-channel pixel intensities
            gaus_median = np.median(frame)
            # compute threshold values for canny using single parameter Canny
            lower_threshold = int(max(0, (1.0 - self.canny_threshold) * gaus_median))
            upper_threshold = int(min(255, (1.0 + self.canny_threshold) * gaus_median))
            # perform Canny edge detection
            edges = cv2.Canny(
                frame,
                lower_threshold,
                upper_threshold
            )
            # Calculate process duration
            duration = timer.stop()
            # Display preview if debug True
            if self.debug_window:
                self.canny_window.update(edges)
            # Compile metrics
            metrics.append(duration)
            # Log
            logging.debug("Applied Canny to frame %d", frame_count)
            # Send everything to the next stage
            self.output_q.put((frame, edges, metrics))
            # Increment frame count
            frame_count += 1







class S5_Segmentation(Process):
    """ Process that finds contours then calculates the minumum enclosing circles. """

    def __init__(self, debug_windows, input_q, output_q):
        """
        debug_windows: Generates preview window for debugging if True.\n
        input_q: Queue that stores edges to segment.\n
        input_q's maxsize must be 1 for sequential stage processing (S1->S2->etc...).\n
        output_q: Queue that stores segmented particles.
        """
        super().__init__()
        self.debug_window = debug_windows
        self.input_q = input_q
        self.output_q = output_q

    def run(self,):
        # Turn on logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(created).6f,%(levelname)s,BS-S5_Segmentation,%(message)s',
            filename='export_log-BSDetect-s5.csv',
            filemode='w'
        )

        # Log the start
        logging.info("Process started with PID=%d", self.pid)

        # Initialise frame counter to 0
        frame_count = 0

        if self.debug_window:
            self.contour_window = Window(CONTOUR_DEBUG_WINDOW_NAME)

        while True:
            # Log the frame retrieval 
            logging.debug("Retrieving frame %d", frame_count)
            # Retrieve the edges (block until queue has something to dequeue)
            frame, edges, metrics = self.input_q.get(block=True, timeout=None)
            # Check if the edges is a quit signal (check whether it's a string first!)
            if type(frame) == str:
                if frame == PROCESS_QUEUE_QUIT_SIGNAL or frame == PROCESS_QUEUE_FQUIT_SIGNAL:
                    # If it is then send quit signal to next stage and break
                    logging.info("Quit signal received - I am now quitting")
                    self.output_q.put((frame, None, None))
                    break
            # Log 
            logging.debug("Applying segmentation to frame %d", frame_count)
            # Log start timestamp
            timer = Timer()
            # 01 - Find the contours
            # RETR_EXTERNAL only retrieves the extreme outer contours
            # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and
            #   diagonal segments and leaves only their end points
            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,       # RetrievalModes
                cv2.CHAIN_APPROX_SIMPLE  # ContourApproximationModes
            )
            # Calculate process duration
            duration = timer.stop()
            # Compile metrics
            metrics.append(duration)
            # Start another timer
            timer = Timer()
            # List to store the particle information (centre coords + radius)
            particles = []
            # 02 - Find minimum enclosuing circles
            for contour in contours:
                # Find the minimum enclosing circle for each contour
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                # Find the centre
                centre = (int(x), int(y))
                # Find the radius
                radius = int(radius)
                # Store the information
                particles.append((centre, radius))
            # Calculate process duration
            duration = timer.stop()
            # Compile metrics
            metrics.append(duration)
            # Compile metrics
            metrics.append(len(particles))
            # Display preview if debug True
            if self.debug_window:
                # create a black mask
                mask = np.zeros_like(edges)
                # draw contours white white fill
                cv2.drawContours(mask, contours, -1, (255), cv2.FILLED)
                # display window
                self.contour_window.update(mask)
            # Log
            logging.debug("Applied segmentation to frame %d", frame_count)
            # Send everything to the next stage
            self.output_q.put((frame, particles, metrics))
            # Increment frame count
            frame_count += 1








class Detector:
    """
    Backscatter detection logic (V1):
        (a) edges are detected using the Canny algorithm.\n
        (b) the detected edges are segmented using minimum enclosing circles (MECs).\n
        (c) the centre coordinates and radius of the detected MECs are returned.
    """

    def __init__(self, canny_threshold, debug_windows=True):
        # Zero-parameter threshold for canny (https://pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/)
        self.canny_threshold = canny_threshold
        # Whether or not to print the intermediate step visualisation
        self.debug_windows = debug_windows

        self.input_q = Queue(1)  # queue to input to the first BS stage
        self.q1_2 = Queue(1)     # queue between S1 and S2
        self.q2_3 = Queue(1)     # queue between S2 and S3
        self.q3_4 = Queue(1)     # queue between S3 and S4
        self.q4_5 = Queue(1)     # queue between S4 and S5
        self.output_q = Queue(1) # queue to output to the main script

        self.stages = [
            S1_Greyscale(input_q=self.input_q, output_q=self.q1_2, debug_windows=self.debug_windows),
            S2_HistogramEqualisation(input_q=self.q1_2, output_q=self.q2_3, debug_windows=self.debug_windows),
            S3_GaussianBlur(input_q=self.q2_3, output_q=self.q3_4, debug_windows=self.debug_windows),
            S4_Canny(input_q=self.q3_4, output_q=self.q4_5, debug_windows=self.debug_windows, canny_threshold=self.canny_threshold),
            S5_Segmentation(input_q=self.q4_5, output_q=self.output_q, debug_windows=self.debug_windows)
        ]


    def detect(self):
        return self.stages, self.input_q, self.output_q


