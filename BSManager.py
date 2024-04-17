import cv2
import numpy as np

from WindowManager import Window
from TimeManager import Timer

### DEBUG WINDOW NAMES
#   These constants store the names for each GUI window
GREYSCALE_DEBUG_WINDOW_NAME = "BSDetector Debug: Greyscale"
GAUSBLUR_DEBUG_WINDOW_NAME = "BSDetector Debug: Gaussian Blur"
CANNY_DEBUG_WINDOW_NAME = "BSDetector Debug: Canny Algorithm"
CONTOUR_DEBUG_WINDOW_NAME = "BSDetector Debug: Detected Contours"
HISTEQU_DEBUG_WINDOW_NAME = "BSDetector Debug: Histogram Equalisation"

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

        # Initialise the real-time metric variables
        self._reset_metrics()

        # Initialise the debug windows if enabled
        if self.debug_windows:
            self.greyscale_window = Window(GREYSCALE_DEBUG_WINDOW_NAME)
            self.gausblur_window = Window(GAUSBLUR_DEBUG_WINDOW_NAME)
            self.canny_window = Window(CANNY_DEBUG_WINDOW_NAME)
            self.contour_window = Window(CONTOUR_DEBUG_WINDOW_NAME)
            self.histequ_window = Window(HISTEQU_DEBUG_WINDOW_NAME)




    def _reset_metrics(self):
        """ (Internal) Initialises the real-time tracking metrics. """
        # Time it takes to apply greyscale filter
        self.greyscale_process_duration = 0
        # Time it takes to apply histogram equalisation
        self.histequ_process_duration = 0
        # Time it takes to apply a Gaussian blur
        self.gausblur_process_duration = 0
        # Time it takes to apply the Canny algorithm
        self.canny_process_duration = 0
        # Time it takes to find contours
        self.findContours_process_duration = 0
        # Time it takes to circle the contours (min enclosing circles)
        self.circleContours_process_duration = 0
        # The total number of MECs on screen
        self.particle_count = 0




    def _get_metrics(self):
        """ (Internal) Returns the real-time metrics using a list. """

        timings = [
            self.greyscale_process_duration,        # Greyscale Conversion Duration (s)
            self.histequ_process_duration,          # Histogram Equalisation Duration (s)
            self.gausblur_process_duration,         # Gaussian Blur Duration (s)
            self.canny_process_duration,            # Canny Algorithm Duration (s)
            self.findContours_process_duration,     # CV2 findContours() Duration (s)
            self.circleContours_process_duration,   # CV2 minEnclosingCircle() Duration (s)
            self.particle_count                     # Number of MECs on screen
        ]

        return timings




    def _greyscale(self, frame):
        """ (Internal) Applies a greyscale filter to the frame. """

        # Log start timestamp
        timer = Timer()

        # apply the single-channel conversion with greyscale filter
        greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate process duration
        self.greyscale_process_duration = timer.stop()

        # output to the debug window if enabled
        if self.debug_windows:
            self.greyscale_window.update(greyscale)

        # return the greyscaled frame
        return greyscale




    def _histequ(self, frame):
        """ (Internal) Applies a histogram equalisation to the frame. """

        # Log start timestamp
        timer = Timer()

        # apply histogram equalisation
        histequ = cv2.equalizeHist(frame)

        # Calculate process duration
        self.histequ_process_duration = timer.stop()

        # output to the debug window if enabled
        if self.debug_windows:
            self.histequ_window.update(histequ)
        
        # return the histogram equalised frame
        return histequ




    def _gausblur(self, frame):
        """ (Internal) Applies a Gaussian blur to the frame. """

        # Log start timestamp
        timer = Timer()

        # apply the Gaussian blur
        gausblur = cv2.GaussianBlur(frame, (5,5), 0)

        # Calculate process duration
        self.gausblur_process_duration = timer.stop()

        # output to the debug window if enabled
        if self.debug_windows:
            self.gausblur_window.update(gausblur)
        
        # return the Gaussian blurred frame
        return gausblur




    def _segmentation(self, edges):
        """ (Internal) Finds contours using cv2.findContours() then calculates the minumum enclosing circles using cv2.minEnclosingCircle(). """

        # Log start timestamp of findContours()
        timer_findContours = Timer()

        # RETR_EXTERNAL only retrieves the extreme outer contours
        # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and
        #   diagonal segments and leaves only their end points
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,       # RetrievalModes
            cv2.CHAIN_APPROX_SIMPLE  # ContourApproximationModes
        )

        # Calculate findContours process duration
        self.findContours_process_duration = timer_findContours.stop()

        # Log start timestamp of minEnclosingCircle()
        timer_circleContours = Timer()

        # List to store the particle information (centre coords + radius)
        particles = []

        for contour in contours:
            # Find the minimum enclosing circle for each contour
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            # Find the centre
            centre = (int(x), int(y))
            # Find the radius
            radius = int(radius)
            # Store the information
            particles.append((centre, radius))

        # Calculate findContours process duration
        self.circleContours_process_duration = timer_circleContours.stop()
    
        # output to the debug window if enabled
        if self.debug_windows:
            # create a black mask
            mask = np.zeros_like(edges)
            # draw contours white white fill
            cv2.drawContours(mask, contours, -1, (255), cv2.FILLED)
            # display window
            self.contour_window.update(mask)

        # return the segmented particle information
        return particles




    def detect(self, input):
        """ Detects the backscatter particles. Returns the particle coordinates and radius, and the real-time metrics. """

        # reset timings
        self._reset_metrics()

        # single channel conversion using greyscaling
        frame = self._greyscale(input)
        
        # apply histogram equalisation to improve contrasts for better Canny
        frame = self._histequ(frame)

        # apply Gaussian blur noise reduction and smoothening, prep for Canny
        frame = self._gausblur(frame)

        # Log start timestamp of Canny process
        timer_canny = Timer()

        # compute the median single-channel pixel intensities
        gaus_median = np.median(frame)
        # compute threshold values for canny using single parameter Canny
        lower_threshold = int(max(0, (1.0 - self.canny_threshold) * gaus_median))
        upper_threshold = int(min(255, (1.0 + self.canny_threshold) * gaus_median))

        # perform Canny edge detection
        canny = cv2.Canny(
            frame,
            lower_threshold,
            upper_threshold
        )

        # Calculate the Canny process duration
        self.canny_process_duration = timer_canny.stop()

        # Calculate particles
        particles = self._segmentation(canny)

        # Log the total number of particles in this frame
        self.particle_count = len(particles)

        # output to the debug window if enabled
        if self.debug_windows:
            self.canny_window.update(canny)

        # return detected particles
        return particles, self._get_metrics()

