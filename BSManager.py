import cv2
import time
import numpy as np

from WindowManager import Window
from TimeManager import Timer

### DEBUG WINDOW NAMES
#   These constants store the names for each GUI window
GRAYSCALE_DEBUG_WINDOW_NAME = "BSDetector Debug: Grayscale"
GAUSBLUR_DEBUG_WINDOW_NAME = "BSDetector Debug: Gaussian Blur"
CANNY_DEBUG_WINDOW_NAME = "BSDetector Debug: Canny Algorithm"
CONTOUR_DEBUG_WINDOW_NAME = "BSDetector Debug: Detected Contours"
HISTEQU_DEBUG_WINDOW_NAME = "BSDetector Debug: Histogram Equalisation"

class Detector:

    def __init__(self, canny_threshold, histogram_equalisation=True, debug_windows=True):
        self.canny_threshold = canny_threshold
        self.histogram_equalisation = histogram_equalisation
        self.debug_windows = debug_windows

        self.grayscale_process_duration = 0
        self.histequ_process_duration = 0
        self.gausblur_process_duration = 0
        self.findContours_process_duration = 0
        self.circleContours_process_duration = 0
        self.preCanny_process_duration = 0
        self.canny_process_duration = 0

        if self.debug_windows:
            self.grayscale_window = Window(GRAYSCALE_DEBUG_WINDOW_NAME)
            self.gausblur_window = Window(GAUSBLUR_DEBUG_WINDOW_NAME)
            self.canny_window = Window(CANNY_DEBUG_WINDOW_NAME)
            self.contour_window = Window(CONTOUR_DEBUG_WINDOW_NAME)
            if self.histogram_equalisation:
                self.histequ_window = Window(HISTEQU_DEBUG_WINDOW_NAME)




    def _grayscale(self, frame):
        # Log start timestamp
        timer = Timer()

        # apply the single-channel conversion with grayscale filter
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate process duration
        self.grayscale_process_duration = timer.stop()

        # output to the debug window if enabled
        if self.debug_windows:
            self.grayscale_window.update(grayscale)

        # return the grayscaled frame
        return grayscale




    def _histequ(self, frame):
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
        # single channel conversion using grayscaling
        frame = self._grayscale(input)
        
        # apply histogram equalisation to improve contrasts for better Canny
        if self.histogram_equalisation:
            frame = self._histequ(frame)

        # apply Gaussian blur noise reduction and smoothening, prep for Canny
        frame = self._gausblur(frame)

        # Log start timestamp of pre-canny processing
        timer_preCanny = Timer()

        # compute the median single-channel pixel intensities
        gaus_median = np.median(frame)
        # compute threshold values for canny using single parameter Canny
        lower_threshold = int(max(0, (1.0 - self.canny_threshold) * gaus_median))
        upper_threshold = int(min(255, (1.0 + self.canny_threshold) * gaus_median))

        # Calculate pre-canny processing duration
        self.preCanny_process_duration = timer_preCanny.stop()

        # Log start timestamp of Canny process
        timer_canny = Timer()

        # perform Canny edge detection
        canny = cv2.Canny(
            frame,
            lower_threshold,
            upper_threshold
        )

        # Calculate the Canny process duration
        self.canny_process_duration = timer_canny.stop()

        particles = self._segmentation(canny)

        # output to the debug window if enabled
        if self.debug_windows:
            self.canny_window.update(canny)

        # return canny and contours
        # return canny, contours
        return canny, particles

