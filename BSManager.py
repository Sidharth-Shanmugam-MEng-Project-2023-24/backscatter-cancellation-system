import cv2
import numpy as np

from WindowManager import Window

### DEBUG WINDOW NAMES
#   These constants store the names for each GUI window
GRAYSCALE_DEBUG_WINDOW_NAME = "BSDetector Debug: Grayscale"
GAUSBLUR_DEBUG_WINDOW_NAME = "BSDetector Debug: Gaussian Blur"
CANNY_DEBUG_WINDOW_NAME = "BSDetector Debug: Canny Algorithm"
COUNTOUR_DEBUG_WINDOW_NAME = "BSDetector Debug: Detected Countours"
HISTEQU_DEBUG_WINDOW_NAME = "BSDetector Debug: Histogram Equalisation"

class Detector:

    def __init__(self, canny_threshold, histogram_equalisation=True, debug_windows=True):
        self.canny_threshold = canny_threshold
        self.histogram_equalisation = histogram_equalisation
        self.debug_windows = debug_windows

        if self.debug_windows:
            self.grayscale_window = Window(GRAYSCALE_DEBUG_WINDOW_NAME)
            self.gausblur_window = Window(GAUSBLUR_DEBUG_WINDOW_NAME)
            self.canny_window = Window(CANNY_DEBUG_WINDOW_NAME)
            # self.countour_window = Window(COUNTOUR_DEBUG_WINDOW_NAME)
            if self.histogram_equalisation:
                self.histequ_window = Window(HISTEQU_DEBUG_WINDOW_NAME)




    def _grayscale(self, frame):
        # apply the single-channel conversion with grayscale filter
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # output to the debug window if enabled
        if self.debug_windows:
            self.grayscale_window.update(grayscale)

        # return the grayscaled frame
        return grayscale




    def _histequ(self, frame):
        # apply histogram equalisation
        histequ = cv2.equalizeHist(frame)

        # output to the debug window if enabled
        if self.debug_windows:
            self.histequ_window.update(histequ)
        
        # return the histogram equalised frame
        return histequ




    def _gausblur(self, frame):
        # apply the Gaussian blur
        gausblur = cv2.GaussianBlur(frame, (5,5), 0)

        # output to the debug window if enabled
        if self.debug_windows:
            self.gausblur_window.update(gausblur)
        
        # return the Gaussian blurred frame
        return gausblur
    



    def detect(self, input):
        # single channel conversion using grayscaling
        frame = self._grayscale(input)
        
        # apply histogram equalisation to improve contrasts for better Canny
        if self.histogram_equalisation:
            frame = self._histequ(frame)

        # apply Gaussian blur noise reduction and smoothening, prep for Canny
        frame = self._gausblur(frame)

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

        # contours = self._findContours(canny)

        # output to the debug window if enabled
        if self.debug_windows:
            self.canny_window.update(canny)

        # return canny and contours
        # return canny, contours
        return canny