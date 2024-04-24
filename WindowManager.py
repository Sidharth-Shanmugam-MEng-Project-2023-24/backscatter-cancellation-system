import matplotlib.pyplot as plt
import numpy as np
import cv2

class Window:
    """ A GUI window manager utilising OpenCV. """

    def __init__(self, name, frame=None, plot=False):
        self.name = name
        self.plot = plot

        # created OpenCV named window
        cv2.namedWindow(self.name)

        # initialise window with input frame or test img

        if not frame:
            # create black image 800x600
            test = np.zeros((600, 800, 3), dtype=np.uint8)

            # Generate a rainbow gradient along the height (600)
            for i in range(600):
                hue = int(180 * i / 600)  # Vary the hue from 0 to 180
                colour = list(
                    map(
                        int,
                        cv2.cvtColor(
                            np.array(
                                [[[hue,255,255]]],
                                dtype=np.uint8
                            ),
                            cv2.COLOR_HSV2BGR
                        )[0, 0]
                    )
                )
                test[i, :, :] = colour

            # display rainbow test image
            if plot:
                plt.imshow(test)
                plt.show() 
            else:
                cv2.imshow(self.name, test)
        else:
            if plot:
                plt.imshow(frame)
                plt.show() 
            else:
                cv2.imshow(self.name, frame)




    def update(self, frame):
        """ Updates the window with the given frame. """
        if self.plot:
            plt.imshow(frame)
            plt.show() 
        else:
            cv2.imshow(self.name, frame)




    def destroy(self):
        """ Gracefully terminates the window instance. """
        cv2.destroyWindow(self.name)

