import autopy
import numpy as np
import cv2
import time


class App(object):
    window_name = 'capture'

    def __init__(self):

        screen_width, screen_height = autopy.screen.get_size()
        self.cap = cv2.VideoCapture(0)

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cam_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.cam_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.LowH = 0
        self.HighH = 255

        self.LowS = 0
        self.HighS = 255

        self.LowV = 0
        self.HighV = 255

        self.frame = None
        self.roi_corners = [[0, self.cam_width],[0, self.cam_height]]
        self.pick_color = False
        self.show_mask = False

        self.show_image = True

        self.initialize_windows()

    def transform_position(self, x, y):
        x2 = (float(x) / self.cam_width) * self.screen_width
        y2 = (float(y) / self.cam_height) * self.screen_height
        return int(x2), int(y2)

    def get_object_position(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        thresholded = cv2.inRange(
            hsv_img,
            (self.LowH, self.LowS, self.LowV),
            (self.HighH, self.HighS, self.HighV)
        )

        kernel = np.ones((5, 5), np.uint8)
        thresholded = cv2.erode(thresholded, kernel)
        thresholded = cv2.dilate(thresholded, kernel)
        thresholded = cv2.dilate(thresholded, kernel)
        thresholded = cv2.erode(thresholded, kernel)
        thresholded = self.best_contour(thresholded)
        M = cv2.moments(thresholded)

        if M['m00'] == 0:  # if region is empty
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        if self.show_mask:
            self.frame = thresholded

        return (cx, cy)

    def initialize_windows(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.click_and_crop)

    def iLowH(self, value):
        self.LowH = value

    def iHighH(self, value):
        self.HighH = value

    def iLowS(self, value):
        self.LowS = value

    def iHighS(self, value):
        self.HighS = value

    def iLowV(self, value):
        self.LowV = value

    def iHighV(self, value):
        self.HighV = value

    def run(self):
        lx = 0
        ly = 0
        while True:
            # Capture frame-by-frame
            ret, self.frame = self.cap.read()
            self.frame = self.equalize_hist_3d(self.frame)
            self.frame = cv2.flip(self.frame, 1)

            k = cv2.waitKey(1)
            if k & 0xFF == ord('s'):
                cv2.imwrite('cap.png', self.frame)
            if k & 0xFF == ord('m'):
                self.show_mask = True
            if k & 0xFF == ord('h'):
                self.show_image = not self.show_image
            if k & 0xFF == ord('p'):
                cv2.imshow(self.window_name, self.frame)
                self.select_roi()
                while self.pick_color:
                    cv2.waitKey(1000)
                cv2.imshow(self.window_name, self.frame)
                k = cv2.waitKey(0)

            p = self.get_object_position(self.frame)
            if p is not None:
                x, y = p
            else:
                x, y = (lx, ly)
            cv2.circle(self.frame, (x,y), 5, (255, 255, 255))

            # Display the resulting frame
            if self.show_image:
                cv2.imshow(self.window_name, self.frame)
            else:
                fx, fy = self.transform_position(x, y)
                autopy.mouse.smooth_move(fx, fy)
            if k & 0xFF == ord('q'):
                break
            lx = x
            ly = y

        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()

    def click_and_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.pick_color:
            if len(self.roi_corners[0]) == 2:
                self.roi_corners = [[], []]
            if len(self.roi_corners[0]) == 1:
                self.pick_color = False
            self.roi_corners[0].append(x)
            self.roi_corners[1].append(y)
            self.roi_corners[0] = sorted(self.roi_corners[0])
            self.roi_corners[1] = sorted(self.roi_corners[1])

            if len(self.roi_corners[0]) == 2:
                fx = (self.roi_corners[0][1] + self.roi_corners[0][0]) / 2
                fy = (self.roi_corners[1][1] + self.roi_corners[1][0]) / 2
                cv2.circle(self.frame, (fx,fy), 5, (255, 255, 255))
                hsv_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                print(hsv_img[fy][fx])
                h = hsv_img[fy][fx][0]
                s = hsv_img[fy][fx][1]
                v = hsv_img[fy][fx][2]
                tolerance = 25
                self.LowH = h - tolerance
                self.HighH = h + tolerance
                self.LowS = s - tolerance
                self.HighS = s + tolerance
                # self.LowV = v - tolerance
                # self.HighV = v + tolerance

    def select_roi(self):
        self.pick_color = True

    def best_contour(self, mask):
        image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        biggest = None
        biggest_count = -1
        for i in range(len(contours)):
            blank = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
            region_mask = cv2.drawContours(blank, contours, i, 255, -1)
            s = cv2.sumElems(region_mask / 255)[0]
            if s > biggest_count:
                biggest = region_mask
                biggest_count = s

        return biggest if biggest_count > 90 else np.zeros((image.shape[0], image.shape[1], 1), np.uint8)

    def equalize_hist_3d(self, img):
        i2 = img
        max_val = img.max()
        min_val = img.min()
        factor = 255.0 / (max_val - min_val)
        eq_img = (np.floor((i2 - min_val) * factor)).astype(np.uint64)
        return cv2.convertScaleAbs(eq_img)
App().run()
