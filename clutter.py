import cv2
import numpy as np
from matplotlib import pyplot as plt

import subprocess
import os

HF = 30 / 1080
CR1 = 500 / 1920
CR2 = 1180 / 1920

L1_SIZE = 256
L2_SIZE = 32

WINDOW = 4  # > 2
ABS_THR = 4


class ClutterDetection:
    def __init__(self, image):
        self.map = None
        self.image = self.trim(self.crop(image))

    @staticmethod
    def trim(im):
        r = int(HF * im.shape[0])
        return im[r:-r]

    @staticmethod
    def crop(im):
        return im[:, int(CR1 * im.shape[1]):int(CR2 * im.shape[1])]

    def calculate_clutter_map(self, debug=False, sm=None):
        orig = self.image.clip(max=150)
        grey = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([grey], [0], None, [L1_SIZE], [0, L1_SIZE])
        hist = hist - hist[150] / 2
        hist[hist < 0] = 0

        if sm is None:
            sm = 0.954 - (hist != 0).argmax() * 0.001834

        v = np.median(grey)
        lv = v * sm

        _, th3 = cv2.threshold(grey, v - lv, v + lv, cv2.THRESH_TOZERO_INV)
        th3 = cv2.resize(th3, (300, 400))

        hist = cv2.calcHist([th3], [0], None, [L2_SIZE], [1, L2_SIZE])

        grad = np.gradient(hist.squeeze())
        g_d = np.gradient(grad)

        s3 = 0
        if grad.max() > 0:
            to_reject = (grad > 200).nonzero()[-1].flatten()

            if len(to_reject) > 0:
                grad[:to_reject[-1]] = 0
                s2 = L2_SIZE - WINDOW / 2

                if debug:
                    print("process: validation pass")
            else:
                s2 = g_d[:max(L2_SIZE - WINDOW, 0)].argmin() - 1

            s3 = grad.argmax()
            if abs(s3 - s2) > ABS_THR:
                s3 = s2

            th3[th3 < s3] = 0

        if debug:
            f, (a, b) = plt.subplots(1, 2)
            a.plot(hist)
            a.plot(grad)
            a.plot(g_d)
            b.imshow(th3, cmap='grey')
            f.suptitle(str(s3))

        imc = cv2.Canny(th3, th3.max() * 2, th3.max() * 2)
        cv2.imwrite('.exp.jpg', imc)

        subprocess.Popen('./magic').wait()
        res = cv2.imread('input.jpg')
        os.remove('.exp.jpg')

        self.map = res
        return res

    def display(self):
        f, (a, b) = plt.subplots(1, 2)
        a.imshow(self.image)
        b.imshow(self.map)


pics = [
    cv2.imread('pics/im.png'),
    cv2.imread('pics/pic_clt_day_open.jpg'),
    cv2.imread('pics/pic_clt_day_closed.jpg'),
    cv2.imread('pics/pic_clt_day_open.jpg'),
    cv2.imread('pics/pic_clt_night_closed.jpg'),
    cv2.imread('pics/pic_clt_night_closed_1.jpg'),
    cv2.imread('pics/pic_free_day_open.jpg')
]

ix = 0
for pic in pics:
    detector = ClutterDetection(pic)
    detector.calculate_clutter_map()
    detector.display()
    plt.savefig(f'runs/test/{ix}.png')
    ix += 1
plt.show()
