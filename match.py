import cv2 as cv
import numpy as np
import unittest
from matplotlib import pyplot as plt



def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def LabImage(bgr_image):
    h, w, channels = bgr_image.shape
    lab_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2LAB)
    # Split LAB channels
    L, a, b = cv.split(lab_image)
    return (L, a, b)


def JointHistogram(a, b):
    #    print(str(np.min(a)) + '  ' + str(np.average(a)) + '  ' + str(np.max(a)))
    #    print(str(np.min(b)) + '  ' + str(np.average(b)) + '  ' + str(np.max(b)))

    xedges, yedges = np.linspace(np.min(a), np.max(a), 128), np.linspace(np.min(b), np.max(b), 128)
    hist, xedges, yedges = np.histogram2d(a.flatten(), b.flatten(), (xedges, yedges))
    xidx = np.clip(np.digitize(a.flatten(), xedges), 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(b.flatten(), yedges), 0, hist.shape[1] - 1)
    c = hist[xidx, yidx]
    return (a.flatten(), b.flatten(), c)


def MutualInformation(a, b, plotter=None):
    hist_2d, x_edges, y_edges = np.histogram2d(a.ravel(), b.ravel(), bins=200)
    mu = mutual_information(hist_2d)
    if plotter != None:
        hist_2d_log = np.zeros(hist_2d.shape)
        non_zeros = hist_2d != 0
        hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
        plotter.imshow(hist_2d_log.T, origin='lower')
    return (mu, hist_2d)


def find_template(fixed,  template):
    h, w = template.shape[:2]


    res = cv.matchTemplate(fixed, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    tl = max_loc
    br = (tl[0] + w, tl[1] + h)
    return (res, max_val, tl, br)


def compare_mse(image_a, image_b):
    diff = image_a - image_b
    diff = diff * diff
    r = np.sum(diff)
    r /= float(image_a.shape[0] * image_a.shape[1])
    return (1.0 - r)


def correlate(cvma, cvmb, mask=None):
    ha, wa = cvma.shape[:2]
    hb, wb = cvmb.shape[:2]

    if ha != hb or wa != wb:
        print("Same Size Required")
        return 0.0

    mean_a, stddev_a = cv.meanStdDev(cvma, mask=mask)
    mean_b, stddev_b = cv.meanStdDev(cvmb, mask=mask)
    n_pixels = hb * wa

    cvmaa = cvma - mean_a
    cvmbb = cvmb - mean_b
    covar1 = np.sum(cvmaa * cvmbb) / n_pixels
    correlation = covar1 / (stddev_a * stddev_b)

    return correlation


def make_gauss(centre, amp, sig, shapdim):
    shape = (shapdim[1],shapdim[0],1)
    sidelenx = shapdim[1]
    sideleny = shapdim[0]
    l = np.zeros(shape, dtype="float32")

    for i in range(0, sideleny):
        for j in range(0, sidelenx):
            l[i, j] = np.sqrt((centre[0] - i) ** 2 + (centre[1] - j) ** 2)
    gaussblob = amp * np.exp(-(l ** 2) / sig)
    return gaussblob


def mi_lum(imga, imgb):
    f, axs = plt.subplots(2, 3, figsize=(20, 10), frameon=False,
                          subplot_kw={'xticks': [], 'yticks': []})
    axs[0, 0].imshow(imga)
    axs[0, 1].imshow(imgb)

    (L, a, b) = LabImage(imga)
    (LL, aa, bb) = LabImage(imgb)

    (x, y, c) = JointHistogram(L, LL)
    axs[1, 0].scatter(x, y, marker='.', cmap='PuBu_r')
    axs[1, 0].set_title(' Luminance ')
    axs[1, 0].set_xlabel('b')
    axs[1, 0].set_ylabel('a')



    plt.autoscale
    plt.show()


__all__ = ['TestCorrelate']

winName = "Match Test"
cv.namedWindow(winName, cv.WINDOW_NORMAL)


class TestCorrelate(unittest.TestCase):

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

    def test_correlate(self):
        self.test_basic_corr()
        self.test_basic()

    def test_basic_corr(self):
        g1 = make_gauss([80, 80], 3.14, 0.25, [160, 160])
        g2 = make_gauss([80, 80], 3.14, 0.25 * 4, [160, 160])

        eps = 0.0000001

        res = correlate(g1, g1)

        self.assertTrue((1.0 - res[0]) < eps)

        res = correlate(g1, g2)

        self.assertTrue(np.fabs(0.80738218 - res[0]) < eps)

    def test_basic(self):
        g1 = make_gauss([80, 80], 3.14, 0.5, [160, 160])
        g2 = make_gauss([7, 7], 3.14, 0.5, [16, 16])
        roi = [70, 90, 70, 90]
        res = find_template(g1,  g2)
        self.assertTrue(res[2] == (83, 64))
        self.assertTrue(res[3] == (99, 80))
        self.assertAlmostEqual(res[1], 1.0, 4)


if __name__ == '__main__':
    unittest.main()
