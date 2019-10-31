import cv2 as cv
import argparse
import os
import requests
from pathlib import Path
import threading
import sys
import common


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='fetch-and-extract')

    parser.add_argument("--video", dest='videofile', help="Video to run detection upon", type=str)
    parser.add_argument("--fetch", dest='frame_index', help="0 based frame", default=0)
    parser.add_argument("--show", dest='show', help="Display frame", type=bool, default=False)
    parser.add_argument("--info", dest='info', help="Run Information", type=bool, default=False)

    return parser.parse_args()


def get_filename(filename):
    # determine the file extension of the current file
    slash = filename.rfind("/")

    return filename[slash + 1:]


# @todo add frame index to be fetched
def fetch_first_frame(fname, base_url=common.__base_url__, download_path=common.__downloads_folder__,
                      overwrite=False):
    pp = Path(fname)
    filename = pp.stem + '.ts'
    url = base_url + os.sep + filename
    image_filename = filename[:-2] + 'jpg'
    download_fqfn = download_path + os.sep + image_filename
    donot_overwrite = not overwrite
    exists = Path(download_fqfn).exists() and donot_overwrite
    if not exists:
        msg = 'downloading ' + url + '...'
        print(msg)
        response = requests.get(url)
        content_type = response.headers['Content-Type'].split('/')[-1]
        with open(download_fqfn, 'wb') as f:
            f.write(response.content)

        cap = cv.VideoCapture(download_fqfn)
        while cap.isOpened():

            ret, frame = cap.read()
            if ret:
                orig_im = frame
                cv.imwrite(download_fqfn, orig_im)
                print('extracting...' + image_filename)
            break
    else:
        msg = 'Using Cached ' + image_filename
        print(msg)

    return (Path(download_fqfn).exists(), download_fqfn)


class fetcherThread(threading.Thread):
    def __init__(self, threadID, filename, base_url=common.__base_url__, download_path=common.__downloads_folder__,
                 overwrite=False):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.base_url = base_url
        self.name = filename
        self.downlood_path = download_path
        self.overwrite = overwrite

    def run(self):
        print("Starting " + self.name)
        fetch_first_frame(self.base_url, self.name, self.downlood_path, self.overwrite)
        print("Exiting " + self.name)


if __name__ == '__main__':
    filename = sys.argv[1]
    ok = fetch_first_frame(filename)
    if not ok[0]: print('File name not recognized')
    else: print(ok[1] + ' Downloaded and Cached here ')


