import argparse
import os
import requests
from fetchandextract import fetch_first_frame
from hascar import process, compare
from pathlib import Path
from spot import SpotObserver, pState
from common import __downloads_folder__, __base_url__


class analyzecars:

    def __init__(self, base_url, index_url, first_index, last_index, compare_process=False):
        self.base_url = base_url
        self.index_url = index_url
        self.cwd = os.getcwd()
        if Path(index_url).exists():
            fo = open(index_url, "r+")
            self.listing = fo.readlines()

        if len(self.listing) == 0:
            r = requests.get(self.base_url + '.txt')
            decoded_content = r.content.decode('utf-8')
            self.listing = decoded_content.splitlines()

        # Create list of video files we have to process
        self.first = int(first_index)
        self.last = int(last_index)
        self.nlisting = list(map(lambda x: int(x.rstrip()[:-3]), self.listing))
        self.nbatch = [x for x in self.nlisting if x >= self.first and x <= self.last]
        self.batch = list(map(lambda x: str(x) + '.ts', self.nbatch))
        self.compare_process = compare_process

        # Instantiate a spot observer
        self.pp = SpotObserver()

        dwp = Path(__downloads_folder__)
        if not dwp.exists():
            dwp.mkdir()

        car = 'car'
        last_out = []
        for idx, file in enumerate(self.batch):
            down_info = fetch_first_frame(file)
            if down_info[0]:
                cur_out = process(down_info[1], False)

                ## If the very first,
                if idx == 0:
                    self.pp.update(cur_out[0], self.nbatch[idx])
                    last_out = cur_out
                    continue

                # Verify / Validate / Improve using the last detection
                compare_result = compare(cur_out, last_out)
                if compare_result:
                    # If images are similar and detection results were identical:
                    # that is both empty or both with car, good to go
                    if cur_out[0] == last_out[0]:
                        self.pp.update(cur_out[0], self.nbatch[idx])
                    else:
                        # if images are similar but we detection result are different
                        # accept the earlier one
                        self.pp.update(last_out[0], self.nbatch[idx])
                else:
                    # If images are dissimilar
                    if cur_out[0] != last_out[0]:
                        # and detection results were also different
                        # accept the new
                        self.pp.update(last_out[0], self.nbatch[idx])
                    else:
                        # detection results were the same but images are dis-similar
                        # go for the old one
                        self.pp.update(last_out[0], self.nbatch[idx])

                last_out = cur_out


            # print(self.pp.report())

        self.pp.reportAll()


def main():
    parser = argparse.ArgumentParser(description='analyze-cars')
    parser.add_argument('--index', '-i', required=True, help='Index File ')
    parser.add_argument('--start', '-s', required=True, help='Starting time stamp')
    parser.add_argument('--end', '-e', required=True, help='Last time stamp')

    args = parser.parse_args()

    place = analyzecars(__base_url__, args.index, args.start, args.end)


if __name__ == '__main__':
    main()

# def fetch_frames(index_array, start_index, end_index, sampling=None):
