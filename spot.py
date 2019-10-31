import time
from enum import Enum
import unittest
from pathlib import Path


class pState(Enum):
    Empty = 3
    NewCar = 1
    SameCar = 2
    UnInitialized = 0


class SpotObserver(object):
    def __init__(self, state=pState.UnInitialized, first=0, last=0):
        self._state = state
        self._first = first
        self._last = last
        self._state_names = {}
        self._state_names[pState.Empty] = 'Empty'
        self._state_names[pState.NewCar] = 'NewCar'
        self._state_names[pState.SameCar] = 'SameCar'
        self._state_names[pState.UnInitialized] = 'UnInitialized'
        self._records = []
        self._outputs = []

    def state(self):
        return self._state

    def first(self):
        return self._first

    def last(self):
        return self._last

    def records(self):
        return self._records

    def reportAllCTime(self):
        for r in self.records():
            output = time.ctime(r[0]) + '     ' + time.ctime(r[1]) + '       ' + str(r[2])
            print(output)

    def reportAll(self):
        for r in self.records():
            output = str(r[0]) + '     ' + str(r[1]) + '       ' + str(r[2])
            print(output)

    def report(self):
        first_str = time.ctime(self._first)
        last_str = time.ctime(self._last)
        return first_str + ' to ' + last_str + '  ' + self._state_names[self._state]

    def record(self):
        self._records.append((self._first, self._last, self._state))

    def update(self, falseIsEmpty, timestamp):

        if self._state == pState.UnInitialized:
            self._last = timestamp
            if not falseIsEmpty:
                self._state = pState.Empty
            else:
                self._state = pState.NewCar
        elif self._state == pState.NewCar:
            if not falseIsEmpty:  # transition to empty
                self.record()
                self._first = timestamp
                #                self._last = self._first
                self._state = pState.Empty
            else:  # staying put. Just update our ts
                self._state = pState.NewCar
                self._last = timestamp
        elif self._state == pState.SameCar:
            if not falseIsEmpty:  # transition to empty
                self.record()
                self._first = timestamp
                #                self._last = self._first
                self._state = pState.Empty
            else:  # staying put. Just update our ts
                self._state = pState.SameCar
                self._last = timestamp
        elif self._state == pState.Empty:
            if falseIsEmpty:  # transition to New Car
                self.record()
                self._first = timestamp
                self._last = self._first
                self._state = pState.NewCar
            else:
                self._last = timestamp

    def finalize(self):
        self.record()


__all__ = ['TestStatus']


class TestStatus(unittest.TestCase):

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

    def test_ctor(self):
        pp = SpotObserver()
        self.assertTrue(pp.state() == pState.UnInitialized)
        self.assertEqual(len(pp.records()), 0)

    def test_simple(self):
        pp = SpotObserver()
        self.assertTrue(pp.state() == pState.UnInitialized)
        self.assertEqual(len(pp.records()), 0)
        pp.update(True, 1538076003)
        self.assertTrue(pp.state() == pState.NewCar)
        self.assertEqual(0, pp.first())
        self.assertEqual(len(pp.records()), 0)
        pp.update(False, 1538076003 + 3600)
        self.assertEqual(len(pp.records()), 1)
        print(str(pp))
        f, l, s = pp.records()[0]
        self.assertEqual(0, f)
        self.assertEqual(l, 1538076003)
        self.assertEqual(s, pState.NewCar)

    def test_more(self):
        pp = SpotObserver(pState.UnInitialized, 1538076003)
        self.assertTrue(pp.state() == pState.UnInitialized)
        self.assertEqual(len(pp.records()), 0)
        result_url = 'results/_0.txt'
        self.assertTrue(Path(result_url).exists())
        fo = open(result_url, "r+")
        lines = fo.readlines()
        for line in lines:
            tokens = line.split(' ')
            ts = int(tokens[0])
            empty = not line.find('Not') >= 0
            pp.update(empty, ts)
        pp.finalize()
        print(len(pp.records()))
        fo.close()
        pp.report()



if __name__ == '__main__':
    unittest.main()
