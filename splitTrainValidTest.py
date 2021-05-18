import shutil
import os
from os.path import basename, splitext
import argparse


class TrainValidTestSplit(object):
    def __init__(self, path, percent_valid=10, percent_test=5):
        self.percent_valid = percent_valid
        self.percent_test = percent_test
        self.path = path
        self.valid = []
        self.test = []
        self.images = []
        self.annotations = []
        self.len_dataset = len(os.listdir(self.path))

    def getDataset(self):

        print(f'Length of dataset is {self.len_dataset}')

        # get the whole dataset
        for i in os.listdir(self.path):
            name1, ext1 = splitext(basename(i))
            if ext1 == '.jpg':
                self.images.append(i)
        for j in os.listdir(self.path):
            name2, ext2 = splitext(basename(j))
            if ext2 == '.txt':
                self.annotations.append(j)
        totalLength = len(self.images) + len(self.annotations)
        print(f'Length of the received dataset of images is {len(self.images)}')
        print(f'Length of the received dataset of annotations is {len(self.annotations)}')
        print(f'Total Length Received is {len(self.images) + len(self.annotations)}')
        if totalLength == self.len_dataset:
            print(f'All data received')
        else:
            raise SystemExit('Some data has been lost')

    def getValid(self):

        # Obtaining validating images
        l = len(self.images)
        valid_part = round(l * (self.percent_valid / 100))
        p = round(l / valid_part)
        for i in range(valid_part):
            n = p * i
            if n > len(self.images):
                break
            else:
                self.valid.append(self.images[n])
        for i, item1 in enumerate(self.images):
            for j, item2 in enumerate(self.valid):
                if item1 == item2:
                    self.images.pop(i)

        # Obtaining testing annotations
        for i in range(valid_part):
            n = p * i
            if n > len(self.annotations):
                break
            else:
                self.valid.append(self.annotations[n])
        for i, item1 in enumerate(self.annotations):
            for j, item2 in enumerate(self.valid):
                if item1 == item2:
                    self.annotations.pop(i)
        print(f'Length of valid dataset is {len(self.valid)}')
        if len(self.images) != len(self.annotations):
            raise SystemExit('Some data has been lost')

    def getTest(self):

        # Obtaining testing images
        l = len(self.images)
        test_part = round(l * (self.percent_test / 100))
        p = round(l / test_part)
        for i in range(test_part):
            n = p * i
            if n > len(self.images):
                break
            else:
                self.test.append(self.images[n])
        for i, item1 in enumerate(self.images):
            for j, item2 in enumerate(self.test):
                if item1 == item2:
                    self.images.pop(i)

        # Obtaining validating annotations
        for i in range(test_part):
            n = p * i
            if n > len(self.annotations):
                break
            else:
                self.test.append(self.annotations[n])
        for i, item1 in enumerate(self.annotations):
            for j, item2 in enumerate(self.test):
                if item1 == item2:
                    self.annotations.pop(i)
        print(f'Length of test dataset is {len(self.test)}')
        if len(self.images) != len(self.annotations):
            raise SystemExit('Some data has been lost')

    def createTrainValidTest(self, path_train, path_valid, path_test):

        # Create train
        for i in self.images:
            shutil.copyfile(f'{self.path}\{i}', f'{path_train}\{i}')
        for i in self.annotations:
            shutil.copyfile(f'{self.path}\{i}', f'{path_train}\{i}')
        print(f'Train was created in quantity {len(os.listdir(path_train))}')

        # Create valid
        for i in self.valid:
            shutil.copyfile(f'{self.path}\{i}', f'{path_valid}\{i}')
        print(f'Valid was created in quantity {len(os.listdir(path_valid))}')

        # Create test
        for i in self.test:
            shutil.copyfile(f'{self.path}\{i}', f'{path_test}\{i}')
        print(f'Test was created in quantity {len(os.listdir(path_test))}')

        if (len(os.listdir(path_train)) + len(os.listdir(path_valid)) + len(os.listdir(path_test))) == self.len_dataset:
            print('Data has not been lost')
        else:
            raise SystemExit('Some data has been lost')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Сreate directories containing train, valid, test')
    parser.add_argument('dataset',
                        type=str,
                        help='enter path to the dataset with images and annotations')
    parser.add_argument('train_path',
                        type=str,
                        help='enter path to the train data')
    parser.add_argument('valid_path',
                        type=str,
                        help='enter path to the valid data')
    parser.add_argument('test_path',
                        type=str,
                        help='enter path to the test data')
    args = parser.parse_args()
    A = TrainValidTestSplit(args.dataset)
    A.getDataset()
    A.getValid()
    A.getTest()
    A.createTrainValidTest(args.train_path, args.valid_path, args.test_path)
