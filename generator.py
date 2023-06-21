import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
# # In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# # This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# # This input consists of a batch of images and its corresponding labels.

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        # self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
        #                    7: 'horse', 8: 'ship', 9: 'truck'}

        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size   #[height,width,channel] ex: ch=3 -> RGB image
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.image_list = []
        self.batch = []
        self.epochs = 0
        self.epochs_list = []

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        #loading image files
        files = sorted(glob.glob(os.path.join(self.file_path, "*.npy")),
                       key=lambda x: int(os.path.basename(x).split(".")[0]))
        self.image_list = [np.load(f) for f in files]

        # loading labels
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)

        self.n_sample = np.arange(len(self.image_list))

    def next(self):
        np.random.seed(0)

        # resize option
        self.image_list = [np.resize(img, self.image_size) if img.shape != self.image_size else img for img in
                           self.image_list]

        # shuffle
        if self.shuffle:
            randomize = np.arange(len(self.image_list))
            np.random.shuffle(randomize)
            self.image_list = np.array(self.image_list)[randomize]
            self.labels = {str(i): self.labels[str(randomize[i])] for i in range(len(randomize))}

        if len(self.n_sample) >= self.batch_size:
            t = np.random.choice(self.n_sample, size=self.batch_size, replace=False)
            self.batch.append(t)
            self.n_sample = np.setdiff1d(self.n_sample, t)

        elif len(self.n_sample) == 0:
            self.epochs += 1
            self.batch = []
            self.n_sample = np.arange(len(self.image_list))
            t = np.random.choice(self.n_sample, size=self.batch_size, replace=False)
            self.batch.append(t)
            self.n_sample = np.setdiff1d(self.n_sample, t)
            self.epochs_list.append([self.batch])
        else:
            t = np.concatenate((self.n_sample, self.batch[0][:((self.batch_size) - len(self.n_sample))]))
            self.batch.append(t)
            self.epochs += 1
            self.epochs_list.append([self.batch])
            self.batch = []
            self.n_sample = np.arange(len(self.image_list))

        images = []
        labels = []

        for i in range(self.batch_size):
            im = t[i]

            # mirror function
            if self.mirroring:
                mirror_function = np.random.choice(['lr', 'ud', 'No mirroring'], size=1)
                if mirror_function != 'No mirroring':
                    if mirror_function == 'lr':
                        self.image_list[im] = np.fliplr(self.image_list[im])
                    elif mirror_function == 'ud':
                        self.image_list[im] = np.flipud(self.image_list[im])

            # rotation function
            if self.rotation:
                a = np.random.choice(['0', '90', '180', '270'])
                if a != '0':
                    if a == '90':
                        self.image_list[im] = np.rot90(self.image_list[im])
                    elif a == '180':
                        self.image_list[im] = np.rot90(self.image_list[im], 2)
                    elif a == '270':
                        self.image_list[im] = np.rot90(self.image_list[im], 3)

            images.append(self.image_list[im])
            labels.append(self.labels[str(im)])

        im_arrays = np.array(images)
        lab_arrays = np.array(labels)

        tuple_array = tuple((im_arrays, lab_arrays))
        return tuple_array

    def current_epoch(self):
        return self.epochs

    def class_name(self, x):
        return self.class_dict.get((x))


    def show(self):
        images, labels = self.next()
        fig = plt.figure()
        for i, (image, title) in enumerate(zip(images, labels)):
            fig.add_subplot(3, int(np.ceil(len(images) / float(3))), i + 1)
            plt.title(self.class_name(labels[i]))
            plt.imshow(image)
        plt.show()