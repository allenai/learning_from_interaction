class Memory:
    """
    The Trainer class requires a Memory, which supplies the trainer with batches of images and targets
    to train the model on. The memory is filled during training from interactions with the iTHOR environment.
    """

    def __init__(self):
        self.prioritized_replay = False

    def add_batch(self, batch: tuple):
        """
        :param batch: a tuple of iterables, such as (batched images, batched targets)
        :return: Nothing

        Possibly preprocesses the batch, and adds it to the memory. If the memory is full, it overwrites the oldest
        entries in the memory.
        """
        raise NotImplementedError

    def initialize_loader(self, batch_size: int):
        """
        :param batch_size
        :return: Nothing

        The memory loader is initialized with the correct batch size. The loader supplies Trainer classes with batches
        to train the model on.
        """
        raise NotImplementedError

    def iterator(self):
        """
        :return: The iterator from which batches are generated, as in:
                        for images, targets in iterator: train_model(images, targets)
        The batch should be returned as a dictionary
        """
        raise NotImplementedError

    def load_memory(self, path):
        """
        :param path: path to pickle file
        :return: Nothing

        Load warm start memory from pickle file
        """
        raise NotImplementedError

    def save_memory(self, path):
        """
        :param path: path to pickle file
        :return: Nothing

        Save current memory to pickle file
        """
        raise NotImplementedError

    def base_image_transform(self, image):
        """
        :param image: The image as it is returned from the Actor class (possibly image / depth pair)
        :return: The image as it would be output by the memory's iterator, minus any data augmentation.

        """
        raise NotImplementedError

    def update_priorities(self, batch_indices: list, priorities: list):
        """
        :param batch_indices: The indices of the images in the mini-batch on which the loss was just evaluated
        :param priorities: The losses correspondig to the individual images of the minibatch
        :return: Nothing. Update the priorities of the images, in case prioritizes replay is used.
        """
        raise NotImplementedError
