from __future__ import absolute_import
from collections import namedtuple
from rl.utils.memory import RingBuffer, sample_batch_indexes
import numpy as np
import pickle

# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience',
                        'state0, action, reward, state1, terminal1')

# A batch
# It stores data element-wise, instead of experience-wise
Batch = namedtuple("Batch", ("state0", "action", "reward", "state1",
                             "terminal1"))


class Memory(object):
    """
    Abstract memory class
    """
    def __init__(self, env):
        self.env = env

    def sample(self, batch_size):
        """
        Get a sample from the memory

        :param int batch_size: size of the batch
        :return: A :class:`Batch` object
        """
        raise NotImplementedError()

    def append(self, experience):
        """Add the experience to the memory"""
        raise NotImplementedError()


class SimpleMemory(Memory):
    """
    A simple memory directly storing experiences in a circular buffer

    Data is stored directly as an array of :class:`Experience`"""

    def __init__(self, env, limit):
        super(SimpleMemory, self).__init__(env)
        self.buffer = RingBuffer(limit)

    def get_idxs(self, idxs, batch_size):
        """Get a non-contiguous series of indexes"""
        # Allocate memory
        state0_batch = np.empty((batch_size, self.env.observation_space.dim))
        action_batch = np.empty((batch_size, self.env.action_space.dim))
        reward_batch = np.empty((batch_size, 1))
        terminal1_batch = np.empty((batch_size, 1), dtype=bool)
        state1_batch = np.empty((batch_size, self.env.observation_space.dim))

        for batch_index, memory_index in enumerate(idxs):
            experience = self.buffer[memory_index]
            state0_batch[batch_index, :] = experience.state0
            action_batch[batch_index, :] = experience.action
            reward_batch[batch_index, :] = experience.reward
            terminal1_batch[batch_index, :] = experience.terminal1
            state1_batch[batch_index, :] = experience.state1

        batch = Batch(
            state0=state0_batch,
            action=action_batch,
            reward=reward_batch,
            terminal1=terminal1_batch,
            state1=state1_batch)

        return batch

    def sample(self, batch_size, batch_idxs=None):
        available_samples = len(self)
        if batch_size > available_samples:
            raise(IndexError("Not enough elements in the memory (currently {}) to sample a batch of size {}".format(len(self), batch_size)))
        if batch_idxs is None:
            # Draw random indexes such that we have at least a single entry before each
            # index.
            batch_idxs = sample_batch_indexes(0, available_samples - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1

        return (self.get_idxs(batch_idxs, batch_size=batch_size))

    def append(self, experience):
        self.buffer.append(experience)

    @classmethod
    def from_file(cls, env, limit, file_path):
        """Create a memory from a pickle file"""
        with open(file_path, "rb") as fd:
            memory_database = pickle.load(fd)

        memory = cls(limit=limit, env=env)

        for experience in memory_database:
            memory.append(Experience(*experience))

        return(memory)

    def save(self, file):
        """Dump the memory into a pickle file"""
        print("Saving memory")
        with open(file, "wb") as fd:
            pickle.dump(self.buffer, fd)

    def dump(self):
        """Get the memory content as a single array"""
        return(self.buffer.dump())

    def __len__(self):
        return(len(self.buffer))
