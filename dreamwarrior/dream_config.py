import os
import configparser

class DreamConfig:
    def __init__(self, configfile=None):
        if configfile is None:
            dirname = os.path.dirname(__file__)
            configfile = os.path.join(dirname, 'dream.ini')

        config = configparser.ConfigParser()
        config.read(configfile)

        # Set environment parameters
        environment = config['environment']
        self.device = environment['device']
        self.frame_skip = environment.getint('frame_skip')

        # Set model components
        model = config['model']
        self.double = model.getboolean('double')
        self.dueling = model.getboolean('dueling')
        self.prioritized = model.getboolean('prioritized')
        self.noisy = model.getboolean('noisy')

        # Set training parameters
        training = config['training']
        self.gamma = training.getfloat('gamma')
        self.frame_limit = int(training.getfloat('frame_limit'))
        self.frame_update = int(training.getfloat('frame_update'))
        self.learning_rate = training.getfloat('learning_rate')
        self.epsilon_start = training.getfloat('epsilon_start')
        self.epsilon_end = training.getfloat('epsilon_end')
        self.epsilon_decay = training.getfloat('epsilon_decay')

        # Set memory parameters
        memory = config['memory']
        self.batch_size = memory.getint('batch_size')
        self.capacity = int(memory.getfloat('capacity'))
        self.alpha = memory.getfloat('alpha')
        self.beta_start = memory.getfloat('beta_start')
        self.beta_frames = int(memory.getfloat('beta_frames'))
        