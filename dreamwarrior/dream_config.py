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
        self.height = environment.getint('height')
        self.device = environment['device']
        self.frame_skip = environment.getint('frame_skip')

        # Set model components
        model = config['model']
        self.double = model.getboolean('double')
        self.dueling = model.getboolean('dueling')
        self.prioritized = model.getboolean('prioritized')
        self.noisy = model.getboolean('noisy')
        self.categorical = model.getboolean('categorical')

        if self.categorical and not self.double:
            raise ValueError('Dream Warrior does not support training non-double DQNs')

        # Set training parameters
        training = config['training']
        self.gamma = training.getfloat('gamma')
        self.min_frames = int(training.getfloat('min_frames'))
        self.frame_limit = int(training.getfloat('frame_limit'))
        self.frame_update = int(training.getfloat('frame_update'))
        self.episode_frame_max = int(training.getfloat('episode_frame_max'))
        self.learning_rate = training.getfloat('learning_rate')
        self.adam_epsilon = training.getfloat('adam_epsilon')
        self.epsilon_start = training.getfloat('epsilon_start')
        self.epsilon_end = training.getfloat('epsilon_end')
        self.epsilon_decay = training.getfloat('epsilon_decay')
        self.atoms = training.getint('atoms')
        self.v_min = training.getint('v_min')
        self.v_max = training.getint('v_max')

        # Set memory parameters
        memory = config['memory']
        self.batch_size = memory.getint('batch_size')
        self.capacity = int(memory.getfloat('capacity'))
        self.alpha = memory.getfloat('alpha')
        self.beta_start = memory.getfloat('beta_start')
        self.beta_frames = int(memory.getfloat('beta_frames'))
        self.multi_step = memory.getint('multi_step')
        