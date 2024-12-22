import signal
import logging
import os

logger = logging.getLogger(__name__)


class SignalHandler:
    def __init__(self):
        self.reset_sample()
        self.reset_save()
        self.register_handlers()

    def register_handlers(self):
        logger.info(
            f'Installing signal handler. Send SIGUSR1 to sample, SIGUSR2 to save model. Current PID: {os.getpid()}')
        signal.signal(signal.SIGUSR1, self.handle_sigusr1)
        signal.signal(signal.SIGUSR2, self.handle_sigusr2)

    def handle_sigusr1(self, signum, frame):
        logger.info("SIGUSR1 received - will force sampling at next step")
        self.force_sample = True

    def handle_sigusr2(self, signum, frame):
        logger.info("SIGUSR2 received - will force save at next step")
        self.force_save = True

    def should_sample(self):
        return self.force_sample

    def should_save(self):
        return self.force_save

    def reset_sample(self):
        self.force_sample = False

    def reset_save(self):
        self.force_save = False
