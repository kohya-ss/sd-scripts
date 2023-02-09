import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fairseq'))

from . import data
from . import models
from . import tasks
# from . import criterions
from . import utils

sys.modules["ofa.data"] = data
sys.modules["ofa.models"] = models
sys.modules["ofa.tasks"] = tasks
# sys.modules["ofa.criterions"] = criterions
sys.modules["ofa.utils"] = utils
