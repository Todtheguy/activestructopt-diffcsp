from matdeeplearn.common.config.build_config import build_config
from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.preprocessor.processor import process_data

class Runner:
    def __init__(self):
        self.config = None

    def __call__(self, config, args):
        with new_trainer_context(args=args, config=config) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer
            self.task.setup(self.trainer)
            self.task.run()

    def checkpoint(self, *args, **kwargs):
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id

class ConfigSetup:
    def __init__(self, run_mode, config_path):
        self.run_mode = run_mode
        self.config_path = config_path
        self.seed = None
        self.submit = None

class Ensemble:
  def __init__(self, ensembleN, config_path):
    self.config_path = config_path
    self.config = build_config(ConfigSetup('train', self.config_path), 0)
    process_data(self.config["dataset"])
    self.ensemble = [Runner() for _ in range(ensembleN)]
  
  def train(self):
    for runner in self.ensemble:
      runner(self.config, ConfigSetup('train', self.config_path))
      runner.trainer.model.eval()
