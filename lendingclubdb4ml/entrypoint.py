from db4ml import execute
from lendingclubdb4ml.common import Job


class SampleJob(Job):
    def launch(self):
        self.logger.info("Launching sample job")

        execute("lendingclub_train", self.spark, self.config_path)
        execute("lending_scoring_pandas", self.spark, self.config_path)

        self.logger.info("Sample job finished!")


if __name__ == "__main__":
    job = SampleJob()
    job.launch()
