import logging
from pathlib import Path


class SecurityAudit:

    def __init__(self, log_directory="logs"):

        directory = Path(log_directory)
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.logger = logging.getLogger("LocalAI.Security")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:

            handler = logging.FileHandler(
                directory / "security.log",
                encoding="utf-8"
            )

            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(message)s"
                )
            )

            self.logger.addHandler(handler)

    def record(self, event, details):

        self.logger.info(
            "%s | %s",
            event,
            details
        )