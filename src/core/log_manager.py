"""
Logging Manager class
"""
import os
import logging.config
from pathlib import Path


class LogManagerBase:
    """
    Base logging manager class
    """
    folder = Path(os.getcwd()) / 'configuration'
    logging.config.fileConfig(str(folder / "logging.conf"))
    logger = logging.getLogger("root")
    LogFullKeys = []

    @classmethod
    def info(cls, data):
        cls.log_info(data=data, case_id="~")

    @classmethod
    def debug(cls, data):
        cls.log_debug(data=data, case_id="~")

    @classmethod
    def error(cls, data):
        cls.log_error(data=data, case_id="~")

    @classmethod
    def log_info(cls, data, indent=0, case_id="~"):
        if isinstance(data, dict):
            cls.log_dict(data, indent, "info", case_id)
        elif isinstance(data, list):
            cls.log_list(data, indent, "info", case_id)
        else:
            getattr(cls.logger, "info")("{} - {}".format(case_id, data))

    @classmethod
    def log_debug(cls, data, indent=0, case_id="~"):
        if isinstance(data, dict):
            cls.log_dict(data, indent, "debug", case_id)
        elif isinstance(data, list):
            cls.log_list(data, indent, "debug", case_id)
        else:
            getattr(cls.logger, "debug")("{} - {}".format(case_id, data))

    @classmethod
    def log_error(cls, data, indent=0, case_id="~"):
        if isinstance(data, dict):
            cls.log_dict(data, indent, "error", case_id)
        elif isinstance(data, list):
            cls.log_list(data, indent, "error", case_id)
        else:
            getattr(cls.logger, "error")("{} - {}".format(case_id, data))

    @classmethod
    def log_dict(cls, data, indent, level, case_id):
        for key, value in data.items():
            if key in cls.LogFullKeys and value != "":
                getattr(cls.logger, level)("{} - {}- {} : FULL".format(case_id, " " * indent, key))

            elif isinstance(value, dict):
                getattr(cls.logger, level)("{} - {}- {}:".format(case_id, " " * indent, key))
                cls.log_dict(value, indent + 3, level, case_id)

            elif isinstance(value, list):
                getattr(cls.logger, level)("{} - {}- {}:".format(case_id, " " * indent, key))
                cls.log_list(value, indent + 3, level, case_id)

            else:
                getattr(cls.logger, level)("{} - {}- {} : {}".format(
                    case_id, " " * indent, key, value))

    @classmethod
    def log_list(cls, data, indent, level, case_id):
        index = -1
        for element in data:
            if isinstance(element, dict):
                index += 1
                getattr(cls.logger, level)("{} - {}* [{}]".format(
                    case_id, " " * indent, str(index)))
                cls.log_dict(element, indent + 3, level, case_id)

            elif isinstance(element, list):
                index += 1
                getattr(cls.logger, level)("{} - {}* [{}]".format(
                    case_id, " " * indent, str(index)))
                cls.log_list(element, indent + 3, level, case_id)

            else:
                getattr(cls.logger, level)("{} - {}* {} ".format(case_id, " " * indent, element))


    @classmethod
    def log_registry(cls, registry):
        cls.info("Registry Content:")
        printable_registry = {}
        for key in registry.keys():
            printable_registry[key] = {}
            for inner_key in registry[key].keys():
                printable_registry[key][inner_key] = registry[key][inner_key]["class_origin"].value
        cls.info(printable_registry)


class LogManager(LogManagerBase):
    """
    Log Manager class inherited from LogManager
    """
    def __init__(self, case_id):
        self.case_id = case_id

    def info(self, data):
        self.log_info(data=data, case_id=self.case_id)

    def debug(self, data):
        self.log_debug(data=data, case_id=self.case_id)

    def error(self, data):
        self.log_error(data=data, case_id=self.case_id)
