import argparse
import logging
from datetime import datetime
import json
import os


def     parse_args():
    parser = argparse.ArgumentParser(description='Simulate intraday market trading.')
    parser.add_argument('-exp', '--experiment_name', type=str, default="run")
    parser.add_argument('-tc', '--config_traders_path', type=str, default="data/trader_config.json")
    parser.add_argument('-moc', '--config_market_operator_path', type=str, default="data/market_operator_config.json")
    parser.add_argument('-l', '--log', type=bool, default=False)
    parser.add_argument('-pl', '--show_plots', type=bool, default=False)
    parsed_args = vars(parser.parse_args())

    return parsed_args


def get_exp_path(experiment_name, **kwargs):
    exp_path = "results/%s_%s" % (experiment_name, datetime.now().strftime("%Y_%m_%d_%H%M%S"))
    os.makedirs(exp_path)
    return exp_path


def load_configs(config_traders_path, config_market_operator_path, **kwargs):
    with open(config_traders_path, "rb") as f:
        config_traders = json.load(f)

    with open(config_market_operator_path, "rb") as f:
        config_market_operator = json.load(f)
    return config_traders, config_market_operator


def save_configs(config_traders, config_market_operator, exp_path):
    with open("%s/%s" % (exp_path, "traders.json"), "w") as f:
        json.dump(config_traders, f)

    with open("%s/%s" % (exp_path, "market_operator.json"), "w") as f:
        json.dump(config_market_operator, f)


def set_up_logger(path, args):
    logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler('{0}/logfile.log'.format(path))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.ERROR if not args["log"] else logging.INFO)
