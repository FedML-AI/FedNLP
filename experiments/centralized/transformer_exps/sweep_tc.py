import argparse
import logging
import os
from time import sleep


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    return parser.parse_args()


# customize the log format
logging.basicConfig(level=logging.INFO,
                    format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

parser = argparse.ArgumentParser()
args = add_args(parser)

os.system("kill $(ps aux | grep \"main_tc.py\" | grep -v grep | awk '{print $2}')")

hps = [
    # "6"
    # "6 e",
    # "6 e,0",
    # "6 e,0,1",
    "7 e,0,1,2",
    "7 e,0,1,2,3",
    "7 e,0,1,2,3,4",
    "7 e,0,1,2,3,4,5"
]

run_id = 0
for hp in hps[::-1]:
    args.hp = hp
    args.run_id = run_id
    layers = hp.split(" ")[-1]
    logging.info("hp = %s" % args.hp)
    os.system("mkdir ./tmp/; touch ./tmp/fedml")
    os.system('nohup sh run_text_classification_sweep.sh '
              '%s '
              '> ./fednlp_tc_freeze_%s.log 2>&1' % (hp, layers))

    logging.info("cleaning the training...")
    os.system("kill $(ps aux | grep \"main_tc.py\" | grep -v grep | awk '{print $2}')")

    sleep(5)
    run_id += 1
