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
    "e",
    "e,0",
    "e,0,1",
    "e,0,1,2",
    "e,0,1,2,3",
    "e,0,1,2,3,4",
    "e,0,1,2,3,4,5"
]

run_id = 0
for hp in hps:
    args.hp = hp
    args.run_id = run_id

    logging.info("hp = %s" % args.hp)
    os.system("mkdir ./tmp/; touch ./tmp/fedml")
    os.system('nohup sh run_text_classification_sweep.sh '
              '{args.hp} '
              '> ./fednlp_tc_freeze_{args.hp}.log 2>&1'.format(args=args))

    logging.info("cleaning the training...")
    os.system("kill $(ps aux | grep \"main_tc.py\" | grep -v grep | awk '{print $2}')")

    sleep(5)
    run_id += 1
