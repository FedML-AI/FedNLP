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


def wait_for_the_training_process():
    pipe_path = "./tmp/fedml"
    if not os.path.exists(os.path.dirname(pipe_path)):
        try:
            os.makedirs(os.path.dirname(pipe_path))
        except OSError as exc:  # Guard against race condition
            print(exc)
    if not os.path.exists(pipe_path):
        open(pipe_path, 'w').close()
    pipe_fd = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(pipe_fd) as pipe:
        while True:
            message = pipe.read()
            if message:
                print("Received: '%s'" % message)
                print("Training is finished. Start the next training with...")
                os.remove(pipe_path)
                return
            sleep(3)
            # print("Daemon is alive. Waiting for the training result.")



# customize the log format
logging.basicConfig(level=logging.INFO,
                    format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

parser = argparse.ArgumentParser()
args = add_args(parser)

os.system("kill $(ps aux | grep \"fedavg_main_se.py\" | grep -v grep | awk '{print $2}')")

# sh run_span_extraction.sh FedAvg "niid_cluster_clients=10_alpha=5.0" 1e-5 0.1 30
# sh run_span_extraction.sh FedProx "niid_cluster_clients=10_alpha=5.0" 1e-5 0.1 30
# sh run_span_extraction.sh FedOPT "niid_cluster_clients=10_alpha=5.0" 1e-5 0.1 30


hps = [
    'FedOPT "uniform_clients=6" 5e-5 1 0.5 6',
    'FedOPT "niid_label_clients=6_alpha=0.01" 5e-5 1 0.5 6',
    'FedOPT "niid_label_clients=6_alpha=0.1" 5e-5 1 0.5 6'
    #'FedOPT "niid_cluster_clients=100_alpha=0.1" 5e-5 1 0.5 6',
    #'FedOPT "niid_cluster_clients=100_alpha=0.001" 5e-5 1 0.5 6'
    #'FedOPT "kmeans_clusters=100" 5e-5 1 0.5 6'
    #'FedOPT "uniform_clients=30" 5e-5 1 0.5 6', 
    #'FedOPT "kmeans_clusters=30" 5e-5 1 0.5 6'
    #'FedOPT "uniform_clients=100" 5e-5 1 0.5 6', 
    #'FedOPT "niid_cluster_clients=100_alpha=0.01" 5e-5 1 0.5 6'
    # 'FedOPT "niid_cluster_clients=30_alpha=0.01" 5e-5 1 0.5 15', 
    #'FedOPT "uniform_clients=30" 5e-5 1 0.5 15', 
    #'FedOPT "niid_cluster_clients=30_alpha=0.1" 5e-5 1 0.5 15', 
    # 'FedOPT "niid_cluster_clients=300_alpha=0.1" 5e-5 1 0.5 30', 
    #'FedAvg "niid_cluster_clients=30_alpha=0.01" 1e-1 1 0.5 15',
    #'FedProx "niid_cluster_clients=30_alpha=0.01" 1e-1 1 0.5 15',
]

run_id = 0
for hp in hps:
    args.hp = hp
    args.run_id = run_id

    logging.info("hp = %s" % args.hp)
    os.system("mkdir /tmp/; touch ./tmp/fedml")
    os.system('nohup sh run_mrqa.sh '
              '{args.hp} '
              '> ./fednlp_se_{args.run_id}.log 2>&1 &'.format(args=args))

    wait_for_the_training_process()

    logging.info("cleaning the training...")
    os.system("kill $(ps aux | grep \"fedavg_main_se.py\" | grep -v grep | awk '{print $2}')")

    sleep(5)
    run_id += 1
