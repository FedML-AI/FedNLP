import argparse
import logging
import os
from time import sleep


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # PipeTransformer related
    parser.add_argument("--run_id", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e5)

    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def wait_for_the_training_process():
    pipe_path = "/tmp/pipe_transformer_training_status"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(pipe_fd) as pipe:
        while True:
            message = pipe.read()
            if message:
                print("Received: '%s'" % message)
                print("Training is finished. Start the next training with...")
                return
            sleep(3)
            print("Daemon is alive. Waiting for the training result.")


# customize the log format
logging.basicConfig(level=logging.INFO,
                    format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

parser = argparse.ArgumentParser()
args = add_args(parser)

lr = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
batch_size = [16]
hpo_list = []

os.system("kill $(ps aux | grep \"main_tc.py\" | grep -v grep | awk '{print $2}')")

run_id = 0
for lr_idx in range(len(lr)):
    for bs_idx in range(len(batch_size)):
        current_lr, current_bs = lr[lr_idx], batch_size[bs_idx]
        args.lr = current_lr
        args.batch_size = current_bs
        args.port = 10000 + run_id
        args.run_id = run_id
        args.b_freeze = "no_freeze"

        """
        NPROC_PER_NODE=$1
        NNODE=$2
        NODE_RANK=$3
        MASTER_ADDR=$4
        MASTER_PORT=$5
        IB=$6
        IF_NAME=$7
        LR=$8
        BZ=$9
        RUN_ID=${10}
        B_FREEZE=${11}
        PIPE_LEN=${12}
        """
        logging.info("current_lr = %f, current_bs = %d" % (current_lr, current_bs))
        os.system('nohup sh run_tc_pipetransformer.sh 4 1 0 192.168.1.73 {args.port} 0 '
                  '"wlx9cefd5fb3821" {args.lr} {args.batch_size} {args.run_id} {args.b_freeze} 4 > '
                  './PipeTransformer-TC_run{args.run_id}.log 2>&1 &'.format(args=args))

        wait_for_the_training_process()

        logging.info("cleaning the training...")
        os.system("kill $(ps aux | grep \"main_tc.py\" | grep -v grep | awk '{print $2}')")

        sleep(10)
        run_id += 1
