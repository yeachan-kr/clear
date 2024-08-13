import os
import sys
import argparse
import time
import datetime
from utils import set_random_seed

# Custom Library
from utils import set_random_seed, load_noise_dataset, load_noisy_ner_dataset, load_real_noise_dataset, Logger, get_num_classes


def run(args):
    # Create directories if not exist.
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    now = datetime.datetime.now()
    # log_file_name = '{}-nr:{}-alg:{}-model:{}-bs:{}-ep:{}-lr:{}.txt'.format(args.dataset.split('/')[-1], args.noise_rate, args.alg, args.model, args.batch_size, args.epochs, args.lr)
    log_file_name = '{}_{}'.format(args.alg, now)
    print('Create LOG file {}'.format(log_file_name))
    sys.stdout = Logger(location=os.path.join(args.logdir, log_file_name))
    print(args)

    if not os.path.exists(args.modeldir):
        os.makedirs(args.modeldir)

    # Data partitioning based on non-iid strategy
    num_class = get_num_classes(args.dataset)
    noise_dataset = load_noise_dataset(dataset=args.dataset, model=args.model, datadir=args.datadir,noise_rate=args.noise_rate, num_class=num_class, noise_type=args.noise_type)

    # Select Solver based on learning strategy
    solver = None

    if args.alg == 'routing_lora':
        from solvers.routing_lora_solver import RoutingLoRASolver
        solver = RoutingLoRASolver(args=args, dataset=noise_dataset)
            
    if args.alg == 'routing_adapter':
        from solvers.routing_adapter_solver import RoutingAdapterSolver
        solver = RoutingAdapterSolver(args=args, dataset=noise_dataset)
    
    if args.alg == 'routing_prefix':
        from solvers.routing_prefix_solver import RoutingPrefixSolver
        solver = RoutingPrefixSolver(args=args, dataset=noise_dataset)
    
    if args.alg == 'routing_bitfit':
        from solvers.routing_bitfit_solver import RoutingBitFitSolver
        solver = RoutingBitFitSolver(args=args, dataset=noise_dataset)
    
    solver.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='SetFit/sst5', help='dataset used for training')
    parser.add_argument('--noise_rate', type=float, default=0.2, help='')
    parser.add_argument('--noise_type', type=str, default='sym', help='')

    # Training configuration
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate (default: 5e-5)')
    parser.add_argument('--epochs', type=int, default=20, help='number of local epochs')
    parser.add_argument('--warm_up', type=int, default=3, help='')

    # Model configuration
    parser.add_argument('--alg', type=str, default='full', help='PEFT routing strategy. Choose PEFT routing from : routing_adapter, routing_lora, routing_prefix, routing_bitfit, None')
    parser.add_argument('--model', type=str, default='bert-base-uncased', help='neural network used in training (please refer to huggingface\'s model list')
    parser.add_argument('--adapter', type=str, default='none', help='PEFT routing strategy. Choose PEFT routing from : routing_adapter, routing_lora, routing_prefix, routing_bitfit, None')
    parser.add_argument('--rank', type=int, default=16, help='')
    parser.add_argument('--r_prob', type=float, default=0.7, help='')

    # Directory configuration conda activate torch37
    parser.add_argument('--datadir', type=str, required=False, default="./data", help="Data directory")
    parser.add_argument('--logdir', type=str, default='./log/', help='dataset used for training')
    parser.add_argument('--modeldir', type=str, default='./save/', help='dataset used for training')

    # Computation configuration
    parser.add_argument('--device', type=str, default='3', help='The device to run the program')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")

    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    set_random_seed(args.init_seed)

    # Start solver
    run(args)

