import logging
import time
from src.logger import init_logger
from src.parser import simple_parser
from src import experiment

def main():
    # Parsing argument
    parser = (
        simple_parser()
    )  # you can adjust the function in parser.py file in order to fit your need
    args = parser.parse_args()

    # init logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")

    # Paramètres
    N = args.num_inputs
    hidden_dim1 = args.hidden_dim1
    hidden_dim2 = args.hidden_dim2
    epochs = args.epochs

    logging.info("Starting main experiment...")
    start_time = time.time()
    experiment.main_experiment(N, hidden_dim1, hidden_dim2, epochs)
    logging.info(f"Main experiment done in {time.time()-start_time:.2f} seconds.")

    logging.info("Starting sample experiment...")
    start_time = time.time()
    N_list = args.sample_sizes
    experiment.compare_models_with_samples(
    N_list=N_list, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, epochs=epochs
    )
    logging.info(f"Sample experiment done in {time.time()-start_time:.2f} seconds.")

    logging.info("Starting number of neurons experiment...")
    start_time = time.time()
    neuron_list = args.neuron_sizes
    experiment.compare_models_with_neurons(neuron_list=neuron_list, N=N, epochs=epochs)
    logging.info(f"number of neurons experiment done in {time.time()-start_time:.2f} seconds.")



if __name__ == "__main__":
    main()