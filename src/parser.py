import argparse


def simple_parser():
    parser = argparse.ArgumentParser(
        description="LEARNING AND GENERALIZATION IN OVERPARAMETRIZED NN", epilog="Developped by Ilyess Doragh and Victor Jéséquel"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level",
    )
    parser.add_argument(
        "--num-inputs", 
        default = 1000, 
        help="Numbers of data points")
    parser.add_argument(
        "--hidden-dim1", 
        default = 200, 
        help="Numbers of neurons in the first hidden layer")
    parser.add_argument(
        "--hidden-dim2", 
        default = 100, 
        help="Numbers of neurons in the second hidden layer")
    parser.add_argument(
        "--epochs", 
        default = 1000, 
        help="Numbers of epochs to train the neural network")
    parser.add_argument(
        "--sample-sizes", 
        default = [200, 500, 1000, 2000, 5000, 10000], 
        help="List of sample sizes to compare")
    parser.add_argument(
        "--neuron-sizes", 
        default = [50, 100, 200, 400], 
        help="List of neuron sizes to compare")
    return parser
