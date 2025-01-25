from src import dataprep, modelling, dataviz
import logging
from tqdm import tqdm

# ----- Expériences principales -----
def main_experiment(N=1000, hidden_dim1=200, hidden_dim2=100, epochs=1000):
    """Main experiment to visualize the performance of different models

    Args:
        N (int, optional): _description_. Defaults to 1000.
        hidden_dim1 (int, optional): _description_. Defaults to 200.
        hidden_dim2 (int, optional): _description_. Defaults to 100.
        epochs (int, optional): _description_. Defaults to 1000.
    """
    x_train, y_train = dataprep.generate_data(N)
    x_test, y_test = dataprep.generate_data(N=200)

    # 2-Layer
    model_two_layer = modelling.TwoLayerNN(input_dim=4, hidden_dim=hidden_dim1)
    logging.info("\nTraining Two-Layer Network...")
    modelling.train_network(model_two_layer, x_train, y_train, epochs=epochs)
    error_two_layer = modelling.evaluate_model(model_two_layer, x_test, y_test)

    # 3-Layer (full)
    model_three_layer = modelling.ThreeLayerNN(
        input_dim=4, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2
    )
    logging.info("\nTraining Three-Layer Network...")
    modelling.train_network(model_three_layer, x_train, y_train, epochs=epochs)
    error_three_layer = modelling.evaluate_model(model_three_layer, x_test, y_test)

    # 3-Layer ( last)
    model_three_layer_last = modelling.ThreeLayerLastOnly(
        input_dim=4, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2
    )
    logging.info("\nTraining Three-Layer (Last Layer Only)...")
    modelling.train_network(model_three_layer_last, x_train, y_train, epochs=epochs)
    error_three_layer_last = modelling.evaluate_model(model_three_layer_last, x_test, y_test)

    # NTK
    # print("\nTraining NTK Approximation Model...")
    # ntk_model = train_ntk_model(model_three_layer, x_train, y_train, epochs=epochs)
    # error_ntk = evaluate_model(
    #     lambda x: ntk_model(model_three_layer(x)), x_test, y_test
    # )
    error_ntk = 0

    results = {
        "Two-Layer NN": error_two_layer,
        "Three-Layer NN": error_three_layer,
        "Three-Layer (Last Layer)": error_three_layer_last,
        "NTK Approximation": error_ntk,
    }

    dataviz.visualize_errors(results)

def compare_models_with_samples(N_list, hidden_dim1=200, hidden_dim2=100, epochs=500):
    """Compare models based on number of samples

    Args:
        N_list (_type_): _description_
        hidden_dim1 (int, optional): _description_. Defaults to 200.
        hidden_dim2 (int, optional): _description_. Defaults to 100.
        epochs (int, optional): _description_. Defaults to 500.

    Returns:
        _type_: _description_
    """
    results = {
        "2-Layer": [],
        "3-Layer": [],
        "3-Layer (Last)": [],
    }

    for N in tqdm(N_list):
        logging.info(f"\nTraining with N = {N}")
        x_train, y_train = dataprep.generate_data(N)
        x_test, y_test = dataprep.generate_data(200)

        # Two-layer model
        model_two_layer = modelling.TwoLayerNN(input_dim=4, hidden_dim=hidden_dim1)
        modelling.train_network(model_two_layer, x_train, y_train, epochs=epochs)
        results["2-Layer"].append(modelling.evaluate_model(model_two_layer, x_test, y_test))

        # 3-layer model
        model_three_layer = modelling.ThreeLayerNN(
            input_dim=4, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2
        )
        modelling.train_network(model_three_layer, x_train, y_train, epochs=epochs)
        results["3-Layer"].append(modelling.evaluate_model(model_three_layer, x_test, y_test))

        # 3-layer last-layer-only model
        model_three_layer_last = modelling.ThreeLayerLastOnly(
            input_dim=4, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2
        )
        modelling.train_network(model_three_layer_last, x_train, y_train, epochs=epochs)
        results["3-Layer (Last)"].append(
            modelling.evaluate_model(model_three_layer_last, x_test, y_test)
        )

    dataviz.compare_models(N_list, results)

def compare_models_with_neurons(neuron_list, N=1000, epochs=500):
    """Compare models based on number of neurons

    Args:
        neuron_list (_type_): _description_
        N (int, optional): _description_. Defaults to 1000.
        epochs (int, optional): _description_. Defaults to 500.

    Returns:
        _type_: _description_
    """
    results = {
        "2-Layer": [],
        "3-Layer": [],
        "3-Layer (Last)": [],
    }

    x_train, y_train = dataprep.generate_data(N)
    x_test, y_test = dataprep.generate_data(200)

    for neurons in tqdm(neuron_list):
        logging.info(f"\nTraining with hidden neurons = {neurons}")

        # Two-layer model
        model_two_layer = modelling.TwoLayerNN(input_dim=4, hidden_dim=neurons)
        modelling.train_network(model_two_layer, x_train, y_train, epochs=epochs)
        results["2-Layer"].append(modelling.evaluate_model(model_two_layer, x_test, y_test))

        # Three-layer model
        model_three_layer = modelling.ThreeLayerNN(
            input_dim=4, hidden_dim1=neurons, hidden_dim2=neurons // 2
        )
        modelling.train_network(model_three_layer, x_train, y_train, epochs=epochs)
        results["3-Layer"].append(modelling.evaluate_model(model_three_layer, x_test, y_test))

        # Three-layer last-layer-only model
        model_three_layer_last = modelling.ThreeLayerLastOnly(
            input_dim=4, hidden_dim1=neurons, hidden_dim2=neurons // 2
        )
        modelling.train_network(model_three_layer_last, x_train, y_train, epochs=epochs)
        results["3-Layer (Last)"].append(
            modelling.evaluate_model(model_three_layer_last, x_test, y_test)
        )

    dataviz.compare_models(neuron_list, results)