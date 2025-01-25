import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import logging

def visualize_errors(results):
    for model, err in results.items():
        logging.info(f"{model}: Test Error = {err:.4f}")

    plt.bar(results.keys(), results.values(), color=["blue", "green", "red", "purple"])
    plt.ylabel("Test Error")
    plt.title("Comparison of Model Performance")
    plt.show()

def compare_models(N_list, results):
    markers = ["o", "s", "D", "^"]
    for model, errors in results.items():
        plt.plot(N_list, errors, label=model, marker=markers.pop(0))

    plt.xlabel("Number of Training Samples")
    plt.ylabel("Test Error")
    plt.title("Model Comparison by Sample Size")
    plt.legend()
    plt.yscale("log", base=4)
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))
    plt.grid()
    plt.show()
    
def log_formatter(val, _):
    if val < 1:
        return f"{val:.1}"
    elif val.is_integer():
        return f"{int(val)}"
    else:
        return f"{val:.2f}"