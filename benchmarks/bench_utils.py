import argparse


def parse_generate():
    """
    Parses the command line arguments of the benchmark to generates
    """
    # TODO validate arguments
    parser = argparse.ArgumentParser(description="Generate benchmark data")
    parser.add_argument("--location", help="Location for generated data",
                        type=str, required=True)
    parser.add_argument("--benchmark", help="Type of benchmark. i.e. kernel",
                        type=str, required=True)
    parser.add_argument("--subtype", help="Subtype. i.e. two-body", type=str)
    args = parser.parse_args()

    return args


def parse_run():
    """
    Parses the comand line arguments for running a benchmark
    """
    pass
