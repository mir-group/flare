import yaml, sys
from flare.bffs.sgp.calculator import SGP_Calculator


def build_map(config):
    flare_file = config.get("file")
    lmp_name = config.get("lmp_name")
    contributor = config.get("contributor")
    sgp_calc, _ = SGP_Calculator.from_file(flare_file)
    sgp_calc.gp_model.write_mapping_coefficients(lmp_name, contributor, 0)


def rebuild_gp(config):
    flare_config = config["flare_calc"]

    # get the SGP settings from file
    flare_from_file, _ = SGP_Calculator.from_file(flare_config.get("file"), build=False)

    flare_from_file


def main():
    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    mode = config.get("otf").get("mode", "fresh")
    if mode == "fresh":
        fresh_start_otf(config)
    elif mode == "restart":
        restart_otf(config)
