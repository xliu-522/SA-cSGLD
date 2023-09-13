import argparse
import json
from src.data import Data
#from unique_names_generator import get_random_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs='?', type=str, default="config/config.json", required=False)
    args = parser.parse_args()
    try:
        with open(args.config, ) as config:
            config = json.load(config)
            #config['model']["random_name"] = get_random_name().replace(" ", "_")
    except:
        print("Error in config")

    print("**** Reading data ****")
    data_obj = Data(config)
    data_obj.read()

    if __name__ == "__main__":
        main()