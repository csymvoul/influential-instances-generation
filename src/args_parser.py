import argparse

class ArgsParser:
    def parse_args():
        parser=argparse.ArgumentParser()

        parser.add_argument("--data", help="The dataset used for training of the model.")
        parser.add_argument("--model", help="The model which will be used for training.")

        args=parser.parse_args()