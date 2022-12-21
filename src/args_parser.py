import argparse

class ArgsParser:
    """
    `ArgsParser` class.

    This class is used for parsing the arguments passed to the program.
    """
    @staticmethod
    def parse_args():
        """
        parse_args function

        Description:
            This function parses the arguments passed to the program.

        Args:
            `None`

        Returns:
            `argparse.Namespace`: The parsed arguments.
        """
        parser=argparse.ArgumentParser()

        parser.add_argument("--data", help="The dataset used for training of the model.")
        parser.add_argument("--model", help="The model which will be used for training.")

        return parser.parse_args()