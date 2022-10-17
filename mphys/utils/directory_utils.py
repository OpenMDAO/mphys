import os

class cd:
    def __init__(self, new_path: str):
        """
        Context manager for changing the current working directory.
        If an empty string is provided, the directory will not change.
        """
        if not new_path:
            new_path = os.getcwd()
        self.new_path = os.path.expanduser(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)
