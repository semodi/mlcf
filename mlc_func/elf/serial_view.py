class serial_view():
    def __init__(self):
        pass

    def __len__(self):
        return 1

    def map_sync(self, *args):
        return list(map(*args))

    def map(self, *args):
        return list(map(*args))
