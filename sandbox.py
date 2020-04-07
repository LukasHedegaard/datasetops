def main():

    def func(s):
        return s

    zeros, others = do.load_mnist().split_filter(func)
    all([(s.lbl == 0) for s in zeros])
