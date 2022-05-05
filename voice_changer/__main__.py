import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clf", action="store_true")
    args = parser.parse_args()
    if args.clf:
        from .classifier.model import MLPClassifier
        from .classifier.data import prepare
        from .classifier.interactive import record

        data = prepare("./data", "./data/mfcc_data.pkl")
        clf = MLPClassifier.load_from_checkpoint(
            r"lab\classifier\epoch=199-step=360600.ckpt", num_of_classes=len(data)
        )

        print('Start recording...')
        record(clf, data)
