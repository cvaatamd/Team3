from datasets import load_dataset

ds_train = load_dataset("openadmet/openadmet-expansionrx-challenge-train-data")
ds_train["train"].to_csv("openadmet_train.csv")

ds_test =load_dataset("openadmet/openadmet-expansionrx-challenge-test-data-blinded")
ds["test"].to_csv("openadmet_test.csv")

