import argparse
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def run_model(train_file, test_file):

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)


    train_df = train_df.dropna()


    y_train = train_df['median_house_value']
    X_train = train_df.drop(columns=['median_house_value'])

    if 'median_house_value' in test_df.columns:
        test_df = test_df.drop(columns=['median_house_value'])

    # Modello migliore
    model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=123)
    model.fit(X_train, y_train)

    # Predizioni sul test set
    predictions = model.predict(test_df)

    with open("S5797859.txt", "w") as fh:
        for pred in predictions:
            fh.write(f"{pred}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    args = parser.parse_args()
    run_model(args.train, args.test)