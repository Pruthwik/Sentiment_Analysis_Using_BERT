"""Use NLTK to create train and test split from movie reviews dataset."""
from argparse import ArgumentParser
from nltk.corpus import movie_reviews as mr
from random import shuffle


def write_lines_to_file(lines, file_path):
    """
    Write lines to a file.
    """
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines))


def create_dataset_using_fileids():
    """Create a movie review dataset using file ids."""
    pos_file_ids = mr.fileids('pos')
    neg_file_ids = mr.fileids('neg')
    raw_data_pos = mr.raw(fileids=pos_file_ids)
    raw_sents_pos = raw_data_pos.split('\n')
    raw_data_neg = mr.raw(fileids=neg_file_ids)
    raw_sents_neg = raw_data_neg.split('\n')
    train_data = raw_sents_pos[: 1000] + raw_sents_neg[: 1000]
    test_data = raw_sents_pos[1001: 1051] + raw_sents_neg[1001: 1051]
    train_labels = [1 if i < 1000 else 0 for i in range(2000)]
    test_labels = [1 if i < 50 else 0 for i in range(100)]
    return train_data, train_labels, test_data, test_labels


def select_items_based_on_indexes(items, selected_indexes):
    """Select items based on indexes."""
    return [items[index] for index in selected_indexes]


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser()
    parser.add_argument('--train', dest='tr', help='Enter the train file where the data will be written')
    parser.add_argument('--test', dest='te', help='Enter the test file where the data will be written')
    args = parser.parse_args()
    train_data, train_labels, test_data, test_labels = create_dataset_using_fileids()
    train_data_with_labels = [text + '\t' + str(train_labels[index]) for index, text in enumerate(train_data)]
    test_data_with_labels = [text + '\t' + str(test_labels[index]) for index, text in enumerate(test_data)]
    total_train_samples = len(train_data)
    all_indexes = list(range(total_train_samples))
    shuffle(all_indexes)
    train_data_with_labels = select_items_based_on_indexes(train_data_with_labels, all_indexes)
    write_lines_to_file(train_data_with_labels, args.tr)
    write_lines_to_file(test_data_with_labels, args.te)


if __name__ == '__main__':
    main()
