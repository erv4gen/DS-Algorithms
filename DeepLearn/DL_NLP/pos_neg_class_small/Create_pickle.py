from DL_NLP.tf_nltk import create_featureset_and_labels
import  pickle

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_featureset_and_labels("C:\\Users\\15764\\Documents\\Datasets\\ft_bin_class\\pos.txt",
                                                                    "C:\\Users\\15764\\Documents\\Datasets\\ft_bin_class\\neg.txt")
    with open("C:\\Users\\15764\\Documents\\Datasets\\ft_bin_class\\sentement_set.pickle", "wb") as f:
        pickle.dump([train_x, train_y, test_x, test_y ], f)