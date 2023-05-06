from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def main():
    # load the binary SST dataset.
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer},
                                                    use_subtrees=True)
    train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    vocab = Vocabulary.from_instances(train_data)

    neg_file = open("vader_negative_words.txt", "w+")
    pos_file = open("vader_positive_words.txt", "w+")
    neu_file = open("vader_neutral_words.txt", "w+")
    neg_count = 0
    pos_count = 0
    neu_count = 0

    for token_idx in range(vocab.get_vocab_size()):
        token = vocab.get_token_from_index(token_idx)
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(token)
        compound_sentiment_score = sentiment_dict["compound"]

        if compound_sentiment_score < 0.0:
            neg_file.write(token + "\n")
            neg_count += 1
        elif compound_sentiment_score > 0.0:
            pos_file.write(token + "\n")
            pos_count += 1
        else:
            neu_file.write(token + "\n")
            neu_count += 1
    
    neg_file.close()
    pos_file.close()
    neu_file.close()

    print("Negative words added: ", neg_count)
    print("Positive words added: ", pos_count)
    print("Neutral words added: ", neu_count)
    
main()

