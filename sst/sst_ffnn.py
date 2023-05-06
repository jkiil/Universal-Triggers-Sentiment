import sys
import os.path
from sklearn.neighbors import KDTree
import torch
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer
sys.path.append('..')
import utils
import attacks

# Simple FFNN classifier that uses the final hidden state to classify Sentiment. Based on AllenNLP
class FFNNClassifier(Model):
    def __init__(self, word_embeddings, word_embedding_dim, d_hidden, vocab):
        """
        Initialize a two-layer feedforward neural network with sigmoid activation.
        """
        super().__init__(vocab)
        self.embed = word_embeddings
        self.hidden = torch.nn.Linear(in_features=word_embedding_dim, out_features=d_hidden, bias=True)
        self.output = torch.nn.Linear(in_features=d_hidden, out_features=vocab.get_vocab_size('labels'), bias=True)
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label) -> torch.Tensor:
        embeddings = self.embed(tokens)
        mean_embeddings = torch.mean(embeddings, dim=1)
        hidden = self.hidden(mean_embeddings)
        logits = self.output(torch.sigmoid(hidden))
        output = {"logits": logits}

        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output
    
    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}

EMBEDDING_TYPE = "w2v" # what type of word embeddings to use

def main(dataset_label_filter, test_triggers, model_no_str):

    # load the binary SST dataset.
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer},
                                                    use_subtrees=True)
    train_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer})
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
    test_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/test.txt')

    vocab = Vocabulary.from_instances(train_data)

    # Randomly initialize vectors
    if EMBEDDING_TYPE == "None":
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=300)
        word_embedding_dim = 300

    # Load word2vec vectors
    elif EMBEDDING_TYPE == "w2v":
        embedding_path = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
        weight = _read_pretrained_embeddings_file(embedding_path,
                                                  embedding_dim=300,
                                                  vocab=vocab,
                                                  namespace="tokens")
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=300,
                                    weight=weight,
                                    trainable=False)
        word_embedding_dim = 300

    # Initialize model, cuda(), and optimizer
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    model = FFNNClassifier(word_embeddings, word_embedding_dim, 256, vocab)
    # model.cuda()

    # where to save the model
    model_path = "tmp/" + EMBEDDING_TYPE + "_FFNN_" + dataset_label_filter + "_" + model_no_str + "model.th"
    vocab_path = "tmp/" + EMBEDDING_TYPE + "_vocab"

    # if the model already exists (its been trained), load the pre-trained weights and vocabulary
    if os.path.isfile(model_path):
        vocab = Vocabulary.from_files(vocab_path)
        model = FFNNClassifier(word_embeddings, word_embedding_dim, 256, vocab)
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f))
    # otherwise train model from scratch and save its weights
    else:
        iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
        iterator.index_with(vocab)
        optimizer = optim.Adam(model.parameters())
        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_data,
                          validation_dataset=dev_data,
                          num_epochs=30,
                          patience=1)
                          #cuda_device=0)
        trainer.train()
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files(vocab_path)
    model.train()
    # model.train().cuda() # rnn cannot do backwards in train mode

    # Register a gradient hook on the embeddings. This saves the gradient w.r.t. the word embeddings.
    # We use the gradient later in the attack.
    utils.add_hooks(model)
    embedding_weight = utils.get_embedding_weight(model) # also save the word embedding matrix

    # Use batches of size universal_perturb_batch_size for the attacks.
    universal_perturb_batch_size = 128
    iterator = BasicIterator(batch_size=universal_perturb_batch_size)
    iterator.index_with(vocab)

    # Build k-d Tree if you are using gradient + nearest neighbor attack
    # tree = KDTree(embedding_weight.numpy())

    # filter the dataset to only positive or negative examples
    # (the trigger will cause the opposite prediction)
    # 0: negative, 1: positive
    targeted_dev_data = []
    for instance in dev_data:
        if instance['label'].label == dataset_label_filter:
            targeted_dev_data.append(instance)

    targeted_test_data = []
    for instance in test_data:
        if instance['label'].label == dataset_label_filter:
            targeted_test_data.append(instance)

    # Get accuracy for triggers we want to test
    if test_triggers is not None:
        utils.get_accuracy(model, targeted_test_data, vocab, trigger_token_ids=None)
        utils.get_accuracy(model, targeted_test_data, vocab, trigger_token_ids=test_triggers)
        return

    # get accuracy before adding triggers
    utils.get_accuracy(model, targeted_dev_data, vocab, trigger_token_ids=None)
    model.train() # rnn cannot do backwards in train mode

    negative_words_file = "negative_words.txt"
    positive_words_file = "positive_words.txt"

    # Initialize the set of token ids to filter out from candidates
    token_id_to_filter = {}

    # Read in negative words and add them to the set
    with open(negative_words_file) as f:
        for line in f:
            token_id_to_filter[vocab.get_token_index(line.strip())] = True

    # Read in positive words and add them to the set
    with open(positive_words_file) as f:
        for line in f:
            token_id_to_filter[vocab.get_token_index(line.strip())] = True

    # initialize triggers which are concatenated to the input
    num_trigger_tokens = 3
    trigger_token_ids = [vocab.get_token_index("the")] * num_trigger_tokens

    # sample batches, update the triggers, and repeat
    for batch in lazy_groups_of(iterator(targeted_dev_data, num_epochs=5, shuffle=True), group_size=1):
        # get accuracy with current triggers
        utils.get_accuracy(model, targeted_dev_data, vocab, trigger_token_ids)
        model.train() # rnn cannot do backwards in train mode

        # get gradient w.r.t. trigger embeddings for current batch
        averaged_grad = utils.get_average_grad(model, batch, trigger_token_ids)

        # pass the gradients to a particular attack to generate token candidates for each token.
        cand_trigger_token_ids = attacks.hotflip_attack(averaged_grad,
                                                        embedding_weight,
                                                        trigger_token_ids,
                                                        num_candidates=40,
                                                        increase_loss=True, token_id_to_filter=token_id_to_filter)

        # Tries all of the candidates and returns the trigger sequence with highest loss.
        trigger_token_ids = utils.get_best_candidates(model,
                                                      batch,
                                                      trigger_token_ids,
                                                      cand_trigger_token_ids)

    # print accuracy after adding triggers
    utils.get_accuracy(model, targeted_dev_data, vocab, trigger_token_ids)

if __name__ == '__main__':
    dataset_label_filter = "0" # "0" for negative attack or "1" for positive attack
    test_triggers = None # None to not test and [id1, id2, id3] to test triggers
    # model_no_str: determines what file path the model is saved at
    # for i in range(1, 6):
    #     main(dataset_label_filter, test_triggers, str(i))
    main(dataset_label_filter, test_triggers, "")