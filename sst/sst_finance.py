import sys
import os.path
import torch
import torch.optim as optim
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField, Field
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, Token
from nltk.tokenize import WhitespaceTokenizer
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
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from typing import Iterable, List, Dict
import pandas as pd
sys.path.append('..')
import utils
import attacks

# Simple LSTM classifier that uses the final hidden state to classify Sentiment. Based on AllenNLP
class LstmClassifier(Model):
    def __init__(self, word_embeddings, encoder, vocab):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}

class FinancialPhraseBankDatasetReader(DatasetReader):
    def __init__(self, tokenizers: Dict[str, Tokenizer], token_indexers: Dict[str, TokenIndexer]):
        super().__init__(lazy=False)
        self._tokenizers = tokenizers
        self._token_indexers = token_indexers

    def _read(self, file_path: str) -> Iterable[Instance]:
        dataset = load_dataset("financial_phrasebank", "sentences_allagree")
        data = dataset['train']
        for row in data:

            # Only keep positive and negative examples (0 is negative and 2 is positive)
            # Convert negative to 0 and positive to 1
            label = row['label']
            if row['label'] == 1:
                continue
            elif row['label'] == 2:
                label = 1

            tokens = self._tokenizers['tokens'].tokenize(row['sentence'])
            label = str(label)
            instance = self.text_to_instance(tokens, label)
            yield instance

    def text_to_instance(self, tokens: List[str], label: str = None) -> Instance:
        tokens = [Token(token) for token in tokens]
        text_field = TextField(tokens, self._token_indexers)
        fields: Dict[str, Field] = {'tokens': text_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

EMBEDDING_TYPE = "w2v" # what type of word embeddings to use

# Trains/Tests LSTM model fine-tuned on finance dataset
def main(dataset_label_filter, test_triggers, model_no_str):
    # Read financial phrasebank dataset from HuggingFace
    # Load the Financial Phrasebank dataset from HuggingFace
    tokenizers = {"tokens": WhitespaceTokenizer()}
    token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
    reader = FinancialPhraseBankDatasetReader(tokenizers=tokenizers, token_indexers=token_indexers)
    data = reader.read("")

     # Split data into train and test and validation sets
     # 80%/10%/10%
    train_data, dev_data = train_test_split(data, test_size=0.2, random_state=116)
    dev_data, test_data = train_test_split(dev_data, test_size=0.5, random_state=116)
    print("Number of training instances:", len(train_data))

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
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding}).cuda()
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim,
                                                    hidden_size=512,
                                                    num_layers=2,
                                                    batch_first=True))
    model = LstmClassifier(word_embeddings, encoder, vocab)
    model.cuda()

    # where to save the model
    model_path = "tmp\\" + EMBEDDING_TYPE + "_finance_" + dataset_label_filter + "_" + model_no_str + "model.th"
    vocab_path = "tmp\\" + EMBEDDING_TYPE + "_finance_" + "vocab"

    # if the model already exists (its been trained), load the pre-trained weights and vocabulary
    if os.path.isfile(model_path):
        vocab = Vocabulary.from_files(vocab_path)
        model = LstmClassifier(word_embeddings, encoder, vocab).cuda()
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
                            num_epochs=5,
                            patience=1,
                            cuda_device=0)
        trainer.train()
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files(vocab_path)
    model.train().cuda() # rnn cannot do backwards in train mode

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
    print(f"dataset_label_filter: {dataset_label_filter}")
    targeted_dev_data = []
    targeted_test_data = []
    for instance in dev_data:
        if instance['label'].label == dataset_label_filter:
            targeted_dev_data.append(instance)
    for instance in test_data:
        if instance['label'].label == dataset_label_filter:
            targeted_test_data.append(instance)

    # Get accuracy for triggers we want to test
    if test_triggers is not None:
        utils.get_accuracy(model, targeted_dev_data, vocab, trigger_token_ids=None)
        utils.get_accuracy(model, targeted_test_data, vocab, trigger_token_ids=test_triggers)
        return

    # get accuracy before adding triggers
    utils.get_accuracy(model, targeted_dev_data, vocab, trigger_token_ids=None)
    model.train() # rnn cannot do backwards in train mode

    negative_words_file = "negative_words.txt"
    positive_words_file = "positive_words.txt"
    # neutral_words_file = "vader_neutral_words.txt"

    # Initialize the set of token ids to filter out from candidates
    token_id_to_filter = {}

    # # Read in negative words and add them to the set
    with open(negative_words_file) as f:
        for line in f:
            token_id_to_filter[vocab.get_token_index(line.strip())] = True

    # Read in positive words and add them to the set
    with open(positive_words_file) as f:
        for line in f:
            token_id_to_filter[vocab.get_token_index(line.strip())] = True

    # Read in neutral words; if a token in the vocab isn't neutral, add to set
    # neutral_set = set()
    # with open(neutral_words_file) as f:
    #     for line in f:
    #         neutral_set.add(vocab.get_token_index(line.strip()))
    # for token_idx in range(vocab.get_vocab_size()):
    #     if token_idx not in neutral_set:
    #         token_id_to_filter[token_idx] = True

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
                                                        increase_loss=True,
                                                        token_id_to_filter=token_id_to_filter)
        # cand_trigger_token_ids = attacks.random_attack(embedding_weight,
        #                                                trigger_token_ids,
        #                                                num_candidates=40)
        # cand_trigger_token_ids = attacks.nearest_neighbor_grad(averaged_grad,
        #                                                        embedding_weight,
        #                                                        trigger_token_ids,
        #                                                        tree,
        #                                                        100,
        #                                                        num_candidates=40,
        #                                                        increase_loss=True)

        # Tries all of the candidates and returns the trigger sequence with highest loss.
        trigger_token_ids = utils.get_best_candidates(model,
                                                      batch,
                                                      trigger_token_ids,
                                                      cand_trigger_token_ids)

    # print accuracy after adding triggers
    utils.get_accuracy(model, targeted_test_data, vocab, trigger_token_ids)

if __name__ == '__main__':
    # Parameters
    dataset_label_filter = "1" # 0 attacks negative, 1 attacks positive
    test_triggers = None # If None, runs the attack as normal; if a list of 3 trigger ids, tests their accuracy on the model
    # model_no_str: determines what file path the model is saved at
    for i in range(1, 6):
        main(dataset_label_filter, test_triggers, str(i))
    # main(dataset_label_filter, test_triggers, "")
