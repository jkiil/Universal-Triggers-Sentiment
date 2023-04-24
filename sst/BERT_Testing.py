import torch
from allennlp.data.iterators import BucketIterator
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.training.trainer import Trainer
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.data.iterators import BucketIterator, BasicIterator
import utils
import torch.nn.functional as F
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.common.util import lazy_groups_of


class SentimentAnalysisModel(Model):
    def __init__(self, bert_embedder):
        super().__init__(Vocabulary())
        self.bert_embedder = bert_embedder
        self.classifier = torch.nn.Linear(
            bert_embedder.get_output_dim(), 2)  # binary classification
        self.accuracy = CategoricalAccuracy()

    def forward(self, tokens, label=None):
        bert_output = self.bert_embedder(tokens["tokens"])
        cls_output = bert_output[:, 0, :]
        logits = self.classifier(cls_output)

        output = {"logits": logits}
        if label is not None:
            loss = F.cross_entropy(logits, label)
            output["loss"] = loss
            self.accuracy(logits, label)

        return output

    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}
    

def hotflip_attack(averaged_grad, embedding_matrix, trigger_token_ids,
                   increase_loss=False, num_candidates=1, token_id_to_filter=None):
    """
    The "Hotflip" attack described in Equation (2) of the paper. This code is heavily inspired by
    the nice code of Paul Michel here https://github.com/pmichel31415/translate/blob/paul/
    pytorch_translate/research/adversarial/adversaries/brute_force_adversary.py

    This function takes in the model's average_grad over a batch of examples, the model's
    token embedding matrix, and the current trigger token IDs. It returns the top token
    candidates for each position.

    If increase_loss=True, then the attack reverses the sign of the gradient and tries to increase
    the loss (decrease the model's probability of the true class). For targeted attacks, you want
    to decrease the loss of the target class (increase_loss=False).
    """
    averaged_grad = averaged_grad.cpu()
    embedding_matrix = embedding_matrix.cpu()
    averaged_grad = averaged_grad.unsqueeze(0)
    gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                                                 (averaged_grad, embedding_matrix))
    if not increase_loss:
        # lower versus increase the class probability.
        gradient_dot_embedding_matrix *= -1

    # Set the gradient of filter tokens to be very low so that they are not selected.
    if token_id_to_filter is not None:
        for token_id in token_id_to_filter:
            gradient_dot_embedding_matrix[:, :, token_id] = -1e9

    if num_candidates > 1:  # get top k options
        _, best_k_ids = torch.topk(
            gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().cpu().numpy()[0]
    _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
    return best_at_each_step[0].detach().cpu().numpy()


class GradientCollector:
    def __init__(self):
        self.gradients = []

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def clear_gradients(self):
        self.gradients = []


def add_hooks(model, gradient_collector):
    bert_model = model.bert_embedder.bert_model
    embedding_layer = bert_model.embeddings.word_embeddings
    embedding_layer.register_backward_hook(gradient_collector.save_gradient)


def get_average_grad(model, batch, trigger_token_ids, gradient_collector):
    # Clear gradients from previous iterations
    gradient_collector.clear_gradients()

    batch_size = batch[0]["tokens"]["tokens"].size(0)
    trigger_token_ids_tensor = torch.LongTensor(trigger_token_ids).unsqueeze(0).repeat(batch_size, 1).cuda(
    ) if torch.cuda.is_available() else torch.LongTensor(trigger_token_ids).unsqueeze(0).repeat(batch_size, 1)

    output = model.forward(
        {"tokens": torch.cat((batch[0]["tokens"]["tokens"], trigger_token_ids_tensor), 1)}, batch[0]["label"])

    # Backward pass
    output["loss"].backward()

    averaged_grad = torch.mean(
        torch.stack(gradient_collector.gradients), dim=0)  # Average the gradients

    return averaged_grad

def get_embedding_weight(model):
    """
    Returns the word embedding weight matrix from the model.
    """
    bert_embedder = model.bert_embedder
    embedding_weight = bert_embedder.bert_model.embeddings.word_embeddings.weight
    return embedding_weight    
    
def main():
    model_name = "bert-base-uncased"
    indexer = PretrainedBertIndexer(pretrained_model=model_name)
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class", token_indexers={"tokens": indexer})

    train_data = reader.read(
        'https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    dev_data = reader.read(
        'https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
    test_data = reader.read(
        'https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/test.txt')

    vocab = Vocabulary.from_instances(train_data)

    bert_embedder = PretrainedBertEmbedder(pretrained_model=model_name)
    model = SentimentAnalysisModel(bert_embedder)

    iterator = BucketIterator(batch_size=8, sorting_keys=[
                              ("tokens", "num_tokens")])
    iterator.index_with(vocab)

    # optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    # trainer = Trainer(model=model,
    #                   optimizer=optimizer,
    #                   iterator=iterator,
    #                   train_dataset=train_data,
    #                   validation_dataset=dev_data,
    #                   num_epochs=3,
    #                   cuda_device=0)  # Change to 0 if using GPU

    # trainer.train()

    # Save the model
    # torch.save(model.state_dict(), "sst_bert_model.pt")

    # Load the trained model for attack
    model.load_state_dict(torch.load("sst_bert_model.pt"))
    model.eval()


        # Register a gradient hook on the embeddings. This saves the gradient w.r.t. the word embeddings.
        # We use the gradient later in the attack.
        # Register a gradient hook on the embeddings. This saves the gradient w.r.t. the word embeddings.
    # We use the gradient later in the attack.
    gradient_collector = GradientCollector()
    add_hooks(model, gradient_collector)
    embedding_weight = get_embedding_weight(model)  # also save the word embedding matrix

    # Use batches of size universal_perturb_batch_size for the attacks.
    universal_perturb_batch_size = 128
    iterator = BasicIterator(batch_size=universal_perturb_batch_size)
    iterator.index_with(vocab)

    # filter the dataset to only positive or negative examples
    # (the trigger will cause the opposite prediction)
    dataset_label_filter = "1"
    targeted_dev_data = []
    for instance in dev_data:
        if instance['label'].label == dataset_label_filter:
            targeted_dev_data.append(instance)

    # get accuracy before adding triggers
    utils.get_accuracy(model, targeted_dev_data, vocab, trigger_token_ids=None)
    model.train()

    negative_words_file = "sst/negative_words.txt"
    positive_words_file = "sst/positive_words.txt"

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
        model.train()  # rnn cannot do backwards in train mode

        # get gradient w.r.t. trigger embeddings for current batch
        averaged_grad = get_average_grad(
            model, batch, trigger_token_ids, gradient_collector)


        # pass the gradients to a particular attack to generate token candidates for each token.
        cand_trigger_token_ids = hotflip_attack(averaged_grad,
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
    utils.get_accuracy(model, test_data, vocab, trigger_token_ids)

if __name__ == "__main__":
    main()

