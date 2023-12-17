import torch
from transformers import AutoModel
from transformers import AutoTokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class MultiLabelSequenceClassification(torch.nn.Module):

    def __init__(self, num_labels=2):
        super(MultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.the_model = AutoModel.from_pretrained('bert-base-uncased')
        self.classifier = torch.nn.Linear(768, num_labels)

        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None,\
                attention_mask=None, labels=None):
        # last hidden layer
        last_hidden_state = self.the_model(input_ids=input_ids,\
                                    attention_mask=attention_mask,\
                                    token_type_ids=token_type_ids)
        # pool the outputs into a mean vector
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),\
                            labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def freeze_the_decoder(self):
        """
        Freeze weight parameters. They will not be updated during training.
        """
        for param in self.the_model.parameters():
            param.requires_grad = False

    def unfreeze_the_decoder(self):
        """
        Unfreeze weight parameters. They will be updated during training.
        """
        for param in self.the_model.parameters():
            param.requires_grad = True

    def pool_hidden_state(self, last_hidden_state):
        """
        Pool the output vectors into a single mean vector
        """
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

def load_model(model_path, device= 'cpu'):

    model = MultiLabelSequenceClassification(num_labels=6)

    state_dict = torch.load(model_path, map_location=device)['state_dict']
    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()
    return model
saved_model_path = "model/bert.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = load_model(saved_model_path, device)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def processing_text(text, num_embeddings=512):
    tokenized_texts = tokenizer.tokenize(text)[:num_embeddings-5]
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_texts)
    input_ids = [tokenizer.build_inputs_with_special_tokens(input_ids)]
    input_ids = pad_sequences(input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")[0]

    attention_masks = [float(i>0) for i in input_ids]

    return [input_ids], [attention_masks]

def generate_predictions(text, num_labels= 6, device="cpu", num_embeddings=512):
    input_ids, attention_masks = processing_text(text, num_embeddings)
    input_ids = torch.tensor(input_ids).to(device)
    attention_masks = torch.tensor(attention_masks).to(device)

    with torch.no_grad():
        logits = loaded_model(input_ids=input_ids, attention_mask=attention_masks)
        logits = logits.sigmoid().detach().cpu().numpy()
        # pred_probs = np.vstack([pred_probs, logits])
    return logits[0]

def runModel(text, num_embeddings):
        predict = generate_predictions(text, num_embeddings)
        predict = np.round(predict,decimals=2)
        return predict.tolist()

