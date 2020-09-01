"""
DOCSTRING
"""
import class_register.register_api
import csv
import numpy
import pytorch_pretrained_bert
import time
import torch

class AbstractLanguageChecker:
    """
    Abstract Class that defines the Backend API of GLTR.

    To extend the GLTR interface, you need to inherit this
    and fill in the defined functions.
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def check_probabilities(self, in_text, topk=40):
        """
        Function that GLTR interacts with to check the probabilities of words.

        Arguments:
            - in_text: str -- The text that you want to check
            - topk: int -- Your desired truncation of the head of the distribution

        Returns:
            - payload: dict -- The wrapper for results in this function, described below

        Payload values:
            bpe_strings: list of str -- Each individual token in the text
            real_topk: list of tuples -- (ranking, prob) of each token
            pred_topk: list of list of tuple -- (word, prob) for all topk
        """
        raise NotImplementedError

    def postprocess(self, token):
        """
        clean up the tokens from any special chars and encode
        leading space by UTF-8 code '\u0120', linebreak with UTF-8 code 266 '\u010A'
        :param token:  str -- raw token text
        :return: str -- cleaned and re-encoded token text
        """
        raise NotImplementedError

@class_register.register_api(name='BERT')
class BERTLM(AbstractLanguageChecker):
    """
    DOCSTRING
    """
    def __init__(self, model_name_or_path="bert-base-cased"):
        super(BERTLM, self).__init__()
        self.device = torch.device('cpu')
        self.tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(
            model_name_or_path, do_lower_case=False)
        self.model = pytorch_pretrained_bert.BertForMaskedLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.mask_tok = self.tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        self.pad = self.tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
        print("Loaded BERT model.")

    def check_probabilities(self, in_text, topk=40, max_context=20, batch_size=20):
        """
        Same behavior as GPT-2.

        Extra param: max_context controls how many words should be fed in left and right.
            Speeds up inference since BERT requires prediction word by word.
        """
        in_text = '[CLS] {} [SEP]'.format(in_text)
        tokenized_text = self.tokenizer.tokenize(in_text)
        y_toks = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(y_toks)
        y = torch.tensor([y_toks]).to(self.device)
        segments_tensor = torch.tensor([segments_ids]).to(self.device)
        input_batches, target_batches = list(), list()
        for min_ix in range(0, len(y_toks), batch_size):
            max_ix = min(min_ix + batch_size, len(y_toks) - 1)
            cur_input_batch, cur_target_batch = list(), list()
            for running_ix in range(max_ix - min_ix):
                tokens_tensor = y.clone()
                mask_index = min_ix + running_ix
                tokens_tensor[0, mask_index + 1] = self.mask_tok
                min_index = max(0, mask_index - max_context)
                max_index = min(tokens_tensor.shape[1] - 1, mask_index + max_context + 1)
                tokens_tensor = tokens_tensor[:, min_index:max_index]
                needed_padding = max_context * 2 + 1 - tokens_tensor.shape[1]
                if min_index == 0 and max_index == y.shape[1] - 1:
                    left_needed = max_context - mask_index
                    right_needed = needed_padding - left_needed
                    p = torch.nn.ConstantPad1d((left_needed, right_needed), self.pad)
                    tokens_tensor = p(tokens_tensor)
                elif min_index == 0:
                    p = torch.nn.ConstantPad1d((needed_padding, 0), self.pad)
                    tokens_tensor = p(tokens_tensor)
                elif max_index == y.shape[1] - 1:
                    p = torch.nn.ConstantPad1d((0, needed_padding), self.pad)
                    tokens_tensor = p(tokens_tensor)
                cur_input_batch.append(tokens_tensor)
                cur_target_batch.append(y[:, mask_index + 1])
            cur_input_batch = torch.cat(cur_input_batch, dim=0)
            cur_target_batch = torch.cat(cur_target_batch, dim=0)
            input_batches.append(cur_input_batch)
            target_batches.append(cur_target_batch)
        real_topk, pred_topk = list(), list()
        with torch.no_grad():
            for src, tgt in zip(input_batches, target_batches):
                logits = self.model(
                    src, torch.zeros_like(src))[:, max_context+1]
                yhat = torch.softmax(logits, dim=-1)
                sorted_preds = numpy.argsort(-yhat.data.cpu().numpy())
                real_topk_pos = list(
                    [int(numpy.where(sorted_preds[i] == tgt[i].item())[0][0])
                     for i in range(yhat.shape[0])])
                real_topk_probs = yhat[numpy.arange(
                    0, yhat.shape[0], 1), tgt].data.cpu().numpy().tolist()
                real_topk.extend(list(zip(real_topk_pos, real_topk_probs)))
                pred_topk.extend([list(zip(self.tokenizer.convert_ids_to_tokens(
                    sorted_preds[i][:topk]),
                    yhat[i][sorted_preds[i][:topk]].data.cpu().numpy().tolist()))
                                  for i in range(yhat.shape[0])])
        bpe_strings = [self.postprocess(s) for s in tokenized_text]
        pred_topk = [[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk]
        payload = {
            'bpe_strings': bpe_strings,
            'real_topk': real_topk,
            'pred_topk': pred_topk}
        return payload

    def postprocess(self, token):
        """
        DOCSTRING
        """
        with_space = True
        with_break = token == '[SEP]'
        if token.startswith('##'):
            with_space = False
            token = token[2:]
        if with_space:
            token = '\u0120' + token
        if with_break:
            token = '\u010A' + token
        return token

@class_register.register_api(name='gpt-2-small')
class LM(AbstractLanguageChecker):
    """
    DOCSTRING
    """
    def __init__(self, model_name_or_path="gpt2"):
        super(LM, self).__init__()
        self.enc = pytorch_pretrained_bert.GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = pytorch_pretrained_bert.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.start_token = '<|endoftext|>'
        print("Loaded GPT-2 model!")

    def check_probabilities(self, in_text, topk=40):
        """
        DOCSTRING
        """
        start_t = torch.full(
            (1, 1), self.enc.encoder[self.start_token],
            device=self.device, dtype=torch.long)
        context = self.enc.encode(in_text)
        context = torch.tensor(
            context, device=self.device, dtype=torch.long).unsqueeze(0)
        context = torch.cat([start_t, context], dim=1)
        logits, _ = self.model(context)
        yhat = torch.softmax(logits[0, :-1], dim=-1)
        y = context[0, 1:]
        sorted_preds = numpy.argsort(-yhat.data.cpu().numpy())
        real_topk_pos = list(
            [int(numpy.where(sorted_preds[i] == y[i].item())[0][0]) for i in range(y.shape[0])])
        real_topk_probs = yhat[numpy.arange(
            0, y.shape[0], 1), y].data.cpu().numpy().tolist()
        real_topk_probs = list(map(lambda x: round(x, 5), real_topk_probs))
        real_topk = list(zip(real_topk_pos, real_topk_probs))
        bpe_strings = [self.enc.decoder[s.item()] for s in context[0]]
        bpe_strings = [self.postprocess(s) for s in bpe_strings]
        pred_topk = [list(zip(
            [self.enc.decoder[p] for p in sorted_preds[i][:topk]],
            list(map(
                lambda x: round(x, 5),
                yhat[i][sorted_preds[i][:topk]].data.cpu().numpy().tolist()))))
                     for i in range(y.shape[0])]
        pred_topk = [[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk]
        payload = {
            'bpe_strings': bpe_strings,
            'real_topk': real_topk,
            'pred_topk': pred_topk}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return payload

    def postprocess(self, token):
        """
        DOCSTRING
        """
        with_space = False
        with_break = False
        if token.startswith('Ġ'):
            with_space = True
            token = token[1:]
        elif token.startswith('â'):
            token = ' '
        elif token.startswith('Ċ'):
            token = ' '
            with_break = True
        token = '-' if token.startswith('â') else token
        token = '“' if token.startswith('ľ') else token
        token = '”' if token.startswith('Ŀ') else token
        token = "'" if token.startswith('Ļ') else token
        if with_space:
            token = '\u0120' + token
        if with_break:
            token = '\u010A' + token
        return token

    def sample_unconditional(self, length=100, topk=5, temperature=1.0):
        """
        Sample `length` words from the model.

        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
        """
        context = torch.full(
            (1, 1), self.enc.encoder[self.start_token],
            device=self.device, dtype=torch.long)
        prev = context
        output = context
        past = None
        with torch.no_grad():
            for i in range(length):
                logits, past = self.model(prev, past=past)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(top_k_logits(logits, k=topk), dim=-1)
                prev = torch.multinomial(probs, num_samples=1)
                output = torch.cat((output, prev), dim=1)
        output_text = self.enc.decode(output[0].tolist())
        return output_text

def main():
    raw_text = ''
    with open('text,txt', newline='') as file:
        raw_text = file.read()
    # tests for BERT
    lm = BERTLM()
    start = time.time()
    payload = lm.check_probabilities(raw_text, topk=5)
    end = time.time()
    print("{:.2f} Seconds for a run with BERT".format(end - start))
    # tests for GPT-2
    lm = LM()
    start = time.time()
    payload = lm.check_probabilities(raw_text, topk=5)
    end = time.time()
    print("{:.2f} Seconds for a check with GPT-2".format(end - start))
    start = time.time()
    sample = lm.sample_unconditional()
    end = time.time()
    print("{:.2f} Seconds for a sample from GPT-2".format(end - start))
    print("SAMPLE:", sample)

def top_k_logits(logits, k):
    """
    Filters logits to only the top k choices
    from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
    """
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(
        logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

if __name__ == '__main__':
    main()
