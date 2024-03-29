from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
from nltk.tokenize import word_tokenize
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, id,text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.id = id


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ori_idx_token,id,lens=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ori_idx_token = ori_idx_token
        self.id = id
        self.lens = lens



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # return ["0", "1", ..., "n"] for multiclass classification
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, id=i,text_b=None, label=label))
        return examples
class AgNewsProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # return ["0", "1", ..., "n"] for multiclass classification
        return ["0","1","2","3"]
        # return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, id=i,text_b=None, label=label))
        return examples

class sst1Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # return ["0", "1", ..., "n"] for multiclass classification
        return ["0","1","2","3","4"]
        # return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, id=i,text_b=None, label=label))
        return examples

class ImdbProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # return ["0", "1", ..., "n"] for multiclass classification
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, id=i,text_b=None, label=label))
        return examples
class trecProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # return ["0", "1", ..., "n"] for multiclass classification
        return ["0", "1","2","3","4","5"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, id=i,text_b=None, label=label))
        return examples
class subjProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # return ["0", "1", ..., "n"] for multiclass classification
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, id=i,text_b=None, label=label))
        return examples
class snliProcessor(DataProcessor):
        
        """Processor for the SST-2 data set (GLUE version)."""

        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

        def get_test_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

        def get_labels(self):
            """See base class."""
            # return ["0", "1", ..., "n"] for multiclass classification
            return ["0", "1","2"]

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                text_b = line[1]
                label = line[2]

                examples.append(
                    InputExample(guid=guid, text_a=text_a, id=i,text_b=text_b, label=label))
            return examples
        
class rottenProcessor(DataProcessor):
        
        """Processor for the SST-2 data set (GLUE version)."""

        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

        def get_test_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

        def get_labels(self):
            """See base class."""
            # return ["0", "1", ..., "n"] for multiclass classification
            return ["0", "1"]

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label = line[1]
                # label = line[2]

                examples.append(
                    InputExample(guid=guid, text_a=text_a, id=i,text_b=None, label=label))
            return examples
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        ori_idx_token = {}
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            # print("tokensa:{} tokensb:{}".format(tokens_a,tokens_b))
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # for tok in tokens_a:
        #     tok_id = tokenizer.vocab.get(tok)
        #     ori_idx_token[tok_id] = tok
        # tok_id = tokenizer.tokenize(tokens_a)
        # tok_id = tokenizer.convert_tokens_to_ids(tokens_a)
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              ori_idx_token=ori_idx_token,
                              id = example.id))
    return features
def _truncate_seq_pair(tokens_a, tokens_b, max_seq_length):
    if len(tokens_a) + len(tokens_b) > max_seq_length:
        if len(tokens_a) > len(tokens_b):
            while len(tokens_a) + len(tokens_b) > max_seq_length and len(tokens_a) != 0: 
                tokens_a.pop()
                
        else:
            while len(tokens_a) + len(tokens_b) > max_seq_length and len(tokens_b) != 0: 
                tokens_b.pop()    
        if len(tokens_a) + len(tokens_b) > max_seq_length:
            if len(tokens_a) > len(tokens_b):
                while len(tokens_a) + len(tokens_b) > max_seq_length and len(tokens_a) != 0: 
                    tokens_a.pop()
            else:
                while len(tokens_a) + len(tokens_b) > max_seq_length and len(tokens_b) != 0: 
                    tokens_b.pop()   
    # while len(tokens_a) + len(tokens_b) > max_seq_length:
        #     if len(tokens_a) > 0:
        #         tokens_a.pop()
        #     if len(tokens_b) > 0:
        #         tokens_b.pop()
    else:
        # while len(tokens)
        tokens_b += ["[PAD]"] * (max_seq_length - (len(tokens_a) + len(tokens_b)))


def convert_examples_to_features_lstm(examples,label_list,max_seq_length,vocab):
    label_map = {label : i for i, label in enumerate(label_list)}
    vocab = {word:i for i,word in enumerate(vocab)}

    features = []
    # lens = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = word_tokenize(example.text_a)
        mask = []
        l = None
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            mask = [1] * max_seq_length
            l = max_seq_length
        else:
            mask = [1] * len(tokens_a) + [0] * (max_seq_length - len(tokens_a))
            l = len(tokens_a)
            tokens_a = tokens_a + ['pad'] * (max_seq_length - len(tokens_a))

        tokens_a = [vocab[word] if word in vocab else 1 for word in tokens_a]
        assert len(tokens_a) == max_seq_length
        assert len(mask) == max_seq_length
        label_id = label_map[example.label]
            
        features.append(
                InputFeatures(input_ids=tokens_a,
                              input_mask=mask,
                              label_id=label_id,
                              segment_ids=None,
                              ori_idx_token=None,
                              id = example.id,
                              lens = l))
    return features
processors = {
    "sst-2": Sst2Processor,
    "agnews": AgNewsProcessor,
    "sst-1":sst1Processor,
    "imdb":ImdbProcessor,
    "trec":trecProcessor,
    "subj":subjProcessor,
    "snli":snliProcessor,
    "rotten_tomatoes":rottenProcessor
}

output_modes = {
    "sst-2": "classification",
    "agnews":"classification",
    "sst-1":"classification",
    "imdb":"classification",
    "trec":"classification",
    "subj":"classification",
    "snli":"classification",
    "rotten_tomatoes":"classification"
}

GLUE_TASKS_NUM_LABELS = {
    "sst-2": 2,
    "agnews": 4,
    "sst-1":5,
    "imdb": 2,
    "trec":6,
    "subj":2,
    "snli":3,
    "rotten_tomatoes":2
}
