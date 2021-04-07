class InputExample(object):
    def __init__(self, guid):
        self.guid = guid

class TextClassificationInputExample(InputExample):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, x0=None, y0=None, x1=None, y1=None):
        """
        Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        super().__init__(guid)
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        if x0 is None:
            self.bboxes = None
        else:
            self.bboxes = [[a, b, c, d] for a, b, c, d in zip(x0, y0, x1, y1)]

class SeqTaggingInputExample(InputExample):
    """A single training/test example for simple sequence tagging."""

    def __init__(self, guid, words, labels, x0=None, y0=None, x1=None, y1=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The tokens of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
            x0: (Optional) list. The list of x0 coordinates for each word.
            y0: (Optional) list. The list of y0 coordinates for each word.
            x1: (Optional) list. The list of x1 coordinates for each word.
            y1: (Optional) list. The list of y1 coordinates for each word.
        """
        super().__init__(guid)
        self.words = words
        self.labels = labels
        if x0 is None:
            self.bboxes = None
        else:
            self.bboxes = [[a, b, c, d] for a, b, c, d in zip(x0, y0, x1, y1)]

class SpanExtractionInputExample(InputExample):
    """A single training/test example for simple span extraction."""

    def __init__(self, guid, text_a, text_b=None, label=None, x0=None, y0=None, x1=None, y1=None):
        """
        Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        super().__init__(guid)
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        if x0 is None:
            self.bboxes = None
        else:
            self.bboxes = [[a, b, c, d] for a, b, c, d in zip(x0, y0, x1, y1)]