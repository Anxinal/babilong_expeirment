"""Rewrites BABILong rows before they are tokenized.

A ``Preprocessor`` is a batched ``datasets.Dataset.map`` function:

    ds = ds.map(Preprocessor(), batched=True)

or pass one to ``load_dataset.load_tasks(..., preprocessor=Preprocessor())``. Subclass it
and override ``process_individual_input`` to change what the model sees.
"""

import re




class Preprocessor:

    def __init__(self):
        pass

    def process_individual_input(self, input: str, question: str, output:str):
        """Return the new ``(input, question, output)`` of one sample. The default keeps all three."""
        return input, question, output
    def process_batch(self, inputs, questions, outputs):
        new_input = []
        new_question = []
        new_outputs = []
        for input, question, output in zip(inputs, questions, outputs):
            newi, newq, newo = self.process_individual_input(input, question, output)
            new_input.append(newi)
            new_question.append(newq)
            newo = "" if not newo else newo
            new_outputs.append(newo)

        return new_input, new_question, new_outputs

    def __call__(self, batch):
        """Batched ``map`` entry point: rewrites the ``input``, ``question`` and ``target`` columns.

        The answer column is called ``target`` in the Hub data; it is the ``output`` here.
        """
        inputs, questions, outputs = self.process_batch(batch["input"], batch["question"], batch["target"])
        return {"input": inputs, "question": questions, "target": outputs}

import random 
import uuid
import string
class NameReplacemen(Preprocessor):

    def __init__(self):
        pass
        self.rng = random.Random(0)

    def _map_generate(self, namelist):
        uuidmap = dict()

        for name in namelist:
            generated = uuid.UUID(int=self.rng.getrandbits(128),version=4).hex[:10]
            uuidmap[name] = generated
        return uuidmap

    def _replace_in_string(self, text, mapping):
        """Replace every whole-word occurrence of each key of *mapping* in *text* with its value.

        All keys are matched in one pass, so a value already inserted is never replaced
        again by another key. A match must not be part of a longer word: ``"Mary"`` is
        replaced in ``"Where is Mary?"`` but not in ``"Maryland"``. Case-sensitive.
        """
        keys = [k for k in mapping if k]
        if not keys:
            return text
        # Longest first, so a key that is a prefix of another never wins the match.
        keys.sort(key=len, reverse=True)
        pattern = r"(?<!\w)(?:" + "|".join(map(re.escape, keys)) + r")(?!\w)"
        return re.sub(pattern, lambda m: mapping[m.group(0)], text)
    
    def process_individual_input(self, input, question, output):
        BABI_NAMES = ["Daniel", "John", "Mary", "Sandra", "Bill", "Fred", "Jeff", "Julie"]
        # individual inputs use new maps to prevent tokeniser from memorising the pattern
        namemap = self._map_generate(BABI_NAMES)
        input, question, output = [self._replace_in_string(to_repalce, namemap) for to_repalce in [input, question, output]]
        return input, question, output
        


    def process_batch(self, inputs, questions, outputs):
        new_input = []
        new_question = []
        new_outputs = []
        for input, question, output in zip(inputs, questions, outputs):
            newi, newq, newo = self.process_individual_input(input, question, output)
            new_input.append(newi)
            new_question.append(newq)
            new_outputs.append(newo)
        return new_input, new_question, new_outputs

    def __call__(self, batch):
        """Batched ``map`` entry point: rewrites the ``input``, ``question`` and ``target`` columns."""
        inputs, questions, outputs = self.process_batch(batch["input"], batch["question"], batch["target"])
        return {"input": inputs, "question": questions, "target": outputs}
