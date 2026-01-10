from docsentinel2.nli_model import NLIModel
from docsentinel2.text_features import diff_words, extract_numbers
from docsentinel2.severity import classify_change


class DiffEngine:
   def __init__(self, embedder, aligner):
       self.embedder = embedder
       self.aligner = aligner
       self.nli = NLIModel()


   def detect_changes(self, old_sentences, new_sentences):
       alignments = self.aligner(old_sentences, new_sentences)
       changes = []


       for idx_old, idx_new in alignments:
           if idx_old is None:
               changes.append({
                   "old": "",
                   "new": new_sentences[idx_new],
                   "label": "ADDED_SENTENCE",
                   "cosine": 0.0,
                   "nli": (0.0, 0.0)
               })
               continue


           if idx_new is None:
               changes.append({
                   "old": old_sentences[idx_old],
                   "new": "",
                   "label": "REMOVED_SENTENCE",
                   "cosine": 0.0,
                   "nli": (0.0, 0.0)
               })
               continue


           old = old_sentences[idx_old]
           new = new_sentences[idx_new]


           cosine = self.embedder.similarity(old, new)
           removed, added = diff_words(old, new)
           old_nums = extract_numbers(old)
           new_nums = extract_numbers(new)


           nums_changed = old_nums != new_nums
           p1, p2 = self.nli.bidirectional(old, new)


           label = classify_change(
               cosine, p1, p2,
               removed, added,
               nums_changed
           )


           if label != "NO_CHANGE":
               changes.append({
                   "old": old,
                   "new": new,
                   "label": label,
                   "cosine": cosine,
                   "nli": (p1, p2)
               })


       return changes