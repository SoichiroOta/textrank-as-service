from gensim.summarization import keywords
import spacy
import pytextrank


class GensimTextRank:
    def keywords(self, text, ratio=0.2):
        return dict(keywords(text, ratio, scores=True))


class PyTextRank:
    def __init__(self, model_name):
        self.model_name = model_name

    def keywords(self, text, ratio=0.2):
        nlp = spacy.load(self.model_name)

        tr = pytextrank.TextRank()
        nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

        doc = nlp(text)

        if doc._.phrases:
            phrase_count = len(doc._.phrases)
            lower_limit_num = max(int(ratio * phrase_count), 1)
            return dict([(p.text, p.rank) for p in doc._.phrases[:lower_limit_num]])
        else:
            return dict()


class TextRank:
    def __init__(self, library=None, model_name=None):
        if library == 'pytextrank':
            self.textrank = PyTextRank(model_name)
        else:
            self.textrank = GensimTextRank()

    def keywords(self, text, ratio=0.2):
        return self.textrank.keywords(text, ratio)