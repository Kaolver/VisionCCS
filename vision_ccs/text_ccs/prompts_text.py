"""Contrast-pair templates: item -> (question, (answer0, answer1)).

item['label'] is the index of the true answer. 'burns' templates are copied
from Appendix I of Burns et al.; 'ps-style' ones paraphrase promptsource.
"""

AG_NEWS_LABELS = ['World politics', 'Sports', 'Business', 'Science and technology']

DBPEDIA_LABELS = ['company', 'educational institution', 'artist', 'athlete',
                  'office holder', 'mean of transportation', 'building',
                  'natural place', 'village', 'animal', 'plant', 'album', 'film',
                  'written work']

CHOICES = ('choice 1', 'choice 2')


def _q(s):
    return f'"{s}"'


def _topic(field):
    # prompts 1-6 of Appendix I.2.1 / I.2.5
    def t(open_, body):
        return lambda it: (open_(it) + body.format(l0=it['label0'], l1=it['label1']), CHOICES)

    consider = lambda it: f'Consider the following example: {_q(it[field])} '
    plain = lambda it: f'{it[field]} '
    return [
        ('burns', t(consider, 'Choice 1: {l0}. Choice 2: {l1}.Between choice 1 and choice 2, the topic of this example is')),
        ('burns', t(consider, 'Choice 1: {l0}. Choice 2: {l1}.Between choice 1 and choice 2, what is the topic of this example?')),
        ('burns', t(consider, 'Which is the topic of this example, choice 1: {l0}, or choice 2: {l1}?')),
        ('burns', t(plain, 'Choice 1: {l0}. Choice 2: {l1}.Between choice 1 and choice 2, the topic of this example is')),
        ('burns', t(plain, 'Choice 1: {l0}. Choice 2: {l1}.Between choice 1 and choice 2, what is the topic of this example?')),
        ('burns', t(plain, 'Which is the topic of this example, choice 1: {l0}, or choice 2: {l1}?')),
    ]


def _ag_news():
    plain = lambda it, s: (f'{it["text"]} ' + s.format(l0=it['label0'], l1=it['label1']), CHOICES)
    return _topic('text') + [
        ('burns', lambda it: plain(it, 'What label best describes this news article, choice 1: {l0}, or choice 2: {l1}?')),
        ('burns', lambda it: plain(it, 'Which section of a newspaper would this article likely appear in, choice 1: {l0}, or choice 2: {l1}?')),
    ]


def _dbpedia():
    plain = lambda it, s: (f'{it["content"]} ' + s.format(l0=it['label0'], l1=it['label1']), CHOICES)
    return _topic('content') + [
        ('burns', lambda it: plain(it, 'What category does the paragraph belong to, choice 1: {l0}, or choice 2: {l1}?')),
        ('burns', lambda it: plain(it, 'What label best describes this paragraph, choice 1: {l0}, or choice 2: {l1}?')),
    ]


def _imdb():
    NP = ('negative', 'positive')
    r = lambda it: it['text']
    return [
        ('burns', lambda it: (f'Consider the following example: {_q(r(it))} Between negative and positive, the sentiment of this example is', NP)),
        ('burns', lambda it: (f'Consider the following example: {_q(r(it))} Between negative and positive, which is the sentiment of this example?', NP)),
        ('ps-style', lambda it: (f'{r(it)} Did the reviewer find this movie good or bad?', ('bad', 'good'))),
        ('ps-style', lambda it: (f'{r(it)} Is this review positive or negative?', NP)),
        ('ps-style', lambda it: (f'{r(it)} How does the viewer feel about the movie?', NP)),
        ('ps-style', lambda it: (f'{r(it)} What sentiment does the writer express for the movie?', NP)),
        ('ps-style', lambda it: (f'The following movie review expresses what sentiment? {r(it)}', NP)),
        ('ps-style', lambda it: (f'{r(it)} What is the sentiment expressed by the reviewer for the movie?', NP)),
    ]


def _amazon():
    NP = ('negative', 'positive')
    tr = lambda it: f'Title: {it["title"]}\nReview: {it["content"]}'
    return [
        ('burns', lambda it: (f'Consider the following example: {_q(it["content"])} Between negative and positive, the sentiment of this example is', NP)),
        ('burns', lambda it: (f'Consider the following example: {_q(it["content"])} Between negative and positive, which is the sentiment of this example?', NP)),
        ('ps-style', lambda it: (f'{tr(it)}\nIs the review positive or negative?', NP)),
        ('ps-style', lambda it: (f'Here is a review left by a customer on a product. Would you say he was satisfied or dissatisfied?\n{tr(it)}', ('dissatisfied', 'satisfied'))),
        ('ps-style', lambda it: (f'Is this product review positive?\n{tr(it)}\nAnswer:', ('No', 'Yes'))),
        ('ps-style', lambda it: (f'{tr(it)}\nWould you say this review depicts the product in a flattering or unflattering light?', ('unflattering', 'flattering'))),
        ('ps-style', lambda it: (f'{tr(it)}\nDoes this product review convey a negative or positive sentiment?', NP)),
        ('ps-style', lambda it: (f'You are considering whether to buy a product. You look at the reviews. Would the following review decrease or increase the chances of you buying the product?\n{tr(it)}', ('decrease', 'increase'))),
    ]


def _boolq():
    NY = ('No', 'Yes')
    return [
        ('ps-style', lambda it: (f'{it["passage"]}\n\nQuestion: {it["question"]}?\nAnswer:', NY)),
        ('ps-style', lambda it: (f'Passage: {it["passage"]}\n\nAfter reading this passage, I have a question: {it["question"]}? True or False?', ('False', 'True'))),
        ('ps-style', lambda it: (f'{it["passage"]}\n\nHaving read that, I wonder {it["question"]}?', NY)),
        ('ps-style', lambda it: (f'Based on the following passage, {it["question"]}? {it["passage"]}', NY)),
        ('ps-style', lambda it: (f'{it["passage"]}\n\nHaving read that, could you tell me {it["question"]}?', NY)),
        ('ps-style', lambda it: (f'EXAM\n1. Answer by yes or no.\n\nDocument: {it["passage"]}\nQuestion: {it["question"]}?', NY)),
        ('ps-style', lambda it: (f'Text: {it["passage"]}\n\nAnswer the following yes/no question: {it["question"]}? Yes or no?', NY)),
    ]


def _copa():
    def conn(it):
        return 'because' if it['question'] == 'cause' else 'so'

    def lc(s):
        return s[:1].lower() + s[1:]

    sols = lambda it: (lc(it['choice1']), lc(it['choice2']))
    prem = lambda it: it['premise'].rstrip('.')
    return [
        ('burns', lambda it: (f'Consider the following premise: {_q(it["premise"])} Choice 1: {it["choice1"]} Choice 2: {it["choice2"]} Q: Which one is more likely to be the {it["question"]}, choice 1 or choice 2?', CHOICES)),
        ('ps-style', lambda it: (f'{prem(it)} {conn(it)}', sols(it))),
        ('ps-style', lambda it: (f'Exercise: choose the most plausible alternative.\n\n{prem(it)} {conn(it)}...\n- {it["choice1"]}\n- {it["choice2"]}', (it['choice1'], it['choice2']))),
        ('ps-style', lambda it: (f'{it["premise"]}\n\nWhat\'s the best option?\n- {it["choice1"]}\n- {it["choice2"]}\n\nWe are looking for {"a cause" if it["question"] == "cause" else "an effect"}', (it['choice1'], it['choice2']))),
        ('ps-style', lambda it: (f'Pick the more likely continuation to the following sentence:\n{prem(it)} {"as a result of" if it["question"] == "cause" else "as a consequence"}:', sols(it))),
        ('ps-style', lambda it: (f'{it["premise"]}\n\nI am hesitating between two options. Help me choose the more likely {it["question"]}:\n- {it["choice1"]}\n- {it["choice2"]}', (it['choice1'], it['choice2']))),
    ]


def _rte():
    NY = ('No', 'Yes')
    p, h = (lambda it: it['premise']), (lambda it: it['hypothesis'])
    return [
        ('burns', lambda it: (f'{p(it)} Question: Does this imply that {_q(h(it))}, yes or no?', ('no', 'yes'))),
        ('ps-style', lambda it: (f'{p(it)}\nQuestion: {h(it)} True or False?', ('False', 'True'))),
        ('ps-style', lambda it: (f'{p(it)}\n\nBased on that paragraph can we conclude that this sentence is true?\n{h(it)}', NY)),
        ('ps-style', lambda it: (f'Given that {p(it)} Does it follow that {h(it)} Yes or no?', NY)),
        ('ps-style', lambda it: (f'{p(it)} Using only the above description and what you know about the world, is {_q(h(it))} definitely correct? Yes or no?', NY)),
        ('ps-style', lambda it: (f'Suppose {p(it)} Can we infer that {_q(h(it))}? Yes or no?', NY)),
        ('ps-style', lambda it: (f'{p(it)} Based on the previous passage, is it true that {_q(h(it))}? Yes or no?', NY)),
    ]


def _qnli():
    ny = ('no', 'yes')
    q, s = (lambda it: it['question']), (lambda it: it['sentence'])
    return [
        ('ps-style', lambda it: (f'Can you answer the question {_q(q(it))} based only on the following:\n{s(it)}', ny)),
        ('ps-style', lambda it: (f'Does the sentence {_q(s(it))} provide a valid answer to the question {_q(q(it))}?', ny)),
        ('ps-style', lambda it: (f'Does knowing that {_q(s(it))} help me answer {_q(q(it))}?', ny)),
        ('ps-style', lambda it: (f'Question: {q(it)}\nSentence: {s(it)}\nDoes the sentence answer the question?', ('No', 'Yes'))),
        ('ps-style', lambda it: (f'{s(it)}\nDoes that sentence have all you need to answer the question {_q(q(it))}?', ny)),
    ]


def _piqa():
    sols = lambda it: (it['sol1'], it['sol2'])
    return [
        ('ps-style', lambda it: (f'Goal: {it["goal"]}\n\nWhich is the correct ending?\n- {it["sol1"]}\n- {it["sol2"]}\n\nAnswer:', sols(it))),
        ('ps-style', lambda it: (it['goal'], sols(it))),
        ('ps-style', lambda it: (f'Solution 1: {it["sol1"]}\nSolution 2: {it["sol2"]}\n\nGoal: {it["goal"]}\n\nGiven the goal, what is the correct solution?\n\nAnswer by copying the correct solution', sols(it))),
        ('ps-style', lambda it: (f'Sentence: {it["goal"]}\n\nChoice 1: {it["sol1"]}\n\nChoice 2: {it["sol2"]}\n\nWhat is the index of the correct choice for ending for the sentence?\n\nAnswer:', ('1', '2'))),
        ('ps-style', lambda it: (f'Here is a goal: {it["goal"]}\n\nHow would you accomplish this goal?', sols(it))),
        ('ps-style', lambda it: (f'Here is a goal: {it["goal"]}\n\nWhich way makes more sense to accomplish this goal?', sols(it))),
        ('ps-style', lambda it: (f'Choose the most sensible solution to achieve the goal:\nGoal: {it["goal"]}\nSolution 1: {it["sol1"]}\nSolution 2: {it["sol2"]}\nAnswer:', ('Solution 1', 'Solution 2'))),
    ]


TEMPLATES = {
    'imdb': _imdb(),
    'amazon_polarity': _amazon(),
    'ag_news': _ag_news(),
    'dbpedia_14': _dbpedia(),
    'rte': _rte(),
    'qnli': _qnli(),
    'copa': _copa(),
    'boolq': _boolq(),
    'piqa': _piqa(),
}

DATASETS = list(TEMPLATES)


def render(dataset, t_idx, item):
    return TEMPLATES[dataset][t_idx][1](item)


def n_templates(dataset):
    return len(TEMPLATES[dataset])


def join_raw(question, answer):
    """Appendix I.1: a line break unless the question ends in a space or line break."""
    sep = '' if question.endswith((' ', '\n')) else '\n'
    return question + sep, answer
