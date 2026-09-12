"""Prompt templates for contrast-pair construction.

Burns runs 8-13 templates per dataset; with a single surface form there is no
variation for truth to be the feature that stays consistent across it.

Each template must differ between its positive and negative rendering ONLY in
the final answer word, or the prefix leaks the label. The zero-shot prefix is
derived as text.split('{a}')[0], so prompt parity is mechanical.
"""

# name -> (text, positive answer word, negative answer word)
# Answer-word casing varies on purpose: 'Yes' and 'yes' are different token ids,
# which breaks the surface direction per-branch centering cannot remove.
TEMPLATES = {
    # 'plain' reproduces the v1 caches exactly; keep it first.
    'plain':    ('{q}? {a}', 'Yes', 'No'),
    'qa':       ('Question: {q}?\nAnswer: {a}', 'Yes', 'No'),
    'qa_lower': ('Q: {q}?\nA: {a}', 'yes', 'no'),
    'based_on': ('Based on the image, {q}? {a}', 'Yes', 'No'),
    'correct':  ('{q}?\nThe correct answer is: {a}', 'yes', 'no'),
}

DEFAULT_TEMPLATES = ['plain']
ALL_TEMPLATES = list(TEMPLATES)


def template_names(spec):
    """Resolve a CLI --templates value; 'all' expands. Unknown names raise."""
    if spec in (None, [], ['default']):
        return list(DEFAULT_TEMPLATES)
    names = ALL_TEMPLATES if spec == ['all'] else list(spec)
    unknown = [n for n in names if n not in TEMPLATES]
    if unknown:
        raise ValueError(f'unknown template(s) {unknown}; known: {ALL_TEMPLATES}')
    return names


def render(name, question, positive):
    """Render one side of a contrast pair."""
    text, pos_word, neg_word = TEMPLATES[name]
    q = question.rstrip('?').strip()
    return text.format(q=q, a=pos_word if positive else neg_word)


def answer_words(name):
    """The (positive, negative) answer words this template contrasts."""
    _, pos_word, neg_word = TEMPLATES[name]
    return pos_word, neg_word


def zeroshot_prefix(name, question):
    """The CCS prompt truncated immediately before the answer word."""
    text, _, _ = TEMPLATES[name]
    q = question.rstrip('?').strip()
    return text.split('{a}')[0].format(q=q).rstrip()
