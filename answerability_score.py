import truecase

stop_words = {"did", "have", "ourselves", "hers", "between", "yourself",
              "but", "again", "there", "about", "once", "during", "out", "very",
              "having", "with", "they", "own", "an", "be", "some", "for", "do", "its",
              "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s",
              "am", "or", "as", "from", "him", "each", "the", "themselves", "until", "below",
              "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were",
              "her", "more", "himself", "this", "down", "should", "our", "their", "while",
              "above", "both", "up", "to", "ours", "had", "she", "all", "no", "at", "any",
              "before", "them", "same", "and", "been", "have", "in", "will", "on", "does",
              "yourselves", "then", "that", "because", "over", "so", "can", "not", "now", "under",
              "he", "you", "herself", "has", "just", "too", "only", "myself", "those", "i", "after",
              "few", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "further",
              "was", "here", "than"}

question_words_global = {'What', 'Which', 'Why', 'Who', 'Whom', 'Whose', 'Where', 'When', 'How'}
question_words_global.update([w.lower() for w in question_words_global])


def remove_stopwords_and_NER_line(question, relevant_words=None, question_words=None):
    global question_words_global
    if relevant_words is None:

        question = question.split()
        if question_words is None:
            question_words = question_words_global

        temp_words = []
        for word in question_words:
            for i, w in enumerate(question):
                if w == word:
                    temp_words.append(w)
                    # If the question type is 'what' or 'which' the following word is generally associated with
                    # with the answer type. Thus it is important that it is considered a part of the question.
                    if i + 1 < len(question) and (w.lower() == "what" or w.lower() == "which"):
                        temp_words.append(question[i + 1])

        question_split = [item for item in question if item not in temp_words]
        ner_words = question_split
        temp_words = []

        for i in ner_words:
            if i[0].isupper() == False:
                if i not in stop_words:
                    temp_words.append(i)

        return [word.lower() for word in temp_words]
    else:
        question_words = question.split()
        temp_words = []
        for i in question_words:
            for j in relevant_words:
                if j.lower() in i:
                    temp_words.append(i)
        return [word.lower() for word in temp_words]


def NER_line(question):
    global question_words_global
    q_types = question_words_global
    question_words = question.split()
    if question_words:
        if question_words[0].lower() in q_types:
            question_words = question_words[1:]

    temp_words = []
    for i in question_words:
        if i[0].isupper():
            temp_words.append(i)

    return [word.lower() for word in temp_words]


def get_stopwords(question):
    global stop_words
    question_words = question.split()
    temp_words = []
    for i in question_words:
        if i.lower() in stop_words:
            temp_words.append(i.lower())

    return [word.lower() for word in temp_words]


def questiontype(question, questiontypes=None):
    global question_words_global
    if questiontypes is None:
        types = question_words_global
        question = question.strip()
        temp_words = []
        question = question.split()

        for word in types:
            for i, w in enumerate(question):
                if w == word:
                    temp_words.append(w)
                    if i + 1 < len(question) and (w.lower() == "what" or w.lower() == "which"):
                        temp_words.append(question[i + 1])

        return [word.lower() for word in temp_words]
    else:
        for i in questiontypes:
            if question.startswith(i + " "):
                return i
            else:
                return []


def get_f_score(pred, ref):
    pred_len = len(pred)
    ref_len = len(ref)

    if pred_len == 0 and ref_len == 0:
        return 1

    tp = 0
    for word in pred:
        if word in ref:
            tp += 1

    p = tp/pred_len if pred_len != 0 else 0
    r = tp/ref_len if ref_len != 0 else 0

    f = (2 * p * r) / (p+r) if (p+r) != 0 else 0

    return f


def get_answerability_score(hypothesis, reference):
    qt_weight = 0.2
    re_weight = 0.36
    ner_weight = 0.41

    hypothesis = truecase.get_true_case(hypothesis)
    reference = truecase.get_true_case(reference)

    relevant_words = None
    questiontypes = None

    impwords_hyp = remove_stopwords_and_NER_line(hypothesis, relevant_words)
    ner_hyp = NER_line(hypothesis)
    qt_hyp = questiontype(hypothesis, questiontypes)
    sw_hyp = get_stopwords(hypothesis)

    impwords_ref = remove_stopwords_and_NER_line(reference, relevant_words)
    ner_ref = NER_line(reference)
    qt_ref = questiontype(reference, questiontypes)
    sw_ref = get_stopwords(reference)

    impwords_f = get_f_score(impwords_hyp, impwords_ref)
    ner_f = get_f_score(ner_hyp, ner_ref)
    qt_f = get_f_score(qt_hyp, qt_ref)
    sw_f = get_f_score(sw_hyp, sw_ref)

    ans_score = re_weight * impwords_f + ner_weight * ner_f + \
                        qt_weight * qt_f + (1 - re_weight - ner_weight - qt_weight) * sw_f

    return ans_score













