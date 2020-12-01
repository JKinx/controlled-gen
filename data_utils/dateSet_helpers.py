""" Helper functions for parsing and preparing dataset from
the dateSet dataset
"""
from num2words import num2words

def dateSet_tuple_to_kvs(entry):
    day, month, year = entry
    assert day > 0 and day <= 31
    assert month > 0 and month <= 12
    assert year >= 2000 and year <= 2020

    if day < 10:
        day = "0 " + str(day)
    else:
        day = " ".join([el for el in str(day)])
    [day0, day1] = day.split(" ")
    month0 = str(month)
    year0 = str(year)

    return [("_day_0", day0), ("_day_1", day1), ("_month_0", month0), 
            ("_year_0", year0)]

def dateSet_filter_yz(y, z):
    filtered = list(filter(lambda x : not("start" in x[0] or "end" in x[0]),
            zip(y, z)))
    y = list(map(lambda x : x[0], filtered))
    z = list(map(lambda x : x[1], filtered))
    return y, z

def dateSet_decode_out(dataset, ys, zs):
    batch_size = len(ys)
    sents = []
    states = []

    for idx in range(batch_size):
        sent = [dataset.id2word[el] for el in ys[idx]]
        sentence, state = dateSet_filter_yz(sent, zs[idx])
        sents.append(sentence)
        states.append(state)

    return sents, states

months = ["january", "february", "march", "april", "may", "june", "july", 
          "august", "september", "october", "november", "december"]
days_numerical = [num2words(n).replace("-", " ") for n in range(1, 32)]
days_ordinal = [num2words(n, ordinal=True).replace("-", " ") 
                    for n in range(1, 32)]

def dateSet_prep_sent(sent):
    sent_len = len(sent)

    i = 0 

    prepped_sent = []
    while i < sent_len:
        word = sent[i]
        
        if word in months:
            prepped_sent += ["_mstart_", word, "_mend_"]
            i += 1
        elif word == "twenty" or word == "thirty":
            next_word = sent[i + 1]

            if next_word in days_ordinal:
                prepped_sent += ["_odstart_", word, next_word, "_odend_"]
                i += 2
            elif next_word in days_numerical:
                prepped_sent += ["_ndstart_", word, next_word, "_ndend_"]
                i += 2
            else:
                prepped_sent += ["_ndstart_", word, "_ndend_"]
                i += 1
        elif word in days_numerical:
            prepped_sent += ["_ndstart_", word, "_ndend_"]
            i += 1
        elif word in days_ordinal:
            prepped_sent += ["_odstart_", word, "_odend_"]
            i += 1
        else:
            prepped_sent.append(word)
            i += 1

    return prepped_sent
