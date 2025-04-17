import streamlit as st
import codecs
import numpy as np

# POS tags
TAGS = ['NN', 'NST', 'NNP', 'PRP', 'DEM', 'VM', 'VAUX', 'JJ', 'RB', 'PSP', 'RP', 'CC', 'WQ', 'QF', 'QC', 'QO', 'CL',
        'INTF', 'INJ', 'NEG', 'UT', 'SYM', 'COMP', 'RDP', 'ECH', 'UNK', 'XC']

# Global variables
wordtypes = []
emission_matrix = []
transmission_matrix = []
tagscount = [0] * len(TAGS)


def load_training_data(training_file):
    global wordtypes, emission_matrix, transmission_matrix, tagscount
    with codecs.open(training_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    exclude = ["<s>", "</s>", "START", "END"]
    for line in lines:
        words = line.strip().split(' ')
        if words and words[0] not in exclude:
            if words[0] not in wordtypes:
                wordtypes.append(words[0])
            if words[1] in TAGS:
                tagscount[TAGS.index(words[1])] += 1

    emission_matrix = np.zeros((len(TAGS), len(wordtypes)))
    transmission_matrix = np.zeros((len(TAGS), len(TAGS)))

    prev_row_id = -1
    for line in lines:
        words = line.strip().split(' ')
        if words and words[0] not in exclude:
            col_id = wordtypes.index(words[0])
            row_id = TAGS.index(words[1])
            emission_matrix[row_id][col_id] += 1
            if prev_row_id != -1:
                transmission_matrix[prev_row_id][row_id] += 1
            prev_row_id = row_id
        else:
            prev_row_id = -1

    for x in range(len(TAGS)):
        if tagscount[x] != 0:
            emission_matrix[x] /= tagscount[x]
            transmission_matrix[x] /= tagscount[x]


def max_connect(x, y, viterbi_matrix, emission, transmission_matrix):
    max_prob = -99999
    path = -1
    for k in range(len(TAGS)):
        val = viterbi_matrix[k][x - 1] * transmission_matrix[k][y]
        if val * emission > max_prob:
            max_prob = val
            path = k
    return max_prob, path


def pos_tag_sentence(sentence):
    test_words = sentence.strip().split(' ')
    pos_tags = [-1] * len(test_words)
    viterbi_matrix = np.zeros((len(TAGS), len(test_words)))
    viterbi_path = np.zeros((len(TAGS), len(test_words)), dtype=int)

    for x in range(len(test_words)):
        for y in range(len(TAGS)):
            emission = emission_matrix[TAGS.index(TAGS[y])][wordtypes.index(test_words[x])] if test_words[
                                                                                                   x] in wordtypes else 0.001
            max_prob, viterbi_path[y][x] = max_connect(x, y, viterbi_matrix, emission,
                                                       transmission_matrix) if x > 0 else (1, -1)
            viterbi_matrix[y][x] = emission * max_prob

    max_val = max(viterbi_matrix[x][-1] for x in range(len(TAGS)))
    max_index = np.argmax([viterbi_matrix[x][-1] for x in range(len(TAGS))])

    for x in range(len(test_words) - 1, -1, -1):
        pos_tags[x] = max_index
        max_index = viterbi_path[max_index][x]

    return " ".join(f"{test_words[i]}_{TAGS[pos_tags[i]]}" for i in range(len(test_words)))


# Streamlit UI
st.title("POS Tagger for Hindi and Tamil")

# Language selection
language = st.selectbox("Select Language:", ["Hindi", "Tamil"])

# Load training data based on selection
training_file = "hindi_training.txt" if language == "Hindi" else "tamil_training.txt"
load_training_data(training_file)

st.write(f"Enter a {language} sentence to get POS tags:")

sentence = st.text_input(f"{language} Sentence:")
if st.button("Tag POS"):
    if sentence:
        result = pos_tag_sentence(sentence)
        st.write("**POS Tagged Sentence:**")
        st.write(result)
    else:
        st.write("Please enter a valid sentence.")
