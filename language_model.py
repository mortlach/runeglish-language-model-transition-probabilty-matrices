import random
import time
import os
import csv
import math
# from line_profiler import LineProfiler


class LanguageModel:
    test_rune_string = "ᚹᛖᛚᚳᚩᛗᛖ ᚹᛖᛚᚳᚩᛗᛖ ᛈᛁᛚᚷᚱᛁᛗ ᛏᚩ ᚦᛖ ᚷᚱᛠᛏ ᛂᚩᚢᚱᚾᛖᚣ ᛏᚩᚹᚪᚱᛞ ᚦᛖ ᛖᚾᛞ ᚩᚠ ᚪᛚᛚ ᚦᛝᛋ ᛁᛏ ᛁᛋ ᚾᚩᛏ ᚪᚾ ᛠᛋᚣ ᛏᚱᛁᛈ ᛒᚢᛏ ᚠᚩᚱ ᚦᚩᛋᛖ ᚹᚻᚩ ᚠᛁᚾᛞ ᚦᛖᛁᚱ ᚹᚪᚣ ᚻᛖᚱᛖ ᛁᛏ ᛁᛋ ᚪ ᚾᛖᚳᛖᛋᛋᚪᚱᚣ ᚩᚾᛖ ᚪᛚᚩᛝ ᚦᛖ ᚹᚪᚣ ᚣᚩᚢ ᚹᛁᛚᛚ ᚠᛁᚾᛞ ᚪᚾ ᛖᚾᛞ ᛏᚩ ᚪᛚᛚ ᛋᛏᚱᚢᚷᚷᛚᛖ ᚪᚾᛞ ᛋᚢᚠᚠᛖᚱᛝ ᚣᚩᚢᚱ ᛁᚾᚾᚩᚳᛖᚾᚳᛖ ᚣᚩᚢᚱ ᛁᛚᛚᚢᛋᛡᚾᛋ ᚣᚩᚢᚱ ᚳᛖᚱᛏᚪᛁᚾᛏᚣ ᚪᚾᛞ ᚣᚩᚢᚱ ᚱᛠᛚᛁᛏᚣ ᚢᛚᛏᛁᛗᚪᛏᛖᛚᚣ ᚣᚩᚢ ᚹᛁᛚᛚ ᛞᛁᛋᚳᚩᚢᛖᚱ ᚪᚾ ᛖᚾᛞ ᛏᚩ ᛋᛖᛚᚠ ᛁᛏ ᛁᛋ ᚦᚱᚩᚢᚷᚻ ᚦᛁᛋ ᛈᛁᛚᚷᚱᛁᛗᚪᚷᛖ ᚦᚪᛏ ᚹᛖ ᛋᚻᚪᛈᛖ ᚩᚢᚱᛋᛖᛚᚢᛖᛋ ᚪᚾᛞ ᚩᚢᚱ ᚱᛠᛚᛁᛏᛁᛖᛋ ᛂᚩᚢᚱᚾᛖᚣ ᛞᛖᛖᛈ ᚹᛁᚦᛁᚾ ᚪᚾᛞ ᚣᚩᚢ ᚹᛁᛚᛚ ᚪᚱᚱᛁᚢᛖ ᚩᚢᛏᛋᛁᛞᛖ ᛚᛁᚳᛖ ᚦᛖ ᛁᚾᛋᛏᚪᚱ ᛁᛏ ᛁᛋ ᚩᚾᛚᚣ ᚦᚱᚩᚢᚷᚻ ᚷᚩᛝ ᚹᛁᚦᛁᚾ ᚦᚪᛏ ᚹᛖ ᛗᚪᚣ ᛖᛗᛖᚱᚷᛖ"
    instance_count = 0
    # counts and transition matrices for RUNE only data
    rune_model_counts = {}
    tm_rune_models = {}
    # counts and transition matrices for RUNE and Word Length adn Index in Word data
    wli_model_counts = {}
    tm_wli_models = {}
    # Laplace smoothing means each possible value starts with a count of 1
    # sentence start and end tags means 31 chars in alphabet
    max_counts = {1: 31, 2: 31, 3: 31 * 31, 4: 31 * 31, 5: 31 * 31 * 31 * 31}
    # some "constants" for the phrase being tested, used during optimizations
    phrase_with_spaces = ""  # a full phrase with word spaces
    phrase_no_spaces = ""  # a phrase with spaces removed
    phrase_no_spaces_list = []  # a list of each char in phrase_no_spaces
    phrase_words = []  # words in phrase_with_spaces
    phrase_word_lengths = []  # word lengths for each word in phrase_words
    phrase_a1_values = []  # A term for ng=1 in P(A|B) (requires word lengths)
    phrase_a2_values = []  # A term for ng=2 in P(A|B) (requires word lengths AND phrase_with_spaces )

    #
    def __init__(self, load_data=True):
        if LanguageModel.instance_count == 0 and load_data:
            ts = time.time()
            print("Runeglish LanguageModel Init: load Data", end='')
            self.load_ngram_data()
            self.load_ngram_wli_data()
            print(f" complete, took {round(time.time() - ts, 1)} seconds")
            LanguageModel.instance_count += 1
            self.test_matrices(False)
            print(f"Runeglish LanguageModel Initialized")

    def load_ngram_data(self):
        [self.load_ngram_model(i) for i in [1, 2, 3, 4, 5]]

    def load_ngram_model(self, ng):
        is_1g = ng == 1
        LanguageModel.rune_model_counts[ng] = {}
        self.load_ngram_counts(f'./data/rune_{ng}-grams-from-gng2.csv',
                               LanguageModel.rune_model_counts[ng], is_1g)
        LanguageModel.tm_rune_models[ng] = {}
        if ng == 1:
            self.calculate_transition_matrix(LanguageModel.rune_model_counts[ng],
                                             LanguageModel.tm_rune_models[ng], LanguageModel.max_counts[ng])
        else:
            self.add_transition_matrix(LanguageModel.rune_model_counts[ng], LanguageModel.tm_rune_models[ng],
                                       LanguageModel.max_counts[ng])

    def load_ngram_wli_data(self):
        [self.load_ngram_wli_model(i) for i in [1, 2, 3, 4]]

    def load_ngram_wli_model(self, ng):
        is_1g = ng == 1
        LanguageModel.wli_model_counts[ng] = {}
        self.load_ngram_with_word_data_counts(f'./data/wli_rune_{ng}-grams-from-gng2.csv',
                                              LanguageModel.wli_model_counts[ng], is_1g)
        LanguageModel.tm_wli_models[ng] = {}
        self.add_transition_matrix(LanguageModel.wli_model_counts[ng],
                                   LanguageModel.tm_wli_models[ng], LanguageModel.max_counts[ng])

    def add_transition_matrix(self, ng_dictionary, tm_dictionary, full_tm_data_len):
        for main_key in ng_dictionary.keys():
            tm_dictionary[main_key] = {}
            self.calculate_transition_matrix(ng_dictionary[main_key], tm_dictionary[main_key], full_tm_data_len, )

    def calculate_transition_matrix(self, sparse_counts_dict, tm_dictionary, full_tm_data_len):
        total_counts = sum(sparse_counts_dict.values()) + full_tm_data_len  # Laplace smoothing, +1 to each bin
        tm_dictionary['zero_score'] = math.log(1 / total_counts)  # store zero_score in dict too
        for k2, v2 in sparse_counts_dict.items():
            tm_dictionary[k2] = math.log(v2 / total_counts)

    def load_ngram_counts(self, fn, d, is_onegram=False):
        with open(os.path.abspath(fn), 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for line in reader:
                k, *counts = line
                if k not in d:
                    d[k] = {}
                if is_onegram:
                    d[k] = int(counts[0])
                else:
                    d[k].update((i, int(j)) for i, j in zip( counts[::2], counts[1::2]))

    def load_ngram_with_word_data_counts(self, fn, d, is_onegram=False):
        with open(os.path.abspath(fn), 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            c = 0
            for line in reader:
                c += 1
                k, *counts = line
                if k not in d:
                    d[k] = {}
                if is_onegram:
                    d[k][counts[0]] = int(counts[1])
                else:
                    d[k].update((i, int(j)) for i, j in zip( counts[::2], counts[1::2]))

    def get_logprob_b_given_wli_rune_a(self, b, a, ng):
        model = LanguageModel.tm_wli_models.get(ng)
        if not model:
            return 0
        if a not in model:
            return -40.0
        return model.get(a, {}).get(b, model[a].get('zero_score', -9999))

    def get_logprob_b_given_rune_a(self, b, a=None, ng=1):
        model = LanguageModel.tm_rune_models.get(ng)
        if not model:
            return 0
        if ng == 1:
            return model.get(b, model.get('zero_score', -9999))
        return model.get(a, {}).get(b, model[a].get('zero_score', -9999))

    def get_logprob_b_given_a(self, b, a=None, ng=1, withword=False):
        if withword == True:
            return self.get_logprob_b_given_wli_rune_a(a=a, b=b, ng=ng)
        else:
            return self.get_logprob_b_given_rune_a(a=a, b=b, ng=ng)

    def get_phrase_string_nospace_prob(self, rune_string, ng, withword):
        ''' get the transition probabilities: P(A|B) '''
        # TODO tidy
        if ng == 1 and withword == False:
            b_values = list(rune_string)
            t = [self.get_logprob_b_given_a(b=r) for i, r in enumerate(b_values)]
        elif withword == False:
            b_values = [rune_string[1:][i:i + ng - 1] for i, x in enumerate(rune_string) if i + ng <= len(rune_string)]
            t = [self.get_logprob_b_given_a(a=LanguageModel.phrase_no_spaces_list[i], b=r, ng=ng, withword=withword) for
                 i, r in enumerate(b_values)]
        elif ng == 1 and withword == True:
            b_values = list(rune_string)
            t = [self.get_logprob_b_given_a(a=LanguageModel.phrase_a1_values[i], b=r, ng=ng, withword=withword) for i, r
                 in enumerate(b_values)]
        elif withword == True:
            b_values = [rune_string[1:][i:i + ng - 1] for i, x in enumerate(rune_string) if i + ng <= len(rune_string)]
            t = [self.get_logprob_b_given_a(a=LanguageModel.phrase_a2_values[i], b=r, ng=ng, withword=withword) for i, r
                 in enumerate(b_values)]
        else:
            b_values = [1]
            t = [1]
        ts = sum(t)
        return ts, ts / len(b_values)

    def get_all_ng_rune_score(self):
        return [self.get_phrase_string_nospace_prob(LanguageModel.phrase_no_spaces, i, False) for i in [1, 2, 3, 4, 5]]

    def get_all_ng_wli_rune_score(self):
        return [self.get_phrase_string_nospace_prob(LanguageModel.phrase_no_spaces, i, True) for i in [1, 2, 3, 4]]

    def get_ng_rune_score(self, ng):
        return self.get_phrase_string_nospace_prob(LanguageModel.phrase_no_spaces, ng, False)

    def get_ng_wli_rune_score(self):
        return [self.get_phrase_string_nospace_prob(LanguageModel.phrase_no_spaces, i, True) for i in [1, 2, 3, 4]]

    def set_phrase_no_space(self, rune_string_no_space):
        '''
            During "phrase optimization" in some external optimizer the word lengths will always be the same
            As only the runes are changing call this to set values based on a string of runes with no spaces
            As only the runes are changing call this to set values based on a string of runes ith no spaces
        '''
        LanguageModel.phrase_no_spaces = rune_string_no_space
        LanguageModel.phrase_no_spaces_list = list(LanguageModel.phrase_no_spaces)
        self.set_a2()

    def set_phrase_word_lengths(self, rune_string):
        LanguageModel.words = rune_string.split()
        LanguageModel.phrase_word_lengths = [len(x) for x in self.words]
        # the uni-gram wli A values do not need runes
        self.set_a1()
        self.set_phrase_no_space(rune_string.replace(" ", ""))

    def set_a1(self):
        LanguageModel.phrase_a1_values = []
        for wl in LanguageModel.phrase_word_lengths:
            [LanguageModel.phrase_a1_values.append(f'{wl} {i}') for i in range(wl)]

    def set_a2(self):
        LanguageModel.phrase_a2_values = []
        [LanguageModel.phrase_a2_values.append(f'{i} {r}') for i, r in
         zip(LanguageModel.phrase_a1_values, LanguageModel.phrase_no_spaces_list)]

    def get_phrase_all_prob(self, rune_string):
        self.set_phrase_word_lengths(rune_string)
        rune_score = self.get_all_ng_rune_score()
        wli_rune_score = self.get_ng_wli_rune_score()

        return rune_score, wli_rune_score

    def isclose_list(self, a, b, rel_tol=1e-9, abs_tol=0):
        """Return True if all corresponding elements of a and b are close within tolerances."""
        return all(math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol) for x, y in zip(a, b))

    def test_matrices(self, p=False):
        rune_score, wli_rune_score = self.get_phrase_all_prob(LanguageModel.test_rune_string)
        print(f'*** Internal Test Phrase ***')
        if p:
            print(f'\n\n{LanguageModel.test_rune_string}')
            print(f'Rune only score: {rune_score}')
            [print(f'ng = {x + 1} total_score = {rune_score[x][0]} score /ng = {rune_score[x][1]}') for x in
             range(len(rune_score))]
            print(f'Rune with (W)ord (L)ength and (I)ndex Score: {wli_rune_score}')
            [print(f'ng = {x + 1} total_score = {wli_rune_score[x][0]} score /ng = {wli_rune_score[x][1]}') for x in
             range(len(wli_rune_score))]
        rune_score_ans = [(-1191.7878611168426, -3.0248422870985854), (-1066.7253720710305, -2.7143139238448613),
                          (-1998.3725451647495, -5.097889145828442), (-2843.246316659359, -7.2717297101262375),
                          (-3880.8931908011787, -9.951008181541484)]
        wli_rune_score_ans = [(-1035.3927894109602, -2.627900480738478), (-819.4961746715194, -2.085231996619642),
                              (-1450.6580444766932, -3.700658276726258), (-2184.985785634128, -5.588198940240737)]
        # omf float assertions, order of summation etc can give slightly different results
        c = 0
        for i in range(len(rune_score)):
            c += rune_score[i][0] - rune_score_ans[i][0]
            c += rune_score[i][1] - rune_score_ans[i][1]
        assert c < 0.00001
        c = 0
        for i in range(len(wli_rune_score)):
            c += wli_rune_score[i][0] - wli_rune_score_ans[i][0]
            c += wli_rune_score[i][1] - wli_rune_score_ans[i][1]
        assert c < 0.00001
        print(f'*** Internal Tests PASSED  ***')


if __name__ == "__main__":
    print("run")
    lm = LanguageModel()
    test_rune_text = """ᚦᛖ ᛚᚩᛋᛋ ᚩᚠ ᛞᛁᚢᛁᚾᛁᛏᚣ ᚦᛖ ᚳᛁᚱᚳᚢᛗᚠᛖᚱᛖᚾᚳᛖ ᛈᚱᚪᚳᛏᛁᚳᛖᛋ ᚦᚱᛖᛖ ᛒᛖᚻᚪᚢᛡᚱᛋ ᚹᚻᛁᚳᚻ ᚳᚪᚢᛋᛖ ᚦᛖ ᛚᚩᛋᛋ ᚩᚠ ᛞᛁᚢᛁᚾᛁᛏᚣ ᚳᚩᚾᛋᚢᛗᛈᛏᛡᚾ ᚹᛖ ᚳᚩᚾᛋᚢᛗᛖ ᛏᚩᚩ ᛗᚢᚳᚻ ᛒᛖᚳᚪᚢᛋᛖ ᚹᛖ ᛒᛖᛚᛁᛖᚢᛖ ᚦᛖ ᚠᚩᛚᛚᚩᚹᛝ ᛏᚹᚩ ᛖᚱᚱᚩᚱᛋ ᚹᛁᚦᛁᚾ ᚦᛖ ᛞᛖᚳᛖᛈᛏᛡᚾ ᚹᛖ ᛞᚩ ᚾᚩᛏ ᚻᚪᚢᛖ ᛖᚾᚩᚢᚷᚻ ᚩᚱ ᚦᛖᚱᛖ ᛁᛋ ᚾᚩᛏ ᛖᚾᚩᚢᚷᚻ ᚹᛖ ᚻᚪᚢᛖ ᚹᚻᚪᛏ ᚹᛖ ᚻᚪᚢᛖ ᚾᚩᚹ ᛒᚣ ᛚᚢᚳᚳ ᚪᚾᛞ ᚹᛖ ᚹᛁᛚᛚ ᚾᚩᛏ ᛒᛖ ᛋᛏᚱᚩᛝ ᛖᚾᚩᚢᚷᚻ ᛚᚪᛏᛖᚱ ᛏᚩ ᚩᛒᛏᚪᛁᚾ ᚹᚻᚪᛏ ᚹᛖ ᚾᛖᛖᛞ ᛗᚩᛋᛏ ᚦᛝᛋ ᚪᚱᛖ ᚾᚩᛏ ᚹᚩᚱᚦ ᚳᚩᚾᛋᚢᛗᛝ ᛈᚱᛖᛋᛖᚱᚢᚪᛏᛡᚾ ᚹᛖ ᛈᚱᛖᛋᛖᚱᚢᛖ ᚦᛝᛋ ᛒᛖᚳᚪᚢᛋᛖ ᚹᛖ ᛒᛖᛚᛁᛖᚢᛖ ᚹᛖ ᚪᚱᛖ ᚹᛠᚳ ᛁᚠ ᚹᛖ ᛚᚩᛋᛖ ᚦᛖᛗ ᚹᛖ ᚹᛁᛚᛚ ᚾᚩᛏ ᛒᛖ ᛋᛏᚱᚩᛝ ᛖᚾᚩᚢᚷᚻ ᛏᚩ ᚷᚪᛁᚾ ᚦᛖᛗ ᚪᚷᚪᛁᚾ ᚦᛁᛋ ᛁᛋ ᚦᛖ ᛞᛖᚳᛖᛈᛏᛡᚾ ᛗᚩᛋᛏ ᚦᛝᛋ ᚪᚱᛖ ᚾᚩᛏ ᚹᚩᚱᚦ ᛈᚱᛖᛋᛖᚱᚢᛝ ᚪᛞᚻᛖᚱᛖᚾᚳᛖ ᚹᛖ ᚠᚩᛚᛚᚩᚹ ᛞᚩᚷᛗᚪ ᛋᚩ ᚦᚪᛏ ᚹᛖ ᚳᚪᚾ ᛒᛖᛚᚩᛝ ᚪᚾᛞ ᛒᛖ ᚱᛁᚷᚻᛏ ᚩᚱ ᚹᛖ ᚠᚩᛚᛚᚩᚹ ᚱᛠᛋᚩᚾ ᛋᚩ ᚹᛖ ᚳᚪᚾ ᛒᛖᛚᚩᛝ ᚪᚾᛞ ᛒᛖ ᚱᛁᚷᚻᛏ ᚦᛖᚱᛖ ᛁᛋ ᚾᚩᚦᛝ ᛏᚩ ᛒᛖ ᚱᛁᚷᚻᛏ ᚪᛒᚩᚢᛏ ᛏᚩ ᛒᛖᛚᚩᛝ ᛁᛋ ᛞᛠᚦ ᛁᛏ ᛁᛋ ᚦᛖ ᛒᛖᚻᚪᚢᛡᚱᛋ ᚩᚠ ᚳᚩᚾᛋᚢᛗᛈᛏᛡᚾ ᛈᚱᛖᛋᛖᚱᚢᚪᛏᛡᚾ ᚪᚾᛞ ᚪᛞᚻᛖᚱᛖᚾᚳᛖ ᚦᚪᛏ ᚻᚪᚢᛖ ᚢᛋ ᛚᚩᛋᛖ ᚩᚢᚱ ᛈᚱᛁᛗᚪᛚᛁᛏᚣ ᚪᚾᛞ ᚦᚢᛋ ᚩᚢᚱ ᛞᛁᚢᛁᚾᛁᛏᚣ ᛋᚩᛗᛖ ᚹᛁᛋᛞᚩᛗ ᚪᛗᚪᛋᛋ ᚷᚱᛠᛏ ᚹᛠᛚᚦ ᚾᛖᚢᛖᚱ ᛒᛖᚳᚩᛗᛖ ᚪᛏᛏᚪᚳᚻᛖᛞ ᛏᚩ ᚹᚻᚪᛏ ᚣᚩᚢ ᚩᚹᚾ ᛒᛖ ᛈᚱᛖᛈᚪᚱᛖᛞ ᛏᚩ ᛞᛖᛋᛏᚱᚩᚣ ᚪᛚᛚ ᚦᚪᛏ ᚣᚩᚢ ᚩᚹᚾ ᚪᚾ ᛁᚾᛋᛏᚱᚢᚳᛏᛡᚾ ᛈᚱᚩᚷᚱᚪᛗ ᚣᚩᚢᚱ ᛗᛁᚾᛞ ᛈᚱᚩᚷᚱᚪᛗ ᚱᛠᛚᛁᛏᚣ"""
    rune_score, wli_rune_score = lm.get_phrase_all_prob(test_rune_text)
    print(f'Main Test Phrase: {test_rune_text} ')
    print(f'Rune only score:')
    [print(f'ng = {x} total_score = {rune_score[x][0]} score /ng = {rune_score[x][1]}') for x in range(len(rune_score))]
    print(f'Rune with (W)ord (L)ength and (I)ndex Score:  {wli_rune_score} ')
    [print(f'ng = {x} total_score = {wli_rune_score[x][0]} score /ng = {rune_score[x][1]}') for x in range(len(wli_rune_score))]
    runes = ["ᚠ", "ᚢ", "ᚦ", "ᚩ", "ᚱ", "ᚳ", "ᚷ", "ᚹ", "ᚻ", "ᚾ", "ᛁ", "ᛂ", "ᛇ", "ᛈ",  "ᛉ", "ᛋ", "ᛏ", "ᛒ", "ᛖ", "ᛗ", "ᛚ", "ᛝ",
             "ᛟ", "ᛞ", "ᚪ", "ᚫ", "ᚣ", "ᛡ", "ᛠ"]
    print("Change 1% of runes and see how it reduces the score ")
    s = "".join([random.choice(runes) if r in runes and random.random() < 0.01 else r for r in test_rune_text])
    print(s==test_rune_text)
    rune_score, wli_rune_score = lm.get_phrase_all_prob(s)
    print(f'NEW Test Phrase: {s} ')
    print(f'Rune only score:')
    [print(f'ng = {x} total_score = {rune_score[x][0]} score /ng = {rune_score[x][1]}') for x in range(len(rune_score))]
    print(f'Rune with (W)ord (L)ength and (I)ndex Score:  {wli_rune_score} ')
    [print(f'ng = {x} total_score = {wli_rune_score[x][0]} score /ng = {rune_score[x][1]}') for x in range(len(wli_rune_score))]
