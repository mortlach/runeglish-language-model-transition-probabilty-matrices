# Runeglish-language-model-transition-matrices
 Transition matrices for character ngrams in Runeglish 
 
 Using Markov's principal, P(A|B), Transition Probability Matrices are calculated for Runeglish characters.
 Where A is: 
 a single rune, i.e. 1 of ᚠᚢᚦᚩᚱᚳᚷᚹᚻᚾᛁᛂᛇᛈᛉᛋᛏᛒᛖᛗᛚᛝᛟᛞᚪᚫᚣᛡᛠ, or word length + index in word + single rune, i.e. 7 3 ᚣ for the 4th char of a 7 runes word
 and B is the next 1 2 3 or 4 runes in the phrase 
 
 Example phrase probability calculation is given, and how the scores are reduced as 1% error is added