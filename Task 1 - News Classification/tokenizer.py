from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

NLTK_STOP_WORDS = set(stopwords.words('english'))
SKLEARN_STOP_WORDS = ENGLISH_STOP_WORDS

# Merge stop words from ntlk and sklearn
STOP_WORDS = NLTK_STOP_WORDS.union(SKLEARN_STOP_WORDS)

from nltk.tokenize import word_tokenize
import re

def word_tokenizer(text:str)->list:
  """
  Tokenize given text using NLTK's word tokenizer
  with normalizing (lowercasing string) and filtering 
  stop words, numbers and punctuation marks.

  Parameters
  ----------
  text : str
    Text to be tokenized

  Returns
  ----------
  list
    List of tokens
  """
  tokens = word_tokenize(text.lower())

  tokens_to_return = list()
  for token in tokens:
    token = token.strip("'")

    # Filter number
    if (re.match(r"^[\d.]+$", token)): 
      continue
    # Filter punctuation mark and stop word
    elif (re.match(r"[\w'-]+", token) and (token not in ['-',"'"]) 
    and (token not in STOP_WORDS)):
      tokens_to_return.append(token)

  return tokens_to_return


# Option1: NLTK PorterStemmer
from nltk.stem.porter import PorterStemmer

def porter_stem_tokenizer(text:str)->list:
  """
  Tokenize given text using custom word tokenizer
  (based on NLTK word tokenizer) with Porter stemmer

  Parameters
  ----------
  text : str
    Text to be tokenized

  Returns
  ----------
  list
    List of tokens
  """
  tokens = word_tokenizer(text)
  stemmer = PorterStemmer()
  stems = list()
  for token in tokens:
    stems.append(stemmer.stem(token))
  return stems

# Option2: NLTK SnowballStemmer
from nltk.stem.snowball import EnglishStemmer as SnowballStemmer

def snowball_stem_tokenizer(text:str)->list:
  """
  Tokenize given text using custom word tokenizer
  (based on NLTK word tokenizer) with Snowball stemmer

  Parameters
  ----------
  text : str
    Text to be tokenized

  Returns
  ----------
  list
    List of tokens
  """
  tokens = word_tokenizer(text)
  stemmer = SnowballStemmer()
  stems = list()
  for token in tokens:
      stems.append(stemmer.stem(token))
  return stems

# Option3: NLTK LancasterStemmer
from nltk.stem.lancaster import LancasterStemmer

def lancaster_stem_tokenizer(text):
  """
  Tokenize given text using custom word tokenizer
  (based on NLTK word tokenizer) with Lancaster stemmer

  Parameters
  ----------
  text : str
    Text to be tokenized

  Returns
  ----------
  list
    List of tokens
  """
  tokens = word_tokenizer(text)
  stemmer = LancasterStemmer()
  stems = list()
  for token in tokens:
    stems.append(stemmer.stem(token))
  return stems

# Option4: NLTK WordNetLemmatizer
from nltk.stem import WordNetLemmatizer

def wordnet_lemma_tokenizer(text:str)->list:
  """
  Tokenize given text using custom word tokenizer
  (based on NLTK word tokenizer) with Wordnet lemmatizer

  Parameters
  ----------
  text : str
    Text to be tokenized

  Returns
  ----------
  list
    List of tokens
  """
  tokens = word_tokenizer(text)
  lemmatizer = WordNetLemmatizer()
  lemmas = list()
  for token in tokens:
    lemmas.append(lemmatizer.lemmatize(token))
  return lemmas

# Option5: NLTK WordNet Lemmatizer with POS
from nltk.tag import pos_tag

def convert_tag(tag:str)->str:
  """
  Convert part-of-speech tag to tag compatible 
  with WordNet lemmatizer.

  Parameters
  ----------
  tag : str
    Text to be tokenized

  Returns
  ----------
  str
    Part-of-speech tag compatible with WordNet lemmatizer; 
    "n" for noun, "v" for verb, "a" for adjective and "r" for adverb
  """
  if tag[0] == 'V':
    return 'v'
  elif tag[0] == 'J':
    return 'a'
  elif tag[0] == 'R':
    return 'r'
  else:
    return 'n'

def wordnet_lemma_pos_tokenizer(text:str)->list:
  """
  Tokenize given text using custom word tokenizer
  (based on NLTK word tokenizer) with Wordnet lemmatizer
  with predicting word's part-of-speech

  Parameters
  ----------
  text : str
    Text to be tokenized

  Returns
  ----------
  list
    List of tokens
  """
  tokens = word_tokenizer(text)
  lemmatizer = WordNetLemmatizer()
  lemmas = list()
  tokens_with_pos_tag = pos_tag(tokens)
  for token in tokens_with_pos_tag:
    word = token[0]
    pos = convert_tag(token[1])
    lemmas.append(lemmatizer.lemmatize(word, pos=pos))
  return lemmas
