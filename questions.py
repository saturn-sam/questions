import math
import nltk
import sys
import os
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_content = dict()
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), encoding="utf8") as f:
            file_content[filename] = f.read()

    return file_content


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    word_list=[]
    tokenized_words = nltk.tokenize.word_tokenize(document.lower())

    for word in tokenized_words:
        if word not in string.punctuation and word not in nltk.corpus.stopwords.words("english"):
            word_list.append(word)

    return word_list


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """


    idfs_dict = {}
     
    total_documents = len(documents)

    for word in sum(documents.values(), start=[]):
        count_of_word = 0
        for doc in documents.values():
            if word in doc:
                count_of_word += 1

        idfs_dict[word] = math.log(total_documents/count_of_word)

    return idfs_dict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_scores = {}
    for file, content in files.items():
        score = 0
        for word in query:
            if word in content:
                score += content.count(word) * idfs[word]
        if score != 0:
            file_scores[file] = score

    files_sorted_by_score = [sfile for sfile, scfile in sorted(file_scores.items(), key=lambda value: value[1], reverse=True)]
    return files_sorted_by_score[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sen_scores = {}
    for sentence, words in sentences.items():
        score = 0
        for word in query:
            if word in words:
                score += idfs[word]
        if score != 0:
            density = sum([words.count(x) for x in query]) / len(words)
            sen_scores[sentence] = (score, density)

    sentence_sorted_by_score = [ssen for ssen, scsen in sorted(sen_scores.items(), key=lambda value: (value[1][0], value[1][1]), reverse=True)]

    return sentence_sorted_by_score[:n]


if __name__ == "__main__":
    main()
