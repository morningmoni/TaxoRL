import codecs

from docopt import docopt
from spacy.en import English
from collections import defaultdict


def main():
    """
    Creates a "knowledge resource" from triplets file
    """

    # Get the arguments
    args = docopt("""Parse the Wikipedia dump and create a triplets file, each line is formatted as follows: X\t\Y\tpath

    Usage:
        parse_wikipedia.py <wiki_file> <vocabulary_file> <out_file>

        <wiki_file> = the Wikipedia dump file
        <vocabulary_file> = a file containing the words to include
        <out_file> = the output file
    """)

    nlp = English()

    wiki_file = args['<wiki_file>']
    vocabulary_file = args['<vocabulary_file>']
    out_file = args['<out_file>']

    # Load the phrase pair files
    with codecs.open(vocabulary_file, 'r', 'utf-8') as f_in:
        vocabulary = set([line.strip() for line in f_in])

    with codecs.open(wiki_file, 'r', 'utf-8') as f_in:
        with codecs.open(out_file, 'w', 'utf-8') as f_out:

            # Read the next paragraph
            for paragraph in f_in:

                # Skip empty lines
                paragraph = paragraph.strip()
                if len(paragraph) == 0:
                    continue

                parsed_par = nlp(unicode(paragraph))

                # Parse each sentence separately
                for sent in parsed_par.sents:
                    dependency_paths = parse_sentence(sent, vocabulary)
                    if len(dependency_paths) > 0:
                        for (x, y), paths in dependency_paths.iteritems():
                            for path in paths:
                                print >> f_out, '\t'.join([x, y, path])


def parse_sentence(sent, vocabulary):
    """
    Get all the dependency paths between nouns in the sentence
    :param sent: the sentence to parse
    :param vocabulary: the words to search paths for
    :return: the list of entities and paths
    """

    # Get all term indices
    indices = [(token.lemma_, sent[i:i+1], i, i) for i, token in enumerate(sent)
               if len(token.orth_) > 2 and token.lemma_ in vocabulary and token.pos_ in ['NOUN', 'VERB', 'ADJ']]

    # Add noun chunks for the current sentence
    # Don't include noun chunks with only one word - these are nouns already included
    indices.extend([(np.orth_, np, np.start, np.end) for np in sent.doc.noun_chunks
                    if sent.start <= np.start < np.end - 1 < sent.end and np.orth_ in vocabulary])

    tokens = [(x[0], x[1], y[0], y[1]) for x in indices for y in indices if x[3] < y[2]]

    # Get all dependency paths between words, up to length 4
    paths = defaultdict(list)
    [paths[(x, y)].append(shortest_path((x_tokens, y_tokens))) for (x, x_tokens, y, y_tokens) in tokens]

    satellites = defaultdict(list)
    [satellites[(x, y)].extend([sat_path for path in paths[(x, y)] for sat_path in get_satellite_links(path)
                                if sat_path is not None]) for (x, y) in paths.keys()]

    filtered_paths = defaultdict(list)
    [filtered_paths[(x, y)].extend(filter(None, [pretty_print(set_x_l, x, set_x_r, hx, lch, hy, set_y_l, y, set_y_r)
                                                 for (set_x_l, x, set_x_r, hx, lch, hy, set_y_l, y, set_y_r)
                                                 in satellites[(x, y)]]))
     for (x, y) in satellites.keys()]

    return filtered_paths


def shortest_path(path):
    """ Returns the shortest dependency path from x to y
    :param x: x token
    :param y: y token
    :return: the shortest dependency path from x to y
    """

    # Get the root token and work on it
    if path is None:
        return None

    x_tokens, y_tokens = path
    x_token = x_tokens.root
    y_token = y_tokens.root

    # Get the path from the root to each of the tokens
    hx = heads(x_token)
    hy = heads(y_token)

    # Get the lowest common head

    # There could be several cases. For example, for x = parrot, y = bird:

    # 1. x is the head of y: "[parrot] and other [birds]"
    if hx == [] and x_token in hy:
        hy = hy[:hy.index(x_token)]
        hx = []
        lch = x_token

    # 2. y is the head of x: "[birds] such as [parrots]"
    elif hy == [] and y_token in hx:
        hx = hx[:hx.index(y_token)]
        hy = []
        lch = y_token

    elif len(hx) == 0 or len(hy) == 0:
        return None

    # 3. x and y have no common head - the first head in each list should be the sentence root, so
    # this is possibly a parse error?
    elif hy[0] != hx[0]:
        return None

    # 4. x and y are connected via a direct parent or have the exact same path to the root, as in "[parrot] is a [bird]"
    elif hx == hy:
        lch = hx[-1]
        hx = hy = []

    # 5. x and y have a different parent which is non-direct, as in "[parrot] is a member of the [bird] family".
    # The head is the last item in the common sequence of both head lists.
    else:
        for i in xrange(min(len(hx), len(hy))):
            # Now we've found the common ancestor in i-1
            if hx[i] is not hy[i]:
                break

        if len(hx) > i:
            lch = hx[i-1]
        elif len(hy) > i:
            lch = hy[i-1]
        else:
            return None

        # The path from x to the lowest common head
        hx = hx[i+1:]

        # The path from the lowest common head to y
        hy = hy[i+1:]

    if lch and check_direction(lch, hx, lambda h: h.lefts):
        return None
    hx = hx[::-1]

    if lch and check_direction(lch, hy, lambda h: h.rights):
        return None

    return (x_token, hx, lch, hy, y_token)


def heads(token):
    """
    Return the heads of a token, from the root down to immediate head
    :param token:
    :return:
    """
    t = token
    hs = []
    while t is not t.head:
        t = t.head
        hs.append(t)
    return hs[::-1]


def check_direction(lch, hs, f_dir):
    """
    Make sure that the path between the term and the lowest common head is in a certain direction
    :param lch: the lowest common head
    :param hs: the path from the lowest common head to the term
    :param f_dir: function of direction
    :return:
    """
    return any(modifier not in f_dir(head) for head, modifier in zip([lch] + hs[:-1], hs))


def get_satellite_links(path):
    """
    Add the "satellites" - single links not already contained in the dependency path added on either side of each noun
    :param x: the X token
    :param y: the Y token
    :param hx: X's head tokens
    :param hy: Y's head tokens
    :param lch: the lowest common ancestor of X and Y
    :return: more paths, with satellite links
    """
    if path is None:
        return []

    x_tokens, hx, lch, hy, y_tokens = path
    paths = [(None, x_tokens, None, hx, lch, hy, None, y_tokens, None)]
    tokens_on_path = set([x_tokens] + hx + [lch] + hy + [y_tokens])

    # Get daughters of x not in the path
    set_xs = [(child, child.idx) for child in x_tokens.children if child not in tokens_on_path]
    set_ys = [(child, child.idx) for child in y_tokens.children if child not in tokens_on_path]

    x_index = x_tokens.idx
    y_index = y_tokens.idx

    for child, idx in set_xs:
        if child.tag_ != 'PUNCT' and len(child.string.strip()) > 1:
            if idx < x_index:
                paths.append((child, x_tokens, None, hx, lch, hy, None, y_tokens, None))
            else:
                paths.append((None, x_tokens, child, hx, lch, hy, None, y_tokens, None))

    for child, idx in set_ys:
        if child.tag_ != 'PUNCT' and len(child.string.strip()) > 1:
            if idx < y_index:
                paths.append((None, x_tokens, None, hx, lch, hy, child, y_tokens, None))
            else:
                paths.append((None, x_tokens, None, hx, lch, hy, None, y_tokens, child))

    return paths


def edge_to_string(t, is_head=False):
    """
    Converts the token to an edge string representation
    :param token: the token
    :return: the edge string
    """
    return '/'.join([t.lemma_.strip().lower(), t.pos_, t.dep_ if t.dep_ != '' and not is_head else 'ROOT'])


def argument_to_string(token, edge_name):
    """
    Converts the argument token (X or Y) to an edge string representation
    :param token: the X or Y token
    :param edge_name: 'X' or 'Y'
    :return:
    """
    return '/'.join([edge_name, token.pos_, token.dep_ if token.dep_ != '' else 'ROOT'])


def direction(dir):
    """
    Print the direction of the edge
    :param dir: the direction
    :return: a string representation of the direction
    """
    # Up to the head
    if dir == UP:
        return '>'
    # Down from the head
    elif dir == DOWN:
        return '<'
    elif dir == SAT:
        return 'V'
    else:
        return '^'


def pretty_print(set_x_l, x, set_x_r, hx, lch, hy, set_y_l, y, set_y_r):
    """
    Filter out long paths and pretty print the short ones
    :return: the string representation of the path
    """
    set_path_x_l = []
    set_path_x_r = []
    set_path_y_r = []
    set_path_y_l = []
    lch_lst = []

    if set_x_l:
        set_path_x_l = [edge_to_string(set_x_l) + '/' + direction(SAT)]
    if set_x_r:
        set_path_x_r = [edge_to_string(set_x_r) + '/' + direction(SAT)]
    if set_y_l:
        set_path_y_l = [edge_to_string(set_y_l) + '/' + direction(SAT)]
    if set_y_r:
        set_path_y_r = [edge_to_string(set_y_r) + '/' + direction(SAT)]

    # X is the head
    if lch == x:
        dir_x = direction(ROOT)
        dir_y = direction(DOWN)
    # Y is the head
    elif lch == y:
        dir_x = direction(UP)
        dir_y = direction(ROOT)
    # X and Y are not heads
    else:
        lch_lst = [edge_to_string(lch, is_head=True) + '/' + direction(ROOT)] if lch else []
        dir_x = direction(UP)
        dir_y = direction(DOWN)

    len_path = len(hx) + len(hy) + len(set_path_x_r) + len(set_path_x_l) + \
               len(set_path_y_r) + len(set_path_y_l) + len(lch_lst)

    if len_path <= MAX_PATH_LEN:
        cleaned_path = '_'.join(set_path_x_l + [argument_to_string(x, 'X') + '/' + dir_x] + set_path_x_r +
                                [edge_to_string(token) + '/' + direction(UP) for token in hx] +
                                lch_lst +
                                [edge_to_string(token) + '/' + direction(DOWN) for token in hy] +
                                set_path_y_l + [argument_to_string(y, 'Y') + '/' + dir_y] + set_path_y_r)
        return cleaned_path
    else:
        return None


# Constants
MAX_PATH_LEN = 4
ROOT = 0
UP = 1
DOWN = 2
SAT = 3

if __name__ == '__main__':
    main()