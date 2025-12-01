#######
# copied from nltk.edit_distance with several changes
#######

def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i           # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j           # row 0: 0,1,2,3,4,...
    return lev


def _edit_dist_step(lev, lev_steps, i, j, s1, s2, transpositions=False):
    c1 = s1[i - 1]
    c2 = s2[j - 1]
    
    # skipping a character in s1
    a = lev[i - 1][j] + 1
    a_steps = lambda: lev_steps[i-1][j] + [{'action': 'skip_s1', 'c1': c1, 'c2': c2, 'i': i-1, 'j': j}]
    # skipping a character in s2
    b = lev[i][j - 1] + 1
    b_steps = lambda: lev_steps[i][j-1] + [{'action': 'skip_s2', 'c1': c1, 'c2': c2, 'i': i, 'j': j-1}]
    # substitution
    c = lev[i - 1][j - 1] + (c1 != c2)
    c_steps = lambda: lev_steps[i-1][j-1] + [{'action': 'sub' if (c1 != c2) else 'no-op', 'c1': c1, 'c2': c2, 'i': i-1, 'j': j-1}]

    # transposition
    d = c + 1  # never picked by default
    d_steps = lambda: c_steps + [{'action': 'transposition', 'c1': c1, 'c2': c2, 'i': i-1, 'j': j-1}]
    if transpositions and i > 1 and j > 1:
        if s1[i - 2] == c2 and s2[j - 2] == c1:
            d = lev[i - 2][j - 2] + 1
            d_steps = lambda: lev_steps[i-2][j-2] + [{'action': 'transposition', 'c1': c1, 'c2': c2, 'i': i-2, 'j': j-2}]

    # pick the cheapest
    cheapest = min((a_steps, a), (b_steps, b), (c_steps, c), (d_steps, d), key=lambda x: x[1])
    lev[i][j] = cheapest[1]
    
    lev_steps[i][j] = cheapest[0]()
    
def edit_distance(s1, s2, transpositions=False):
    """    
    Calculate the Levenshtein edit-distance between two strings.
    The edit distance is the number of characters that need to be
    substituted, inserted, or deleted, to transform s1 into s2.  For
    example, transforming "rain" to "shine" requires three steps,
    consisting of two substitutions and one insertion:
    "rain" -> "sain" -> "shin" -> "shine".  These operations could have
    been done in other orders, but at least three steps are needed.

    This also optionally allows transposition edits (e.g., "ab" -> "ba"),
    though this is disabled by default.

    Algo explanation:
    This is a dynamic programming algorithm, the matrix is of size len(s1) x len(s2).
    Each cell represents the distance between the substrings s1[:i] and s2[:j].
    The distance between the entire s1 and s2 is the value in the bottom right cell (last cell).


    :param s1, s2: The strings to be analysed
    :param transpositions: Whether to allow transposition edits
    :type s1: str
    :type s2: str
    :type transpositions: bool
    :rtype int
    """
    
        
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)
    
    lev_steps = []
    for i in range(len1 + 1):
        lev_steps.append([[]] * (len2 + 1))  # initialize 2D array to zero

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(lev, lev_steps, i + 1, j + 1, s1, s2, transpositions=transpositions)
            
    return lev[len1][len2], lev_steps[len1][len2]
