import string
import re



# Function to remove spaces and punctuation
def remove_spaces_and_punctuation(text):
    cleaned = re.sub(r'[' + string.punctuation + string.whitespace + '\s]', '', text)
    cleaned = ''.join([char for char in cleaned if char.isalnum()])
    
    return cleaned

def find_substring(s, sb):
    
    # if text appears as it is - return its indices
    if sb.lower().strip() in s.lower():
        start_index = s.lower().index(sb.lower().strip())
        return start_index, start_index + len(sb.strip())

    # Remove spaces and punctuation from both strings
    modified_s = remove_spaces_and_punctuation(s).lower()
    modified_sb = remove_spaces_and_punctuation(sb).lower()

    # Find the modified substring in the modified string
    index = modified_s.find(modified_sb)

    # If substring is not found, return -1 for both start and end
    if index == -1:
        return -1, -1

    # Find the actual start index in the original string
    actual_start_index = 0
    count = 0
    for char in s:
        is_char_removed = remove_spaces_and_punctuation(char) == ''
        if count == index and not is_char_removed: # the second term - to not stop count when getting to a space/punctuation
            break
        if not is_char_removed:
            count += 1
        actual_start_index += 1

    # Find the actual end index in the original string
    actual_end_index = actual_start_index
    modified_sb_length = len(modified_sb)
    while modified_sb_length > 0:
        if s[actual_end_index].isalnum():
            modified_sb_length -= 1
        actual_end_index += 1

    assert remove_spaces_and_punctuation(sb.lower())==remove_spaces_and_punctuation(s[actual_start_index:actual_end_index].lower()), "found substring doesn't match indices" # make sure span align with sb
    return actual_start_index, actual_end_index
