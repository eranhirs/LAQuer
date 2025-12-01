def replace_json_key_with_valid_json_end(decoded: str) -> str:
    """
    Given a decoded string that is supposed to be a JSON object but may be incomplete,
    this function attempts to fix it by ensuring it ends with a proper JSON ending.
    """
    
    did_model_genreate_json_end = "}" in decoded
    if did_model_genreate_json_end:
        # remove text after }
        decoded = decoded.split("}")[0] + "}"
        return decoded.strip()
    else:
        is_json_ends_with_comma = decoded.replace("\n", "").strip().endswith(",")
        if is_json_ends_with_comma:
            return ",".join(decoded.split(",")[:-1]) + "}"
        else:
            is_json_ends_with_quotes = decoded.replace("\n", "").strip().endswith("\"")
            is_json_ends_with_null = decoded.replace("\n", "").strip().endswith("null")
            if is_json_ends_with_quotes or is_json_ends_with_null:
                return decoded + "}"
            else:
                return decoded + "\"}"

