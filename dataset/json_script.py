import json
import codecs

def parse_json(json_data):
    """
    Parse json file returned from Microsoft Entity Linking system

    Return a list of (mention_start, mention_end, mention_text) pairs
    """
    pairs = []
    jobj = json.loads(json_data)
    for entity in jobj["entities"]:
        mentions = entity["matches"]
        for mention in mentions:
            mention_text = mention["text"]
            mention_len = len(mention_text)
            entries = mention["entries"]
            for entry in entries:
                offset = entry["offset"]
                pairs.append((offset, offset+mention_len, mention_text))
    return pairs