
def match_strength(longdesc, text_str):
    text_list = text_str.split('$$$')
    num_of_attribs = len(text_list)
    in_count = 0
    for attrib in text_list:
        attrib = attrib.replace(" ", "")
        attrib = attrib.replace(",", "")
        longdesc = longdesc.replace(" ", "")
        longdesc = longdesc.replace(",", "")

        if attrib.upper() in longdesc.upper():
            in_count = in_count + 1
    return in_count / num_of_attribs


def max_match(match_dict):
    return max(match_dict, key=match_dict.get)


def search_text(text,df):
    """
    takes text as input and resturns matched key
    :param text:
    :param df:
    :return:
    """
    distance_dict = {}
    for i, each_sent in enumerate(df['long description'].tolist()):
        distance_dict[df.loc[i]['Image']] = match_strength(each_sent, text)
    match_key = max_match(distance_dict)
    print(match_key)
    res = [match_key]
    return res