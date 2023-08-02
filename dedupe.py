def dedupe_hashable(items):
    """
    a = [1, 5, 2, 1, 9, 1, 5, 10]
    list(dedupe_hashable(a))
    [1, 5, 2, 9, 10]
    :param items:
    :return:
    """
    seen = set()
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)


def dedupe_map(items, key=None):
    seen = set()
    for item in items:
        val = item if key is None else key(item)
        if val not in seen:
            yield item
            seen.add(val)
