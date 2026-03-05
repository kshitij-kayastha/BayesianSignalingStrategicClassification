def set_partitions(collection):
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in set_partitions(collection[1:]):
        for i in range(len(smaller)):
            yield smaller[:i] + [[first] + smaller[i]] + smaller[i+1:]
        yield [[first]] + smaller


def display_queue(Q, P):
    res = "[  "
    for (a_id, b_id) in Q:
        res += f"({P[a_id]}, {P[b_id]})  "
    res += "]"
    print(res)
