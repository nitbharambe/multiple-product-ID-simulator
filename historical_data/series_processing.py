def get_time(s):

    day, month, residue = s.split('.', 2)
    _, quarter, _ = residue.split(' ', 2)
    h, m = quarter.split(':')
    q = int(h) * 4 + int(m) / 15

    return [int(day), int(month), int(q), int(h), int(m)]
