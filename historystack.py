class HistoryStask:
    def __init__(self, n):
        self._n = n
        self._list_of_objects = [[] for i in range(n)]

    def __getitem__(self, i):
        return self._list_of_objects[i]

    def len(self, i):
        return len(self[i])

    def push(self, i, e):
        self[i].append(e.copy())

    def pop(self, i):
        return self[i].pop()
