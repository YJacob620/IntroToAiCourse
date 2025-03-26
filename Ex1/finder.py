class Finder:
    def __init__(self, _list: list):
        self.data = _list

    def find(self) -> list:
        ans: list = []
        for item in self.data:
            if "AI" in item:
                ans.append(item)
        return ans

    def getData(self):
        return self.data
