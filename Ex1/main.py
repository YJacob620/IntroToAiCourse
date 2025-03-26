from finder import Finder

if __name__ == '__main__':
    my_list = ["item1AI", "item2", "item3AI", "item4", "1hvbinAI156"]

    finder: Finder = Finder(my_list)
    my_list_dup = finder.getData()
    my_list_AI = finder.find()

    print(my_list_dup)
    print(my_list_AI)
