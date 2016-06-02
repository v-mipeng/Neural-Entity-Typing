class BaseClass(object):
    def __init__(self):
        print("base init")

    def test(self):
        self.sub_fun()

    def sub_fun(self, value = True):
        print("in base fun")

class SubClass(BaseClass):
    def __init__(self):
        super(SubClass, self).__init__()

    def sub_fun(self, value = True):
        print("in sub fun")

