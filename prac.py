class A(object):
    def __init__(self):
        self.x = 2
    
    def a1(self):
        """ This is an instance method. """
        print("Hello from an instance of A\n" * self.x, "      ")

    @classmethod
    def a2(cls):
        """ This is a classmethod. """
        print("Hello from class A")

class B(object):
    def __init__(self):
        pass
    
    def b1(self):
        print(A().a1())  # => prints 'Hello from an instance of A'
        A.a2()           # Call the class method directly using the class

b = B()
b.b1()