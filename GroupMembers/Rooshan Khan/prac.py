# class A(object):
#     def __init__(self):
#         self.x = 2
    
#     def a1(self):
#         """ This is an instance method. """
#         print("Hello from an instance of A\n" * self.x, "      ")

#     @classmethod
#     def a2(cls):
#         """ This is a classmethod. """
#         print("Hello from class A")

# class B(object):
#     def __init__(self):
#         pass
    
#     def b1(self):
#         print(A().a1())  # => prints 'Hello from an instance of A'
#         A.a2()           # Call the class method directly using the class

# b = B()
# b.b1()
import torch
# Tensor1=torch.rand(1,2,3,4)
# Tensor2=Tensor1
# print(Tensor1.shape,Tensor1)
# Tensor1=Tensor1.transpose(1,2)
# print(Tensor1.shape,Tensor1)
# Tensor1=Tensor1.contiguous()
# print(Tensor1.shape,Tensor1)
# # print(key[0,0].size()[0],"*",key[0,0,0].size()[0],'=',key[0,0].size()[0]*key[0,0,0].size()[0])
# Tensor3=Tensor1.view(Tensor1.size()[0],Tensor1[0].size()[0],Tensor1[0,0].size()[0]*Tensor1[0,0,0].size()[0])
# print(Tensor1.shape,Tensor1)
# Tensor2=Tensor2.view(Tensor1.size()[0],Tensor1[0].size()[0],Tensor1[0,0].size()[0]*Tensor1[0,0,0].size()[0])
# print(Tensor2.shape,Tensor2)

# print(Tensor3==Tensor2)

# print(torch.zeros(Tensor3.shape))

m1 = torch.tensor([[1,2],[3,4]])
m2 = torch.tensor([[1,2],[3,4]])
m3=m1==m2
a = torch.tensor([[True,True],[True,True]])
print(m3)
print(torch.any(~m3))
print(torch.any(~(m1==m2)).item()) # We use `tensor.item()` in Python to convert a 0-dim tensor to a number