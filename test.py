import os
import copy
from functools import reduce 

# class Person:  
#     def __init__(self, first_name, email):
#         self.first_name = first_name
#         self._email = email
    
#     def update_email(self,new_email):
#         self._email = new_email 

#     def email(self):        
#         return self._email

# th = Person('TH', 'tk@mail.com')
# print(th.email)

# th._email = 'new_tk@mail.com'
# print(th.email)


# a = 3 
# b = 3
# print(id(a))
# print(id(b))

# list = [1,2, [3,4]]
# a = copy.copy(list)
# b= copy.deepcopy(list)
# print(id(list))
# print(id(a))
# print(id(b))

# a = 'cheesezh'
# b = 'cheesezh'
# print(id(a))
# print(id(b))

# class A:
#     def __init__(self):
#        self.name = 'twiss'
#        #self.age='24'
#     def method(self):
#        print("method print")

# Instance = A()
# print(getattr(Instance , 'name', 'not find'))   #如果Instance 对象中有属性name则打印self.name的值,否则打印'not find'

# print (getattr(Instance , 'age', 'not find'))  #如果Instance 对象中有属性age，则打印self.age的值，否则打印'not find'

# print (getattr(Instance, 'method', 'default')) #如果有方法method，否则打印其地址，否则打印default

# print (getattr(Instance, 'method', 'default')) #如果有方法method，运行函数并打印None，否则打印default

# print(reduce(lambda x,y:x*y,range(1,4)))

# a = [1, 2, 3, 4, 5]
# b = ['a', 'b', 'c']
# c = ['A', 'B', 'C', 'D', 'E']
# z = zip(a, b, c)
# print(list(z))

# class Duck:
#     def say(self):
#         print("嘎嘎")
# class Dog:
#     def say(self):
#         print("汪汪")
# def speak(duck):
#     duck.say()

# duck = Duck()
# dog = Dog()
# speak(duck) # 嘎嘎
# speak(dog) # 汪汪


# def flist(l):
#     l.append(0)
#     print(id(l))    # 每次打印的id相同
#     print(l)

# ll = []
# print(id(ll))
# flist(ll)   # [0]
# flist(ll)   # [0,0]
# print("=" * 10)

# def fstr(s):
#     print(id(s)) # 和入参ss的id相同
#     s += "a"
#     print(id(s))  # 和入参ss的id不同，每次打印结果不相同
#     print(s)

# ss = "sun"
# print(id(ss))
# fstr(ss)    # a
# fstr(ss)    # a

# def clear_list(l):
#     l = []
# ll = [1,2,3]
# clear_list(ll)
# print(ll)
# def fl(l=[1]):
#     l.append(1)
#     print(l)
# fl()
# fl()

# a = 10
# b = 5
# c = a//b
# print ("7 - c 的值为：", c)

# L = [0, 1, 3, 4, 5,7]
# L.sort()
# l_len = len(L)
# n = (l_len - 1) // 2 #向下取整

# print(n)


# if l_len & 0x1:
#     print(L[n])
# else:
#     print("%.1f" %((L[n] + L[n+1]) / 2.0))

# def multi():
#     return [lambda x : i*x for i in range(4)]
# print([m(3) for m in multi()])

# class Person():
#     def __init__(self,name):
#         self.name = name
#         print('Person')
 
# class Male(Person):
#     def __init__(self,age):
#         # super().__init__('xiaoming')
#         super(Male, self).__init__('xiaoming')
#         self.age = age
#         print("Male")
 
# m = Male(12)
# print(m.__dict__)

# class Animal():
#     def __init__(self,name):
#         self.name = name

# class Person(Animal):
#     def __init__(self,name,age):
#         super().__init__(name)
#         self.age = age
#         print('Person')

# class Male(Person):
#     def __init__(self,name,age):
#         super(Person,self).__init__(name,age)
#         print("Male")

# m = Male('xiaoming',12)
# super(Male,m).__init__('xiaoming',12)
# print(m.__dict__)


class Animal: 
    def eat(self): 
        print("---吃-----") 
    def drink(self): 
        print("----喝-----")
    def sleep(self): 
        print("----睡觉-----") 
        
class Dog(Animal): 
    def bark(self): 
        print("---汪汪叫----")

class Xiaotq(Dog):
    def fly(self): 
        print("----飞-----")
    def bark(self): 
        print("----狂叫-----") 
        #调用被重写的父类的方法 
        # #1 必须加上# self 
        Dog.bark(self) 
        # #2 
        super().bark() 
        
xiaotq = Xiaotq() 
xiaotq.fly() 
xiaotq.bark()
xiaotq.eat()