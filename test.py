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

# class Animal: 
#     def eat(self): 
#         print("---吃-----") 
#     def drink(self): 
#         print("----喝-----")
#     def sleep(self): 
#         print("----睡觉-----") 
        
# class Dog(Animal): 
#     def bark(self): 
#         print("---汪汪叫----")

# class Xiaotq(Dog):
#     def fly(self): 
#         print("----飞-----")
#     def bark(self): 
#         print("----狂叫-----") 
#         #调用被重写的父类的方法 
#         # #1 必须加上# self 
#         Dog.bark(self) 
#         # #2 
#         super().bark() 
        
# xiaotq = Xiaotq() 
# xiaotq.fly() 
# xiaotq.bark()
# xiaotq.eat()

# generator = (i for i in range(1,5))
# print(next(generator))
# print(next(generator))
# for i in generator:
#     print(i)
# for i in generator:
#     print(i)


# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# from PIL import Image
# import os
# from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
# from torchvision.utils import make_grid

# writer = SummaryWriter("logs")

# class MyData(Dataset):

#     def __init__(self, root_dir, image_dir, label_dir, transform):
#         self.root_dir = root_dir
#         self.image_dir = image_dir
#         self.label_dir = label_dir
#         self.label_path = os.path.join(self.root_dir, self.label_dir)
#         self.image_path = os.path.join(self.root_dir, self.image_dir)
#         self.image_list = os.listdir(self.image_path)
#         self.label_list = os.listdir(self.label_path)
#         self.transform = transform
#         # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的 image和 label是一一对应的
#         self.image_list.sort()
#         self.label_list.sort()

#     #一般的dataset都要重新__getitem__ 和 __len__
#     #__getitem__用于把image和标签组成一队并返回
#     #__len__用于返回整个数据集的长度
#     def __getitem__(self, idx):
#         img_name = self.image_list[idx]
#         label_name = self.label_list[idx]
#         img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
#         label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
#         img = Image.open(img_item_path)

#         with open(label_item_path, 'r') as f:
#             label = f.readline()

#         # img = np.array(img)
#         img = self.transform(img)
#         sample = {'img': img, 'label': label}
#         return sample

#     def __len__(self):
#         assert len(self.image_list) == len(self.label_list)
#         return len(self.image_list)

# if __name__ == '__main__':
#     transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
#     root_dir = "dataset/train"
#     image_ants = "ants_image"   #为类别文件夹，文件夹下面存放的是同一类的数据
#     label_ants = "ants_label"
#     ants_dataset = MyData(root_dir, image_ants, label_ants, transform)
#     image_bees = "bees_image"
#     label_bees = "bees_label"
#     bees_dataset = MyData(root_dir, image_bees, label_bees, transform)
#     train_dataset = ants_dataset + bees_dataset

#     # transforms = transforms.Compose([transforms.Resize(256, 256)])
#     dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2)

#     writer.add_image('error', train_dataset[119]['img'])
#     writer.close()
#     # for i, j in enumerate(dataloader):
#     #     # imgs, labels = j
#     #     print(type(j))
#     #     print(i, j['img'].shape)
#     #     # writer.add_image("train_data_b2", make_grid(j['img']), i)
#     #
#     # writer.close()

# class Test:

#     def __init__(self, foo):
#         self.__foo = foo

#     def __bar(self):
#         print(self.__foo)
#         print('__bar')


# def main():
#     test = Test('hello')
#     # AttributeError: 'Test' object has no attribute '__bar'
#     #test.__bar()
#     # AttributeError: 'Test' object has no attribute '__foo'
#     print(test.__foo)


# if __name__ == "__main__":
#     main()


# class Test:

#     def __init__(self, foo):
#         self.__foo = foo

#     def __bar(self):
#         print(self.__foo)
#         print('__bar')


# def main():
#     test = Test('hello')
#     test._Test__bar()
#     print(test._Test__foo)


# if __name__ == "__main__":
#     main()

# class Person(object):

#     def __init__(self, name, age):
#         self._name = name
#         self._age = age

#     # 访问器 - getter方法
#     @property
#     def name(self):
#         return self._name

#     # 访问器 - getter方法
#     @property
#     def age(self):
#         return self._age

#     # 修改器 - setter方法
#     @age.setter
#     def age(self, age):
#         self._age = age

#     def play(self):
#         if self._age <= 16:
#             print('%s正在玩飞行棋.' % self._name)
#         else:
#             print('%s正在玩斗地主.' % self._name)


# def main():
#     person = Person('王大锤', 12)
#     person.play()
#     person.age = 22
#     person.play()
#     # person.name = '白元芳'  # AttributeError: can't set attribute


# if __name__ == '__main__':
#     main()

# a = (1,2,3)    # tuple
# b = (1,2,3)
# print(id(a))
# print(id(b))、

# name = 'Chris'
# 1. f stringsprint(f'Hello {name}')# 
# 2. % operatorprint('Hey %s %s' % (name, name))# 
# 3. formatprint( "My name is {}".format((name)))

# def logging(func): 
#     def log_function_called: 
#         print(f'{func} called.') 
#     func 
#     return log_function_called

# def my_name: 
#     print('chris')
#     def friends_name: 
#         print('naruto')
# my_name
# friends_name#=> chris#=> naruto


# class Car : 
#     def __init__(self, color, speed): 
#         self.color = color 
#         self.speed = speed

# car = Car('red','100mph')
# car.speed#=> '100mph'
# def func: 
#     print('Im a function')
# func
# def add_three(x): 
#     return x + 3
# li = [1,2,3]
# [i for i in map(add_three, li)]#=> [4, 5, 6]

# name = 'chr'
# def add_chars(s): 
#     s += 'is' 
#     print(s)

# add_chars(name) 
# print(name)#=> chris#=> chr

# li = [1,2]
# def add_element(seq): 
#     seq.append(3) 
#     print(seq)
# add_element(li) 
# print(li)#=> [1, 2, 3]#=> [1, 2, 3]


# li3 = [['a'],['b'],['c']]
# li4 = list(li3)
# li3.append([4])
# print(li4)#=> [['a'], ['b'], ['c']]     li3[0][0] = ['X']      print(li4)#=> [[['X']], ['b'], ['c']]




