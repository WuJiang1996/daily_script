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


# a=lambda x,y:x+y
# print(a(3,11))

# a=[1,2,4,2,4,5,7,10,5,5,7,8,9,0,3]
# a.sort()
# # print(a)
# last=a[-1]
# # print(last)
# for i in range(len(a)-2,-1,-1):
#     print(i)
#     if last==a[i]:
#         del a[i]
#     else:
#         last=a[i]
# print(a)



# a = "hello world"
# b = "hello world"
# print(a is b)  # 输出 True
# print(a == b)  # 输出 True 

# print(3 / 2.0)

# a = "你好"
# print(a)

# def func():
#     while True:
#         print("before yield")
#         x = yield
#         print("after yield:",x)

# g = func()
# next(g) # 程序运行到yield并停在该处,等待下一个next
# g.send(1) # 给yield发送值1,这个值被赋值给了x，并且打印出来,然后继续下一次循环停在yield处
# g.send(2) # 给yield发送值2,这个值被赋值给了x，并且打印出来,然后继续下一次循环停在yield处
# next(g) # 没有给x赋值，执行print语句，打印出None,继续循环停在yield处

# print(5 // 2)


# class A:
#     def func(self):
#         print("Hi")
#     def monkey(self):
#         print("Hi, monkey")

# def outer_monkey(a):  # a 这个参数是没有用到的，因为func有一个参数，如果这个函数没有参数的话不能这样直接赋值
#     print("Hi,outer monkey")

# a = A()
# A.func=outer_monkey
# a.func()

# class Circle(object):
#    __pi = 3.14

#    def __init__(self, r):
#        self.r = r

#    def area(self):
#        """
#         圆的面积
#        """
#        return self.r **2* self.__pi

# circle1 = Circle(1)
# print(Circle.__pi)  # 抛出AttributeError异常
# print(circle1.__pi)  # 抛出AttributeError异常



# class Animal(object):  #  python3中所有类都可以继承于object基类
#    def __init__(self, name, age):
#        self.name = name
#        self.age = age

#    def call(self):
#        print(self.name, '会叫')

# ######
# # 现在我们需要定义一个Cat 猫类继承于Animal，猫类比动物类多一个sex属性。 
# ######
# class Cat(Animal):
#    def __init__(self,name,age,sex):
#     #    super(Cat, self).__init__(name,age)  # 不要忘记从Animal类引入属性
#        print(super())
#        super().__init__(name,age)  # 不要忘记从Animal类引入属性
#        self.sex=sex

# if __name__ == '__main__':  # 单模块被引用时下面代码不会受影响，用于调试
#    c = Cat('喵喵', 2, '男')  #  Cat继承了父类Animal的属性
#    c.call()  # 输出 喵喵 会叫 ，Cat继承了父类Animal的方法 



# import time
# import datetime

# def test():
#     start_time = datetime.datetime.now()
#     for i in range(3):
#         time.sleep(1)
#     end_time = datetime.datetime.now()
#     print('执行结束,执行时间为：', end_time - start_time)

# test()
# import logging
# def use_logging(func):
#     def wrapper(*args, **kwargs):
#         logging.warn("%s is running" % func.__name__)
#         return func(*args)
#     return wrapper

# @use_logging
# def foo():
#     print("i am foo")

# foo()


# class Foo(object):
#     def __init__(self, func):
#         self.func = func
#     def __call__ (self):
#         print ('class decorator runing')
#         self.func()
#         print ('class decorator ending')

# @Foo
# def bar( ):
#     print ('bar')
# bar()



# v= dict.fromkeys(['k1','k2'],[])
# print(v)
# v['k1'].append(666)     
# print(v)
# v['k1']=777
# print(v)

# a = range(10)
# print(a[::-3])



# class CalibDataLoader:
#     def __init__(self, batch_size, width, height, calib_count, calib_images_dir):
#         self.index = 0
#         self.batch_size = batch_size
#         self.width = width
#         self.height = height
#         self.calib_count = calib_count
#         self.image_list = glob.glob(os.path.join(calib_images_dir, "*.jpg"))
#         assert (
#             len(self.image_list) > self.batch_size * self.calib_count
#         ), "{} must contains more than {} images for calibration.".format(
#             calib_images_dir, self.batch_size * self.calib_count
#         )
#         self.calibration_data = np.zeros((self.batch_size, 3, height, width), dtype=np.float32)

#     def reset(self):
#         self.index = 0

#     def next_batch(self):
#         if self.index < self.calib_count:
#             for i in range(self.batch_size):
#                 image_path = self.image_list[i + self.index * self.batch_size]
#                 assert os.path.exists(image_path), "image {} not found!".format(image_path)
#                 image = cv2.imread(image_path)
#                 image = Preprocess(image, self.width, self.height)
#                 self.calibration_data[i] = image
#             self.index += 1
#             return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
#         else:
#             return np.array([])

#     def __len__(self):
#         return self.calib_count

# def Preprocess(input_img, width, height):
#     img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (width, height)).astype(np.float32)
#     img = img / 255.0
#     img = np.transpose(img, (2, 0, 1))
#     return img



# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit

# class Calibrator(trt.IInt8EntropyCalibrator2):
#     def __init__(self, data_loader, cache_file=""):
#         trt.IInt8EntropyCalibrator2.__init__(self)
#         self.data_loader = data_loader
#         self.d_input = cuda.mem_alloc(self.data_loader.calibration_data.nbytes)
#         self.cache_file = cache_file
#         data_loader.reset()

#     def get_batch_size(self):
#         return self.data_loader.batch_size

#     def get_batch(self, names):
#         batch = self.data_loader.next_batch()
#         if not batch.size:
#             return None
#         # 把校准数据从CPU搬运到GPU中
#         cuda.memcpy_htod(self.d_input, batch)

#         return [self.d_input]

#     def read_calibration_cache(self):
#         # 如果校准表文件存在则直接从其中读取校准表
#         if os.path.exists(self.cache_file):
#             with open(self.cache_file, "rb") as f:
#                 return f.read()

#     def write_calibration_cache(self, cache):
#         # 如果进行了校准，则把校准表写入文件中以便下次使用
#         with open(self.cache_file, "wb") as f:
#             f.write(cache)
#             f.flush()


# def build_engine():
#     builder = trt.Builder(TRT_LOGGER)
#     network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#     config = builder.create_builder_config()
#     parser = trt.OnnxParser(network, TRT_LOGGER)
#     assert os.path.exists(onnx_file_path), "The onnx file {} is not found".format(onnx_file_path)
#     with open(onnx_file_path, "rb") as model:
#         if not parser.parse(model.read()):
#             print("Failed to parse the ONNX file.")
#             for error in range(parser.num_errors):
#                 print(parser.get_error(error))
#             return None

#     print("Building an engine from file {}, this may take a while...".format(onnx_file_path))

#     # build tensorrt engine
#     config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 * (1 << 30))  
#     if mode == "INT8":
#         config.set_flag(trt.BuilderFlag.INT8)
#         calibrator = Calibrator(data_loader, calibration_table_path)
#         config.int8_calibrator = calibrator
#     else mode == "FP16":
#         config.set_flag(trt.BuilderFlag.FP16)

#     engine = builder.build_engine(network, config)
#     if engine is None:
#         print("Failed to create the engine")
#         return None
#     with open(engine_file_path, "wb") as f:
#         f.write(engine.serialize())

#     return engine


# #量化预处理与训练保持一致，数据对齐
# def preprocess_v1(image_raw):
#     h, w, c = image_raw.shape
#     image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
#     # Calculate widht and height and paddings
#     r_w = width / w
#     r_h = height / h
#     if r_h > r_w:
#         tw = width
#         th = int(r_w * h)
#         tx1 = tx2 = 0
#         ty1 = int((height - th) / 2)
#         ty2 = height - th - ty1
#     else:
#         tw = int(r_h * w)
#         th = height
#         tx1 = int((width - tw) / 2)
#         tx2 = width - tw - tx1
#         ty1 = ty2 = 0
#     # Resize the image with long side while maintaining ratio
#     image = cv2.resize(image, (tw, th))
#     # Pad the short side with (128,128,128)
#     image = cv2.copyMakeBorder(
#         image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
#     )
#     image = image.astype(np.float32)
#     # Normalize to [0,1]
#     image /= 255.0
#     # HWC to CHW format:
#     image = np.transpose(image, [2, 0, 1])
#     # CHW to NCHW format
#     #image = np.expand_dims(image, axis=0)
#     # Convert the image to row-major order, also known as "C order":
#     #image = np.ascontiguousarray(image)
#     return image

# #构建IInt8EntropyCalibrator量化器
# class Calibrator(trt.IInt8EntropyCalibrator):
#     def __init__(self, stream, cache_file=""):
#         trt.IInt8EntropyCalibrator.__init__(self)       
#         self.stream = stream
#         self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
#         self.cache_file = cache_file
#         stream.reset()

#     def get_batch_size(self):
#         return self.stream.batch_size

#     def get_batch(self, names):
#         batch = self.stream.next_batch()
#         if not batch.size:   
#             return None

#         cuda.memcpy_htod(self.d_input, batch)

#         return [int(self.d_input)]

#     def read_calibration_cache(self):
#         # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
#         if os.path.exists(self.cache_file):
#             with open(self.cache_file, "rb") as f:
#                 logger.info("Using calibration cache to save time: {:}".format(self.cache_file))
#                 return f.read()

#     def write_calibration_cache(self, cache):
#         with open(self.cache_file, "wb") as f:
#             logger.info("Caching calibration data for future use: {:}".format(self.cache_file))
#             f.write(cache)

# #加载onnx模型,构建tensorrt engine
# def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="",\
#                fp16_mode=False, int8_mode=False, calibration_stream=None, calibration_table_path="", save_engine=False):
#     """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
#     def build_engine(max_batch_size, save_engine):
#         """Takes an ONNX file and creates a TensorRT engine to run inference with"""
#         with trt.Builder(TRT_LOGGER) as builder, \
#                 builder.create_network(1) as network,\
#                 trt.OnnxParser(network, TRT_LOGGER) as parser:
            
#             # parse onnx model file
#             if not os.path.exists(onnx_file_path):
#                 quit('ONNX file {} not found'.format(onnx_file_path))
#             print('Loading ONNX file from path {}...'.format(onnx_file_path))
#             with open(onnx_file_path, 'rb') as model:
#                 print('Beginning ONNX file parsing')
#                 parser.parse(model.read())
#                 assert network.num_layers > 0, 'Failed to parse ONNX model. \
#                             Please check if the ONNX model is compatible '
#             print('Completed parsing of ONNX file')
#             print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))        
            
#             # build trt engine
#             builder.max_batch_size = max_batch_size
#             builder.max_workspace_size = 1 << 30 # 1GB
#             builder.fp16_mode = fp16_mode
#             if int8_mode:
#                 builder.int8_mode = int8_mode
#                 assert calibration_stream, 'Error: a calibration_stream should be provided for int8 mode'
#                 builder.int8_calibrator  = Calibrator(calibration_stream, calibration_table_path)
#                 print('Int8 mode enabled')
#             engine = builder.build_cuda_engine(network) 
#             if engine is None:
#                 print('Failed to create the engine')
#                 return None   
#             print("Completed creating the engine")
#             if save_engine:
#                 with open(engine_file_path, "wb") as f:
#                     f.write(engine.serialize())
#             return engine
        
#     if os.path.exists(engine_file_path):
#         # If a serialized engine exists, load it instead of building a new one.
#         print("Reading engine from file {}".format(engine_file_path))
#         with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#             return runtime.deserialize_cuda_engine(f.read())
#     else:
#         return build_engine(max_batch_size, save_engine)



# #数据预处理和量化预处理保持一致，故不做展示
# # #对模型的三个输出进行解析，生成返回模型预测的bboxes信息
# void postProcessParall(const int height, const int width, int scale_idx, float postThres, tensor_t * origin_output, vector<int> Strides, vector<Anchor> Anchors, vector<Bbox> *bboxes)
# {
#     Bbox bbox;
#     float cx, cy, w_b, h_b, score;
#     int cid;
#     const float *ptr = (float *)origin_output->pValue;
#     for(unsigned long a=0; a<3; ++a){
#         for(unsigned long h=0; h<height; ++h){
#             for(unsigned long w=0; w<width; ++w){
#                 const float *cls_ptr =  ptr + 5;
#                 cid = argmax(cls_ptr, cls_ptr+NUM_CLASS);
#                 score = sigmoid(ptr[4]) * sigmoid(cls_ptr[cid]);
#                 if(score>=postThres){
#                     cx = (sigmoid(ptr[0]) * 2.f - 0.5f + static_cast<float>(w)) * static_cast<float>(Strides[scale_idx]);
#                     cy = (sigmoid(ptr[1]) * 2.f - 0.5f + static_cast<float>(h)) * static_cast<float>(Strides[scale_idx]);
#                     w_b = powf(sigmoid(ptr[2]) * 2.f, 2) * Anchors[scale_idx * 3 + a].width;
#                     h_b = powf(sigmoid(ptr[3]) * 2.f, 2) * Anchors[scale_idx * 3 + a].height;
#                     bbox.xmin = clip(cx - w_b / 2, 0.F, static_cast<float>(INPUT_W - 1));
#                     bbox.ymin = clip(cy - h_b / 2, 0.f, static_cast<float>(INPUT_H - 1));
#                     bbox.xmax = clip(cx + w_b / 2, 0.f, static_cast<float>(INPUT_W - 1));
#                     bbox.ymax = clip(cy + h_b / 2, 0.f, static_cast<float>(INPUT_H - 1));
#                     bbox.score = score;
#                     bbox.cid = cid;
#                     //std::cout<< "bbox.cid : " << bbox.cid << std::endl;
#                     bboxes->push_back(bbox);
#                 }
#                 ptr += 5 + NUM_CLASS;
#             }
#         }
#     }
# }


# #include <string>
# #include <vector>
# #include "iostream"  
# //#include <fstream>  
# //#include < ctime >
# //#include <direct.h>
# //#include <io.h>

# // ncnn
# #include "ncnn/layer.h"
# #include "ncnn/net.h"
# #include "ncnn/benchmark.h"
# //#include "gpu.h"

# #include "opencv2/core/core.hpp"
# #include "opencv2/highgui/highgui.hpp"
# #include <opencv2/imgproc.hpp>
# #include "opencv2/opencv.hpp"  

# using namespace std;
# using namespace cv;

# static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
# static ncnn::PoolAllocator g_workspace_pool_allocator;

# static ncnn::Net yolov5;

# class YoloV5Focus : public ncnn::Layer
# {
# public:
#  YoloV5Focus()
#  {
#   one_blob_only = true;
#  }

#  virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
#  {
#   int w = bottom_blob.w;
#   int h = bottom_blob.h;
#   int channels = bottom_blob.c;

#   int outw = w / 2;
#   int outh = h / 2;
#   int outc = channels * 4;

#   top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
#   if (top_blob.empty())
#    return -100;

# #pragma omp parallel for num_threads(opt.num_threads)
#   for (int p = 0; p < outc; p++)
#   {
#    const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
#    float* outptr = top_blob.channel(p);

#    for (int i = 0; i < outh; i++)
#    {
#     for (int j = 0; j < outw; j++)
#     {
#      *outptr = *ptr;

#      outptr += 1;
#      ptr += 2;
#     }

#     ptr += w;
#    }
#   }

#   return 0;
#  }
# };
# DEFINE_LAYER_CREATOR(YoloV5Focus)

# struct Object
# {
#  float x;
#  float y;
#  float w;
#  float h;
#  int label;
#  float prob;
# };

# static inline float intersection_area(const Object& a, const Object& b)
# {
#  if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y)
#  {
#   // no intersection
#   return 0.f;
#  }

#  float inter_width = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
#  float inter_height = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);

#  return inter_width * inter_height;
# }

# static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
# {
#  int i = left;
#  int j = right;
#  float p = faceobjects[(left + right) / 2].prob;

#  while (i <= j)
#  {
#   while (faceobjects[i].prob > p)
#    i++;

#   while (faceobjects[j].prob < p)
#    j--;

#   if (i <= j)
#   {
#    // swap
#    std::swap(faceobjects[i], faceobjects[j]);

#    i++;
#    j--;
#   }
#  }

# #pragma omp parallel sections
#  {
# #pragma omp section
#   {
#    if (left < j) qsort_descent_inplace(faceobjects, left, j);
#   }
# #pragma omp section
#   {
#    if (i < right) qsort_descent_inplace(faceobjects, i, right);
#   }
#  }
# }


# static void qsort_descent_inplace(std::vector<Object>& faceobjects)
# {
#  if (faceobjects.empty())
#   return;

#  qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
# }

# static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
# {
#  picked.clear();

#  const int n = faceobjects.size();

#  std::vector<float> areas(n);
#  for (int i = 0; i < n; i++)
#  {
#   areas[i] = faceobjects[i].w * faceobjects[i].h;
#  }
#  for (int i = 0; i < n; i++)
#  {
#   const Object& a = faceobjects[i];

#   int keep = 1;
#   for (int j = 0; j < (int)picked.size(); j++)
#   {
#    const Object& b = faceobjects[picked[j]];

#    // intersection over union
#    float inter_area = intersection_area(a, b);
#    float union_area = areas[i] + areas[picked[j]] - inter_area;
#    // float IoU = inter_area / union_area
#    if (inter_area / union_area > nms_threshold)
#     keep = 0;
#   }
#   if (keep)
#    picked.push_back(i);
#  }
# }

# static inline float sigmoid(float x)
# {
#  return static_cast<float>(1.f / (1.f + exp(-x)));
# }

# static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
# {
#  const int num_grid = feat_blob.h;

#  int num_grid_x;
#  int num_grid_y;
#  if (in_pad.w > in_pad.h)
#  {
#   num_grid_x = in_pad.w / stride;
#   num_grid_y = num_grid / num_grid_x;
#  }
#  else
#  {
#   num_grid_y = in_pad.h / stride;
#   num_grid_x = num_grid / num_grid_y;
#  }

#  const int num_class = feat_blob.w - 5;

#  const int num_anchors = anchors.w / 2;

#  for (int q = 0; q < num_anchors; q++)
#  {
#   const float anchor_w = anchors[q * 2];
#   const float anchor_h = anchors[q * 2 + 1];

#   const ncnn::Mat feat = feat_blob.channel(q);

#   for (int i = 0; i < num_grid_y; i++)

#   {
#    for (int j = 0; j < num_grid_x; j++)
#    {
#     const float* featptr = feat.row(i * num_grid_x + j);

#     // find class index with max class score
#     int class_index = 0;
#     float class_score = -FLT_MAX;
#     for (int k = 0; k < num_class; k++)
#     {
#      float score = featptr[5 + k];
#      if (score > class_score)
#      {
#       class_index = k;
#       class_score = score;
#      }
#     }

#     float box_score = featptr[4];

#     float confidence = sigmoid(box_score) * sigmoid(class_score);

#     if (confidence >= prob_threshold)
#     {
#      float dx = sigmoid(featptr[0]);
#      float dy = sigmoid(featptr[1]);
#      float dw = sigmoid(featptr[2]);
#      float dh = sigmoid(featptr[3]);

#      float pb_cx = (dx * 2.f - 0.5f + j) * stride;
#      float pb_cy = (dy * 2.f - 0.5f + i) * stride;

#      float pb_w = pow(dw * 2.f, 2) * anchor_w;
#      float pb_h = pow(dh * 2.f, 2) * anchor_h;

#      float x0 = pb_cx - pb_w * 0.5f;
#      float y0 = pb_cy - pb_h * 0.5f;
#      float x1 = pb_cx + pb_w * 0.5f;
#      float y1 = pb_cy + pb_h * 0.5f;

#      Object obj;
#      obj.x = x0;
#      obj.y = y0;
#      obj.w = x1 - x0;
#      obj.h = y1 - y0;
#      obj.label = class_index;
#      obj.prob = confidence;
#      objects.push_back(obj);
#     }
#    }
#   }
#  }
# }

# extern "C" {

#  void release()
#  {
#   fprintf(stderr, "YoloV5Ncnn finished!");

#   //ncnn::destroy_gpu_instance();
#  }

#  int init()
#  {
#   fprintf(stderr, "YoloV5Ncnn init!\n");
#   ncnn::Option opt;
#   opt.lightmode = true;
#   opt.num_threads = 4;
#   opt.blob_allocator = &g_blob_pool_allocator;
#   opt.workspace_allocator = &g_workspace_pool_allocator;
#   opt.use_packing_layout = true;

#   yolov5.opt = opt;

#   yolov5.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
#   // init param
#   {
#    int ret = yolov5.load_param("yolov5s.param");  
#    if (ret != 0)
#    {
#     std::cout << "ret= " << ret << std::endl;
#     fprintf(stderr, "YoloV5Ncnn, load_param failed");
#     return -301;
#    }
#   }

#   // init bin
#   {
#    int ret = yolov5.load_model("yolov5s.bin");  
#    if (ret != 0)
#    {
#     fprintf(stderr, "YoloV5Ncnn, load_model failed");
#     return -301;
#    }
#   }
#   return 0;
#  }

#  int detect(cv::Mat img, std::vector<Object> &objects)
#  {

#   double start_time = ncnn::get_current_time();
#   const int target_size = 320;

#   // letterbox pad to multiple of 32
#   const int width = img.cols;//1280
#   const int height = img.rows;//720
#   int w = img.cols;//1280
#   int h = img.rows;//720
#   float scale = 1.f;
#   if (w > h)
#   {
#    scale = (float)target_size / w;//640/1280
#    w = target_size;//640
#    h = h * scale;//360
#   }
#   else
#   {
#    scale = (float)target_size / h;
#    h = target_size;
#    w = w * scale;
#   }
#   cv::resize(img, img, cv::Size(w, h));
#   ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, w, h);

#   // pad to target_size rectangle
#   // yolov5/utils/datasets.py letterbox
#   int wpad = (w + 31) / 32 * 32 - w;
#   int hpad = (h + 31) / 32 * 32 - h;
#   ncnn::Mat in_pad;
#   ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
#   // yolov5
#   //std::vector<Object> objects;
#   {
#    const float prob_threshold = 0.4f;
#    const float nms_threshold = 0.51f;

#    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
#    in_pad.substract_mean_normalize(0, norm_vals);

#    ncnn::Extractor ex = yolov5.create_extractor();
#    //ex.set_vulkan_compute(use_gpu);

#    ex.input("images", in_pad);
#    std::vector<Object> proposals;

#    // anchor setting from yolov5/models/yolov5s.yaml

#    // stride 8
#    {
#     ncnn::Mat out;
#     ex.extract("output", out);
#     ncnn::Mat anchors(6);
#     anchors[0] = 10.f;
#     anchors[1] = 13.f;
#     anchors[2] = 16.f;
#     anchors[3] = 30.f;
#     anchors[4] = 33.f;
#     anchors[5] = 23.f;

#     std::vector<Object> objects8;
#     generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);
#     proposals.insert(proposals.end(), objects8.begin(), objects8.end());
#    }

#    // stride 16
#    {
#     ncnn::Mat out;
#     ex.extract("771", out);

#     ncnn::Mat anchors(6);
#     anchors[0] = 30.f;
#     anchors[1] = 61.f;
#     anchors[2] = 62.f;
#     anchors[3] = 45.f;
#     anchors[4] = 59.f;
#     anchors[5] = 119.f;

#     std::vector<Object> objects16;
#     generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

#     proposals.insert(proposals.end(), objects16.begin(), objects16.end());
#    }
#    // stride 32
#    {
#     ncnn::Mat out;
#     ex.extract("791", out);
#     ncnn::Mat anchors(6);
#     anchors[0] = 116.f;
#     anchors[1] = 90.f;
#     anchors[2] = 156.f;
#     anchors[3] = 198.f;
#     anchors[4] = 373.f;

#     anchors[5] = 326.f;

#     std::vector<Object> objects32;
#     generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

#     proposals.insert(proposals.end(), objects32.begin(), objects32.end());
#    }

#    // sort all proposals by score from highest to lowest
#    qsort_descent_inplace(proposals);
#    // apply nms with nms_threshold
#    std::vector<int> picked;
#    nms_sorted_bboxes(proposals, picked, nms_threshold);

#    int count = picked.size();
#    objects.resize(count);
#    for (int i = 0; i < count; i++)
#    {
#     objects[i] = proposals[picked[i]];

#     // adjust offset to original unpadded
#     float x0 = (objects[i].x - (wpad / 2)) / scale;
#     float y0 = (objects[i].y - (hpad / 2)) / scale;
#     float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
#     float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

#     // clip
#     x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
#     y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
#     x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
#     y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);
#     objects[i].x = x0;
#     objects[i].y = y0;
#     objects[i].w = x1;
#     objects[i].h = y1;
#    }
#   }

#   return 0;
#  }
# }


# static const char* class_names[] = {
#  "four_fingers","hand_with_fingers_splayed","index_pointing_up","little_finger",
#  "ok_hand","raised_fist","raised_hand","sign_of_the_horns","three","thumbup","victory_hand"
# };


# void draw_face_box(cv::Mat& bgr, std::vector<Object> object) //主要的emoji显示函数
# {
#  for (int i = 0; i < object.size(); i++)
#  {
#   const auto obj = object[i];
#   cv::rectangle(bgr, cv::Point(obj.x, obj.y), cv::Point(obj.w, obj.h), cv::Scalar(0, 255, 0), 3, 8, 0);
#   std::cout << "label:" << class_names[obj.label] << std::endl;
#   string emoji_path = "emoji\\" + string(class_names[obj.label]) + ".png"; //这个是emoji图片的路径
#   cv::Mat logo = cv::imread(emoji_path);
#   if (logo.empty()) {
#    std::cout << "imread logo failed!!!" << std::endl;
#    return;
#   }
#   resize(logo, logo, cv::Size(80, 80));
#   cv::Mat imageROI = bgr(cv::Range(obj.x, obj.x + logo.rows), cv::Range(obj.y, obj.y + logo.cols));  //emoji的图片放在图中的位置，也就是手势框的旁边
#   logo.copyTo(imageROI); //把emoji放在原图中
#  }

# }

# int main()
# {
#  Mat frame;

#  VideoCapture capture(0);
#  init();
#  while (true)
#  {
#   capture >> frame;            
#   if (!frame.empty()) {          
#    std::vector<Object> objects;
#    detect(frame, objects);
#    draw_face_box(frame, objects);
#    imshow("window", frame);  
#   }
#   if (waitKey(20) == 'q')    
#    break;
#  }

#  capture.release();     

#  return 0;
# }


# import numpy as np 

# a = np.array([0.0, 10.0, 20.0, 30.0])
# b = np.array([1.0, 2.0, 3.0])
# # print(a[:, np.newaxis] + b)
# print(a[:, np.newaxis])



# import torch
# a = torch.randn(4)
# print("a:",a)
# min = torch.linspace(-1, 1, steps=4)
# print("min:",min)
# print(torch.clamp(a, min=min))


# a=torch.randint(low=0,high=10,size=(10,1))
# print(a)
# a=torch.clamp(a,3,9)
# print(a)


import torch

x = torch.randn(128, 20)  # 输入的维度是（128，20）
m = torch.nn.Linear(20, 30)  # 20,30是指维度
output = m(x)
print('m.weight.shape:\n ', m.weight.shape)
print('m.bias.shape:\n', m.bias.shape)
print('output.shape:\n', output.shape)

# ans = torch.mm(input,torch.t(m.weight))+m.bias 等价于下面的
ans = torch.mm(x, m.weight.t()) + m.bias   
print('ans.shape:\n', ans.shape)

print(torch.equal(ans, output))