# import cv2  #调用opencv模块
 
# imgpath = "./1.jpg" #写入路径
# img = cv2.imread(imgpath)  #读取图像
# print(img.shape) #查看图片尺寸
 
# """
# 在图片的指定位置加边框，左上角的像素坐标是（100，100）
# 右下角的像素坐标是（500，500）
# 且一定注意，像素坐标值都是正整数
# (0,255,0)表示边框颜色是绿色
# 3表示边框的线宽度为3
# """
 
# cv2.rectangle(img,(440,387),(525,437),(0,0,255),3)
# # cv2.imshow("Image", img)   #显示图片
# cv2.waitKey (0)  #边框等待时长
# cv2.destroyAllWindows() #关闭所有边框
# cv2.imwrite("./2.jpg", img) #导出图片，注意新的文件不要与原图重名，否则会覆盖原图




from moviepy.editor import VideoFileClip, ColorClip, CompositeVideoClip
 
# my_clip = VideoFileClip("video1.mp4").subclip(0.0, 20)
my_clip = VideoFileClip("video.mp4").subclip(0.0,26)
 
mask1 = ColorClip((137, 2), (255, 0, 0)).set_position((706, 612)).set_start(4).set_end(18)
mask2 = ColorClip((137, 2), (255, 0, 0)).set_position((706, 692)).set_start(4).set_end(18)
mask3 = ColorClip((2, 80), (255, 0, 0)).set_position((706, 612)).set_start(4).set_end(18)
mask4 = ColorClip((2, 80), (255, 0, 0)).set_position((843, 612)).set_start(4).set_end(18)
result = CompositeVideoClip([my_clip, mask1, mask2, mask3, mask4])


# mask1 = ColorClip((500, 2), (255, 0, 0)).set_position((20, 620)).set_start(1).set_end(5)
# mask2 = ColorClip((500, 2), (255, 0, 0)).set_position((20, 800)).set_start(6).set_end(10)
# mask3 = ColorClip((2, 183), (255, 0, 0)).set_position((20, 620)).set_start(11).set_end(15)
# mask4 = ColorClip((2, 183), (255, 0, 0)).set_position((520, 620)).set_start(16).set_end(20)
# result = CompositeVideoClip([my_clip, mask1, mask2, mask3, mask4])
 
result.write_videofile("test_2.mp4", fps=25)






# 去掉因为3x3卷积的padding多出来的行或者列
class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]

# 论文提出的3x3+1x3+3x1
class ACBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)


    # forward函数
    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            # print(square_outputs.size())
            # return square_outputs
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            # print(vertical_outputs.size())
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            # print(horizontal_outputs.size())
            return square_outputs + vertical_outputs + horizontal_outputs