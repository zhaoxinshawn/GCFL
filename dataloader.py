import copy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform1=None,transform2 = None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        # RGB format
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform1 = transform1
        self.transform2 = transform2
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2  = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform1(img1)
        img2 = self.transform2(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class SYSUTriData(data.Dataset):
    def __init__(self, data_dir, transform_color=None,transform_gray = None, transform_thermal = None,
                 colorIndex=None, grayIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_gray_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_gray_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        # RGB format
        self.train_color_image = train_color_image
        self.train_gray_image = train_gray_image
        self.train_thermal_image = train_thermal_image

        self.transform_color = transform_color
        self.transform_gray=transform_gray
        self.transform_thermal = transform_thermal

        self.cIndex = colorIndex
        self.gIndex = grayIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img_color, target_color = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img_gray, target_gray = self.train_gray_image[self.gIndex[index]], self.train_gray_label[self.gIndex[index]]
        img_thermal, target_thermal  = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img_color = self.transform_color(img_color)
        img_gray = self.transform_gray(img_gray)
        img_thermal = self.transform_thermal(img_thermal)

        return img_color, img_gray, img_thermal, target_color, target_gray, target_thermal

    def __len__(self):
        return len(self.train_color_label)
class SYSUTri2Data(data.Dataset):
    def __init__(self, data_dir,
                 transform_color=None, transform_thermal=None,
                 transform_gcolor=None, transform_gthermal=None,
                 colorIndex=None, thermalIndex=None,
                 gcolorIndex=None, gthermalIndex=None):
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')
        self.train_gcolor_label=copy.deepcopy(self.train_color_label)
        # train_gray_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        # self.train_gray_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        self.train_gthermal_label=copy.deepcopy(self.train_thermal_label)
        # RGB format
        self.train_color_image = train_color_image
        self.train_gcolor_image = copy.deepcopy(train_color_image)
        self.train_gthermal_image = copy.deepcopy(train_thermal_image)
        self.train_thermal_image = train_thermal_image

        self.transform_color = transform_color
        self.transform_gcolor = transform_gcolor
        self.transform_gthermal = transform_gthermal
        self.transform_thermal = transform_thermal

        self.cIndex = colorIndex
        self.gcIndex = gcolorIndex
        self.gtIndex = gthermalIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img_color, target_color = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img_thermal, target_thermal = self.train_thermal_image[self.tIndex[index]], \
                                      self.train_thermal_label[self.tIndex[index]]
        img_gcolor, target_gcolor = self.train_gcolor_image[self.gcIndex[index]], \
                                    self.train_gcolor_label[self.gcIndex[index]]
        img_gthermal, target_gthermal = self.train_gthermal_image[self.gtIndex[index]], \
                                        self.train_gthermal_label[self.gtIndex[index]]
        img_color = self.transform_color(img_color)
        img_thermal = self.transform_thermal(img_thermal)
        img_gcolor = self.transform_gcolor(img_gcolor)
        img_gthermal = self.transform_gthermal(img_gthermal)

        return img_color, img_thermal, img_gcolor, img_gthermal, \
               target_color, target_thermal, target_gcolor, target_gthermal

    def __len__(self):
        return len(self.train_color_label)
class RegDBTriData(data.Dataset):
    def __init__(self, data_dir, trial, transform_color=None,transform_gray = None, transform_thermal = None,
                 colorIndex=None, grayIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'

        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, self.train_color_label = load_data(train_color_list)
        gray_img_file, self.train_gray_label = load_data(train_color_list)
        thermal_img_file, self.train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_gray_image=copy.deepcopy(train_color_image)
        # train_gray_image = []
        # for i in range(len(gray_img_file)):
        #     img = Image.open(data_dir + gray_img_file[i])
        #     img = img.resize((144, 288), Image.ANTIALIAS)
        #     pix_array = np.array(img)
        #     train_gray_image.append(pix_array)
        # train_gray_image = np.array(train_gray_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            # img.show()
            pix_array = np.array(img)
            # plt.imshow(pix_array)
            # plt.show()
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        # train_gray_image = np.vstack([copy.deepcopy(train_color_image),copy.deepcopy(train_thermal_image)])

        # RGB format
        self.train_color_image = train_color_image
        self.train_gray_image = train_gray_image
        self.train_thermal_image = train_thermal_image

        self.transform_color = transform_color
        self.transform_gray = transform_gray
        self.transform_thermal = transform_thermal

        self.cIndex = colorIndex
        self.gIndex = grayIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img_color, target_color = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img_gray, target_gray = self.train_gray_image[self.gIndex[index]], self.train_gray_label[self.gIndex[index]]
        img_thermal, target_thermal = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[
            self.tIndex[index]]

        img_color = self.transform_color(img_color)
        img_gray = self.transform_gray(img_gray)

        # import matplotlib.pyplot as plt
        # pic = np.transpose(img_thermal)  # ,axes=[2,1,0])
        # plt.imshow(img_thermal)
        # plt.show()
        img_thermal = self.transform_thermal(img_thermal)

        return img_color, img_gray, img_thermal, target_color, target_gray, target_thermal

        # img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        # img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        #
        # img1 = self.transform1(img1)
        # img2 = self.transform2(img2)
        #
        # return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
class RegDBTri2Data(data.Dataset):
    def __init__(self, data_dir, trial,
                 transform_color=None, transform_thermal = None,
                 transform_gcolor = None, transform_gthermal = None,
                 colorIndex=None, thermalIndex=None,
                 gcolorIndex=None, gthermalIndex=None ):
        # Load training images (path) and labels
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, self.train_color_label = load_data(train_color_list)
        thermal_img_file, self.train_thermal_label = load_data(train_thermal_list)
        gcolor_img_file, self.train_gcolor_label = load_data(train_color_list)
        gthermal_img_file, self.train_gthermal_label = load_data(train_thermal_list)

        # self.train_gray_label=self.train_color_label+self.train_thermal_label

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            # img.show()
            pix_array = np.array(img)
            # plt.imshow(pix_array)
            # plt.show()
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        train_gcolor_image = copy.deepcopy(train_color_image)
        train_gthermal_image = copy.deepcopy(train_thermal_image)

        # RGB format
        self.train_color_image = train_color_image
        self.train_gcolor_image = train_gcolor_image
        self.train_gthermal_image = train_gthermal_image
        self.train_thermal_image = train_thermal_image

        self.transform_color = transform_color
        self.transform_gcolor = transform_gcolor
        self.transform_gthermal = transform_gthermal
        self.transform_thermal = transform_thermal

        # self.transform_color1=transform_regdb1
        # self.transform_color2 = transform_regdb2

        self.cIndex = colorIndex
        self.gcIndex = gcolorIndex
        self.gtIndex = gthermalIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img_color, target_color = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img_thermal, target_thermal = self.train_thermal_image[self.tIndex[index]],\
                                      self.train_thermal_label[self.tIndex[index]]
        img_gcolor, target_gcolor = self.train_gcolor_image[self.gcIndex[index]],\
                                    self.train_gcolor_label[self.gcIndex[index]]
        img_gthermal, target_gthermal = self.train_gthermal_image[self.gtIndex[index]],\
                                        self.train_gthermal_label[self.gtIndex[index]]
        # if random.uniform(0, 1) > 0.5:
        #     img_color = self.transform_color1(img_color)
        #     img_gcolor = self.transform_color1(img_gcolor)
        # else:
        #     img_color = self.transform_color2(img_color)
        #     img_gcolor = self.transform_color2(img_gcolor)
        img_color = self.transform_color(img_color)
        img_thermal = self.transform_thermal(img_thermal)
        img_gcolor = self.transform_gcolor(img_gcolor)
        img_gthermal = self.transform_gthermal(img_gthermal)

        return img_color,  img_thermal,img_gcolor,img_gthermal,\
               target_color, target_thermal,target_gcolor, target_gthermal

        # img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        # img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        #
        # img1 = self.transform1(img1)
        # img2 = self.transform2(img2)
        #
        # return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform_color=None,transform_thermal = None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        # RGB format
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # RGB format
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform_color = transform_color
        self.transform_thermal = transform_thermal
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform_color(img1)
        img2 = self.transform_thermal(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
class RegDB_Rgary_Data(data.Dataset):
    def __init__(self, data_dir, trial, transform_color=None,transform_gray = None, transform_thermal = None,
                 colorIndex=None, grayIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, self.train_color_label = load_data(train_color_list)
        gray_img_file, self.train_gray_label = load_data(train_color_list)
        thermal_img_file, self.train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_gray_image = copy.deepcopy(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            # img.show()
            pix_array = np.array(img)
            # plt.imshow(pix_array)
            # plt.show()
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        # RGB format
        self.train_color_image = train_color_image
        self.train_gray_image = train_gray_image
        self.train_thermal_image = train_thermal_image

        self.transform_color = transform_color
        self.transform_gray = transform_gray
        self.transform_thermal = transform_thermal

        self.cIndex = colorIndex
        self.gIndex = grayIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img_color, target_color = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img_gray, target_gray = self.train_gray_image[self.gIndex[index]], self.train_gray_label[self.gIndex[index]]
        img_thermal, target_thermal = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[
            self.tIndex[index]]

        img_color = self.transform_color(img_color)
        img_gray = self.transform_gray(img_gray)
        img_thermal = self.transform_thermal(img_thermal)

        return img_color, img_gray, img_thermal, target_color, target_gray, target_thermal

    def __len__(self):
        return len(self.train_color_label)
class RegDB_Igary_Data(data.Dataset):
    def __init__(self, data_dir, trial, transform_color=None, transform_gray=None, transform_thermal=None,
                 colorIndex=None, grayIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, self.train_color_label = load_data(train_color_list)
        gray_img_file, self.train_gray_label = load_data(train_thermal_list)
        thermal_img_file, self.train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            # img.show()
            pix_array = np.array(img)
            # plt.imshow(pix_array)
            # plt.show()
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        train_gray_image = copy.deepcopy(train_thermal_image)

        # RGB format
        self.train_color_image = train_color_image
        self.train_gray_image = train_gray_image
        self.train_thermal_image = train_thermal_image

        self.transform_color = transform_color
        self.transform_gray = transform_gray
        self.transform_thermal = transform_thermal

        self.cIndex = colorIndex
        self.gIndex = grayIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img_color, target_color = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img_gray, target_gray = self.train_gray_image[self.gIndex[index]], self.train_gray_label[self.gIndex[index]]
        img_thermal, target_thermal = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[
            self.tIndex[index]]

        img_color = self.transform_color(img_color)
        img_gray = self.transform_gray(img_gray)
        img_thermal = self.transform_thermal(img_thermal)

        return img_color, img_gray, img_thermal, target_color, target_gray, target_thermal

    def __len__(self):
        return len(self.train_color_label)
class RegDB_2gary_Data(data.Dataset):
    def __init__(self, data_dir, trial,
                 transform_color=None, transform_thermal = None,
                 transform_gcolor = None, transform_gthermal = None,
                 colorIndex=None, thermalIndex=None,
                 gcolorIndex=None, gthermalIndex=None ):
        # Load training images (path) and labels
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, self.train_color_label = load_data(train_color_list)
        thermal_img_file, self.train_thermal_label = load_data(train_thermal_list)
        gcolor_img_file, self.train_gcolor_label = load_data(train_color_list)
        gthermal_img_file, self.train_gthermal_label = load_data(train_thermal_list)

        # self.train_gray_label=self.train_color_label+self.train_thermal_label

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            # img.show()
            pix_array = np.array(img)
            # plt.imshow(pix_array)
            # plt.show()
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        train_gcolor_image = copy.deepcopy(train_color_image)
        train_gthermal_image = copy.deepcopy(train_thermal_image)

        # RGB format
        self.train_color_image = train_color_image
        self.train_gcolor_image = train_gcolor_image
        self.train_gthermal_image = train_gthermal_image
        self.train_thermal_image = train_thermal_image

        self.transform_color = transform_color
        self.transform_gcolor = transform_gcolor
        self.transform_gthermal = transform_gthermal
        self.transform_thermal = transform_thermal

        # self.transform_color1=transform_regdb1
        # self.transform_color2 = transform_regdb2

        self.cIndex = colorIndex
        self.gcIndex = gcolorIndex
        self.gtIndex = gthermalIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img_color, target_color = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img_thermal, target_thermal = self.train_thermal_image[self.tIndex[index]],\
                                      self.train_thermal_label[self.tIndex[index]]
        img_gcolor, target_gcolor = self.train_gcolor_image[self.gcIndex[index]],\
                                    self.train_gcolor_label[self.gcIndex[index]]
        img_gthermal, target_gthermal = self.train_gthermal_image[self.gtIndex[index]],\
                                        self.train_gthermal_label[self.gtIndex[index]]
        # if random.uniform(0, 1) > 0.5:
        #     img_color = self.transform_color1(img_color)
        #     img_gcolor = self.transform_color1(img_gcolor)
        # else:
        #     img_color = self.transform_color2(img_color)
        #     img_gcolor = self.transform_color2(img_gcolor)
        img_color = self.transform_color(img_color)
        img_thermal = self.transform_thermal(img_thermal)
        img_gcolor = self.transform_gcolor(img_gcolor)
        img_gthermal = self.transform_gthermal(img_gthermal)

        return img_color,  img_thermal,img_gcolor,img_gthermal,\
               target_color, target_thermal,target_gcolor, target_gthermal

        # img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        # img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        #
        # img1 = self.transform1(img1)
        # img2 = self.transform2(img2)
        #
        # return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class TestData_RegDB(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(224, 224)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)

        return img1, target1

    def __len__(self):
        return len(self.test_image)

class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(224, 224)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label


def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)

    return color_pos, thermal_pos
# 存放同id 的位置


class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def  __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, batchSize, per_img):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        sample_color = np.arange(batchSize)
        sample_thermal = np.arange(batchSize)
        N = np.maximum(len(train_color_label), len(train_thermal_label))

        # per_img = 4
        per_id = batchSize / per_img
        for j in range(N // batchSize + 1): #batchSize=32
            batch_idx = np.random.choice(uni_label, int(per_id), replace=False)# 随机选8人

            for s, i in enumerate(range(0, batchSize, per_img)): # 0~32 per 4
                sample_color[i:i + per_img] = np.random.choice(color_pos[batch_idx[s]], per_img, replace=False)#每人选4张图片
                sample_thermal[i:i + per_img] = np.random.choice(thermal_pos[batch_idx[s]], per_img, replace=False)

            if j == 0:
                index1 = sample_color
                index2 = sample_thermal
            else:
                index1 = np.hstack((index1, sample_color))
                index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1 #32->64->96...
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N
class TwoMSampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def  __init__(self, train_color_label,  train_thermal_label, color_pos,thermal_pos, batchSize, per_img):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label) #类别

        sample_color = np.arange(batchSize)
        sample_thermal = np.arange(batchSize)
        N = np.maximum(len(train_color_label), len(train_thermal_label))

        # per_img = 4
        per_id = batchSize / per_img
        for j in range(N // batchSize + 1):
            batch_idx = np.random.choice(uni_label, int(per_id), replace=False)

            for s, i in enumerate(range(0, batchSize, per_img)):
                sample_color[i:i + per_img] = np.random.choice(color_pos[batch_idx[s]], per_img, replace=False)
                sample_thermal[i:i + per_img] = np.random.choice(thermal_pos[batch_idx[s]], per_img, replace=False)

            if j == 0:
                index_color = sample_color
                index_thermal = sample_thermal
            else:
                index_color = np.hstack((index_color, sample_color))
                index_thermal = np.hstack((index_thermal, sample_thermal))



        self.index_color = index_color
        self.index_thermal = index_thermal

        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index_color)))

    def __len__(self):
        return self.N
class Tri1Sampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def  __init__(self, train_color_label,  train_thermal_label, color_pos,thermal_pos, batchSize, per_img):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label) #类别

        sample_color = np.arange(batchSize)
        sample_gray = np.arange(batchSize)
        sample_thermal = np.arange(batchSize)
        N = np.maximum(len(train_color_label), len(train_thermal_label))

        # per_img = 4
        per_id = batchSize / per_img
        for j in range(N // batchSize + 1):
            batch_idx = np.random.choice(uni_label, int(per_id), replace=False)

            for s, i in enumerate(range(0, batchSize, per_img)):
                sample_color[i:i + per_img] = np.random.choice(color_pos[batch_idx[s]], per_img, replace=False)
                sample_gray[i:i + per_img] = sample_color[i:i+per_img]
                sample_thermal[i:i + per_img] = np.random.choice(thermal_pos[batch_idx[s]], per_img, replace=False)

            if j == 0:
                index_color = sample_color
                index_thermal = sample_thermal
                index_gray = sample_gray
            else:
                index_color = np.hstack((index_color, sample_color))
                index_thermal = np.hstack((index_thermal, sample_thermal))
                index_gray = np.hstack((index_gray, sample_gray))


        self.index_color = index_color
        self.index_thermal = index_thermal
        self.index_gray = index_gray

        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index_color)))

    def __len__(self):
        return self.N
class Tri2Sampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def  __init__(self, train_color_label,  train_thermal_label, color_pos,thermal_pos, batchSize, per_img):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label) #类别

        sample_color = np.arange(batchSize)
        sample_gcolor = np.arange(batchSize)
        sample_gthermal = np.arange(batchSize)
        sample_thermal = np.arange(batchSize)
        N = np.maximum(len(train_color_label), len(train_thermal_label))

        # per_img = 4
        per_id = batchSize / per_img
        for j in range(N // batchSize + 1):
            batch_idx = np.random.choice(uni_label, int(per_id), replace=False)

            for s, i in enumerate(range(0, batchSize, per_img)):
                sample_color[i:i + per_img] = np.random.choice(color_pos[batch_idx[s]], per_img, replace=False)
                sample_thermal[i:i + per_img] = np.random.choice(thermal_pos[batch_idx[s]], per_img, replace=False)
                sample_gcolor[i:i + per_img] = sample_color[i:i + per_img]
                sample_gthermal[i:i + per_img] = sample_thermal[i:i + per_img]
            if j == 0:
                index_color = sample_color
                index_thermal = sample_thermal
                index_gcolor = sample_gcolor
                index_gthermal = sample_gthermal

            else:
                index_color = np.hstack((index_color, sample_color))
                index_thermal = np.hstack((index_thermal, sample_thermal))
                index_gcolor = np.hstack((index_gcolor, sample_gcolor))
                index_gthermal = np.hstack((index_gthermal, sample_gthermal))

        self.index_color = index_color
        self.index_thermal = index_thermal
        self.index_gcolor = index_gcolor
        self.index_gthermal = index_gthermal

        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index_color)))

    def __len__(self):
        return self.N