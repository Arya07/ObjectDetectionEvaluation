import yarp
import sys
import numpy as np
from PIL import Image
import os
import xml.etree.ElementTree as ET
import numpy as np
import time
import random

# Initialise YARP
yarp.Network.init()

class iCWT_player(yarp.RFModule):
    def configure(self, rf):
        self.dataset_folder = '/home/elisa/Data/Ho3D'
        self.images_folder = self.dataset_folder + '/train'
        self.imageset = self.dataset_folder + '/ImageSets/imageset_test.txt'

        self.image_w = 640
        self.image_h = 480


        self.output_image_port = yarp.Port()
        self.output_image_port.open('/depthCamera/rgbImage:o')
        print('{:s} opened'.format('/depthCamera/rgbImage:o'))

        self.cmd_port = yarp.Port()
        self.cmd_port.open('/Ho3DPlayer/cmd:i')
        print('{:s} opened'.format('/Ho3DPlayer/cmd:i'))
        self.attach(self.cmd_port)

        print('Preparing output image...')
        self.out_buf_image = yarp.ImageRgb()
        self.out_buf_image.resize(self.image_w, self.image_h)
        self.out_buf_array = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
        self.out_buf_image.setExternal(self.out_buf_array.data, self.out_buf_array.shape[1], self.out_buf_array.shape[0])

        with open(self.imageset, 'r') as f:
            print('Imageset opened')
            self.lines = f.readlines()
            #self.lines = sorted(self.lines)

        self.counter = 0
        self.state = 'do_nothing'
        return True

    def cleanup(self):
        print('Cleanup function')
        self.output_image_port.close()
        self.cmd_port.close()

    def interruptModule(self):
        print('Interrupt function')
        self.output_image_port.interrupt()
        self.cmd_port.interrupt()
        return True

    def getPeriod(self):
        return 0.005

    def respond(self, cmd, reply):
        if cmd.get(0).asString() == 'imageset':
            self.imageset = self.dataset_folder + '/ImageSets/' + cmd.get(1).asString() + '.txt'
            if os.path.exists(self.imageset):
                with open(self.imageset, 'r') as f:
                    self.lines = f.readlines()
                    #self.lines = sorted(self.lines)
                self.counter = 0
                reply.addString('Imageset loaded')
            else:
                reply.addString('Imageset does not exist')
        elif cmd.get(0).asString() == 'start':
            self.counter = 0
            self.state = 'stream'
            reply.addString('Stream started from first image')
        elif cmd.get(0).asString() == 'pause':
            self.state = 'do_nothing'
            reply.addString('Stream paused')
        elif cmd.get(0).asString() == 'resume':
            self.state = 'stream'
            reply.addString('Stream resumed')
        elif cmd.get(0).asString() == 'quit':
            reply.addString('Quitting')
            self.state = 'quit'
        else:
            print('Command {:s} not recognized'.format(command.get(0).asString()))
            reply.addString('Command {:s} not recognized'.format(command.get(0).asString()))
        
        return True


    def updateModule(self):

        if self.state == 'stream':
            item = self.lines[self.counter]
            item = item.rstrip()
            print(item)
            im_info = item.split('/')

            if os.path.exists(os.path.join(self.images_folder, im_info[0], 'rgb', im_info[1] + '.jpg')):
                image = np.array(Image.open(os.path.join(self.images_folder, im_info[0], 'rgb', im_info[1] + '.jpg')))
            elif os.path.exists(os.path.join(self.images_folder, im_info[0], 'rgb', im_info[1] + '.ppm')):
                image = np.array(Image.open(os.path.join(self.images_folder, im_info[0], 'rgb', im_info[1] + '.ppm')))
            elif os.path.exists(os.path.join(self.images_folder, im_info[0], 'rgb', im_info[1] + '.png')):
                image = np.array(Image.open(os.path.join(self.images_folder, im_info[0], 'rgb', im_info[1] + '.png')))
            else:
                print('Image {} not found in folder {}'.format(item, self.images_folder))

            self.out_buf_array[:, :] = image[:,:,0:3]
            self.output_image_port.write(self.out_buf_image)

            if self.counter >= len(self.lines)-1:
                print('The dataset is finished. Type start or resume to restart it.')
                self.counter = 0
                self.state = 'do_nothing'
            else:
                self.counter = self.counter + 1
                time.sleep(0.5)
        elif self.state == 'quit':
            self.cleanup()
            sys.exit("Ports closed. Quitting.")
        elif self.state == 'do_nothing':
            pass
        else:
            pass
        return True


if __name__ == '__main__':

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("assignment_DL-segmentation")
    conffile = rf.find("from").asString()
    if not conffile:
        print('Using default conf file')
        rf.setDefaultConfigFile('DLcls_forTest2.ini')
    else:
        rf.setDefaultConfigFile(rf.find("from").asString())

    rf.configure(sys.argv)

    # Run module
    player = iCWT_player()
    # try:
    player.runModule(rf)
    # finally:
    #     print('Closing SegmentationDrawer due to an error..')
    #     player.cleanup()
