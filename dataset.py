from __future__ import print_function
from six.moves import xrange
import numpy as np
import tensorflow as tf
import random
import os, glob
from tqdm import tqdm
 
 
class CitySpaces():
    def __init__(self,
                 data_dir='datasets/cityspaces',
                 image_size=(512,256),
                 seed=0):
 
        #import label and images from datasets
        searchFine = os.path.join( data_dir, 'CameraSeg','*.png')
        labels = sorted(glob.glob( searchFine))
 
        searchFine = os.path.join( data_dir, 'CameraRGB','*.png' )
        images = sorted(glob.glob( searchFine ))
 
 
        #searchFine = os.path.join( data_dir, "gtFine", 'train' , "*" , "*labelTrainIds*.png" )
        #labels = sorted(glob.glob( searchFine ))

        #searchFine = os.path.join( data_dir, "leftImg8bit", 'train' , "*" , "*leftImg8bit*.png" )
        #images = sorted(glob.glob( searchFine ))
        #for l,i in zip(labels[target],images[target]):
        #    assert( ''.join(os.path.basename(l).split('_')[:2]) ==
        #                        ''.join(os.path.basename(i).split('_')[:2])), (l,i)
 
        #assert( len(labels) == 93 and len(images) == 93)
 
        self.images = images
        self.image_size = image_size
        self.labels = labels
 
 
 
    def build_queue(self,crop=(128,256),resize=(128,256),z_range=0.05,batch_size=2,num_threads=1):
        with tf.device('/cpu'):
                    
            im_name,l_name = tf.train.slice_input_producer([self.images,self.labels],num_epochs=None,shuffle=True)
        

            binary = tf.read_file(im_name)
            #Tensor("DecodePng:0", shape=(?, ?, 3), dtype=uint8, device=/device:CPU:*)
            image = tf.image.decode_png(binary,channels=3)

            binary = tf.read_file(l_name) 
            #Tensor("DecodePng_1:0", shape=(?, ?, 1), dtype=uint8, device=/device:CPU:*)  
            label = tf.image.decode_png(binary,channels = 1)


            #Tensor("split:0", shape=(512, 1024, 3), dtype=uint8, device=/device:CPU:*)
            #Tensor("split:1", shape=(512, 1024, 1), dtype=uint8, device=/device:CPU:*)
            cropped = tf.random_crop(tf.concat([image,label],axis=2),list(crop)+[4])
            cropped_im,cropped_label = tf.split(cropped,[3,1],axis=2)

            #Tensor("Squeeze:0", shape=(256, 512, 3), dtype=float32, device=/device:CPU:*)
            #Tensor("Squeeze_1:0", shape=(256, 512, 1), dtype=uint8, device=/device:CPU:*)
            resized_im = tf.image.resize_images(cropped_im,resize)
            resized_label = tf.image.resize_images(cropped_label,resize,tf.image.ResizeMethod.NEAREST_NEIGHBOR)


            # if coin < 0.5   flip horizontally  if coin > 0.5 keep it
            coin = tf.random_uniform([], 0., 1.0)
            pp = tf.cond(tf.less(coin,.5),
                        lambda: tf.image.flip_left_right(resized_im),
                        lambda: resized_im)
            # Tensor("cond/Merge:0", shape=(256, 512, 3), dtype=float32, device=/device:CPU:*)


            # if coin < 0.5 flip horizontally if coin > 0.5 keep it
            resized_label = tf.cond(tf.less(coin,.5),
                        lambda: tf.image.flip_left_right(resized_label),
                        lambda: resized_label)
            
            # resize_label :Tensor("cond_1/Merge:0", shape=(256, 512, 1), dtype=uint8, device=/device:CPU:*)
           
            # Gamma augmentation; formula (14)
            z = tf.random_uniform([],minval=-1.*z_range,maxval=z_range)
            gamma = tf.log(0.5+2**(-0.5)*z) / tf.log(0.5-2**(-0.5)*z)

            pp = (tf.cast(pp,tf.float32) / 255.0)**(gamma)
            #pp : Tensor("pow:0", shape=(256, 512, 3), dtype=float32, device=/device:CPU:*)


            # convert 255 to label 19.
            mask = tf.cast(tf.equal(resized_label, 255),tf.int32)
            resized_label = mask * 19 + (1-mask) * tf.cast(resized_label,tf.int32)
            resized_label = tf.squeeze(resized_label,axis=2)
           
 
            # Build task batch
            #if (target == 'train'):
            imnames, x, y = tf.train.batch(
                    [im_name,pp, resized_label],
                    batch_size=batch_size,
                    num_threads=num_threads,
                    capacity=10*batch_size,
                    allow_smaller_final_batch=True)
            
            #Tensor("batch:0", shape=(?,), dtype=string, device=/device:CPU:*)
            #Tensor("batch:1", shape=(?, 256, 512, 3), dtype=float32, device=/device:CPU:*)
            #Tensor("batch:2", shape=(?, 256, 512), dtype=int32, device=/device:CPU:*)

            return imnames,x,y
             
 
if __name__ == "__main__":
    cityspaces = CitySpaces()
 
    imnames, images, labels = cityspaces.build_queue()

    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)
 
    import itertools
    try:
        # Start Queueing
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        for _it in tqdm(itertools.count()) : # Slice Input producer will throw OutOfRange exception
            if( coord.should_stop() ): break
            names,ims,las = sess.run([imnames,images,labels])
            print(names,ims.shape,np.min(ims),np.max(ims),las.shape,np.min(las),np.max(las))
    except Exception as e:
        coord.request_stop(e)
    finally :
        coord.request_stop()
        coord.join(threads)

