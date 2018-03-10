from keras_config.SSDConfig import SSDConfig
from keras.layers import Input, Conv2D, MaxPooling2D,GlobalAveragePooling2D,ZeroPadding2D,Flatten,Dense,Reshape, concatenate,Activation
from keras_layers.SSD import PriorBox, Normalize,BBoxUtility
from keras.models import Model
import keras.backend as K
from keras_losses.SSD import MultiboxLoss
import pickle
from keras_datareaders.ssd_voc_reader import XML_preprocessor
from keras_generators.SSD_Generator import Generator
from keras.callbacks import EarlyStopping
from utils.progressbar.keras.ProgressBarCallback import ProgressBarCallback
config = SSDConfig()

def SSD300v2(input_shape):
    input_layer = Input(shape=input_shape)

    # Block 1
    conv1_1 = Conv2D(64, (3, 3),name='conv1_1',padding='same',activation='relu')(input_layer)            
    conv1_2 = Conv2D(64, (3, 3),name='conv1_2',padding='same',activation='relu')(conv1_1)                
    pool1 = MaxPooling2D(name='pool1',pool_size=(2, 2),strides=(2, 2),padding='same' )(conv1_2)          

    # Block 2
    conv2_1 = Conv2D(128, (3, 3),name='conv2_1',padding='same',activation='relu')(pool1)                 
    conv2_2 = Conv2D(128, (3, 3),name='conv2_2',padding='same',activation='relu')(conv2_1)               
    pool2 = MaxPooling2D(name='pool2',pool_size=(2, 2),strides=(2, 2),padding='same')(conv2_2)           

    # Block 3
    conv3_1 = Conv2D(256, (3, 3),name='conv3_1',padding='same',activation='relu')(pool2)                 
    conv3_2 = Conv2D(256, (3, 3),name='conv3_2',padding='same',activation='relu')(conv3_1)               
    conv3_3 = Conv2D(256, (3, 3),name='conv3_3',padding='same',activation='relu')(conv3_2)               
    pool3 = MaxPooling2D(name='pool3',pool_size=(2, 2),strides=(2, 2),padding='same')(conv3_3)           

    # Block 4
    conv4_1 = Conv2D(512, (3, 3),name='conv4_1',padding='same',activation='relu')(pool3)                 
    conv4_2 = Conv2D(512, (3, 3),name='conv4_2',padding='same',activation='relu')(conv4_1)               
    conv4_3 = Conv2D(512, (3, 3),name='conv4_3',padding='same',activation='relu')(conv4_2)               
    pool4 = MaxPooling2D(name='pool4',pool_size=(2, 2),strides=(2, 2),padding='same')(conv4_3)           

    # Block 5
    conv5_1 = Conv2D(512, (3, 3),name='conv5_1',padding='same',activation='relu')(pool4)
    conv5_2 = Conv2D(512, (3, 3),name='conv5_2',padding='same',activation='relu')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3),name='conv5_3',padding='same',activation='relu')(conv5_2)
    pool5 = MaxPooling2D(name='pool5',pool_size=(3, 3),strides=(1, 1),padding='same')(conv5_3)

    # FC6
    fc6 = Conv2D(1024, (3, 3),name='fc6',dilation_rate=(6, 6),padding='same',activation='relu')(pool5)

    # FC7
    fc7 = Conv2D(1024, (1, 1),name='fc7',padding='same',activation='relu')(fc6)

    # Block 6
    conv6_1 = Conv2D(256, (1, 1),name='conv6_1',padding='same',activation='relu')(fc7)
    conv6_2 = Conv2D(512, (3, 3),name='conv6_2',strides=(2, 2),padding='same',activation='relu')(conv6_1)

    # Block 7
    conv7_1 = Conv2D(128, (1, 1),name='conv7_1',padding='same',activation='relu')(conv6_2)
    conv7_1z = ZeroPadding2D(name='conv7_1z')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3),name='conv7_2',padding='valid',strides=(2, 2),activation='relu')(conv7_1z)

    # Block 8
    conv8_1 = Conv2D(128, (1, 1),name='conv8_1',padding='same',activation='relu')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3),name='conv8_2',padding='same',strides=(2, 2),activation='relu')(conv8_1)

    # Last Pool
    pool6 = GlobalAveragePooling2D(name='pool6')(conv8_2)

    # Prediction from conv4_3
    num_priors = config.PRIORBOX[0]['num_prior']
    img_size = (input_shape[1], input_shape[0])
    name = 'conv4_3_norm_mbox_conf'
    if config.NUM_CLASS != 21:
        name += '_{}'.format(config.NUM_CLASS)

    '''
    conv4_3:?,38,38,512
    conv4_3_norm : ?,38,38,512
    conv4_3_norm_mbox_loc : ?,38,38,12
    conv4_3_norm_mbox_loc_flat : ?, 38*38*12 => ?, 17328
    conv4_3_norm_mbox_conf ： ?,38,38,63
    conv4_3_norm_mbox_conf_flat : ?,38*38*63
    conv4_3_norm_mbox_priorbox : ?,4332,8
    '''
    conv4_3_norm = Normalize(20, name='conv4_3_norm')(conv4_3)
    conv4_3_norm_mbox_loc = Conv2D(num_priors * 4, (3, 3),name='conv4_3_norm_mbox_loc',padding='same')(conv4_3_norm) # 4 代表的是4个坐标位置
    conv4_3_norm_mbox_loc_flat = Flatten(name='conv4_3_norm_mbox_loc_flat')(conv4_3_norm_mbox_loc)
    conv4_3_norm_mbox_conf = Conv2D(num_priors * config.NUM_CLASS, (3, 3),name=name,padding='same')(conv4_3_norm)
    conv4_3_norm_mbox_conf_flat = Flatten(name='conv4_3_norm_mbox_conf_flat')(conv4_3_norm_mbox_conf)
    conv4_3_norm_mbox_priorbox = PriorBox(img_size, config.PRIORBOX[0]['min_size'],name='conv4_3_norm_mbox_priorbox',aspect_ratios=[2],variances=[0.1, 0.1, 0.2, 0.2])(conv4_3_norm)

    # Prediction from fc7
    num_priors = config.PRIORBOX[1]['num_prior']
    name = 'fc7_mbox_conf'
    if config.NUM_CLASS != 21:
        name += '_{}'.format(config.NUM_CLASS)

    '''
    fc7:?,19,19,1024
    fc7_mbox_loc : ?,19,19,24
    fc7_mbox_loc_flat : ?, 19*19*24 => ?, 8664
    fc7_mbox_conf ： ?,19,19,126
    fc7_mbox_conf_flat : ?,19*19*126
    fc7_mbox_priorbox : ?,2166,8
    '''
    fc7_mbox_conf = Conv2D(num_priors * config.NUM_CLASS, (3, 3),padding='same',name=name)(fc7)
    fc7_mbox_conf_flat = Flatten(name='fc7_mbox_conf_flat')(fc7_mbox_conf)
    fc7_mbox_loc = Conv2D(num_priors * 4, (3, 3),name='fc7_mbox_loc',padding='same')(fc7)
    fc7_mbox_loc_flat = Flatten(name='fc7_mbox_loc_flat')(fc7_mbox_loc)
    fc7_mbox_priorbox = PriorBox(img_size, config.PRIORBOX[1]['min_size'],name='fc7_mbox_priorbox',max_size=config.PRIORBOX[1]['max_size'],
                                 aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2])(fc7)

    # Prediction from conv6_2
    num_priors = config.PRIORBOX[2]['num_prior']
    name = 'conv6_2_mbox_conf'
    if config.NUM_CLASS != 21:
        name += '_{}'.format(config.NUM_CLASS)

    '''
    conv6_2:?,10,10,512
    conv6_2_mbox_loc : ?,10,10,24
    conv6_2_mbox_loc_flat : ?, 10*10*24 => ?, 2400
    conv6_2_mbox_conf ： ?,10,10,126
    conv6_2_mbox_conf_flat : ?,10*10*126
    conv6_2_mbox_priorbox : ?,600,8
    '''
    conv6_2_mbox_conf = Conv2D(num_priors * config.NUM_CLASS, (3, 3),padding='same',name=name)(conv6_2)
    conv6_2_mbox_conf_flat = Flatten(name='conv6_2_mbox_conf_flat')(conv6_2_mbox_conf)
    conv6_2_mbox_loc = Conv2D(num_priors * 4, (3, 3,),name='conv6_2_mbox_loc',padding='same')(conv6_2)
    conv6_2_mbox_loc_flat = Flatten(name='conv6_2_mbox_loc_flat')(conv6_2_mbox_loc)
    conv6_2_mbox_priorbox = PriorBox(img_size, config.PRIORBOX[2]['min_size'],max_size=config.PRIORBOX[2]['max_size'],
                                     aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2],name='conv6_2_mbox_priorbox')(conv6_2)

    # Prediction from conv7_2
    num_priors = config.PRIORBOX[3]['num_prior']
    name = 'conv7_2_mbox_conf'
    if config.NUM_CLASS != 21:
        name += '_{}'.format(config.NUM_CLASS)

    '''
    conv7_2:?,5,5,256
    conv7_2_mbox_loc : ?,5,5,24
    conv7_2_mbox_loc_flat : ?, 5*5*24 => ?, 600
    conv7_2_mbox_conf ： ?,5,5,126
    conv7_2_mbox_conf_flat : ?,5*5*126
    conv7_2_mbox_priorbox : ?,150,8
    '''
    conv7_2_mbox_conf = Conv2D(num_priors * config.NUM_CLASS, (3, 3),padding='same',name=name)(conv7_2)
    conv7_2_mbox_conf_flat = Flatten(name='conv7_2_mbox_conf_flat')(conv7_2_mbox_conf)
    conv7_2_mbox_loc = Conv2D(num_priors * 4, (3, 3),padding='same',name='conv7_2_mbox_loc')(conv7_2)
    conv7_2_mbox_loc_flat = Flatten(name='conv7_2_mbox_loc_flat')(conv7_2_mbox_loc)
    conv7_2_mbox_priorbox = PriorBox(img_size, config.PRIORBOX[3]['min_size'],max_size=config.PRIORBOX[3]['max_size'],
                                     aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2],name='conv7_2_mbox_priorbox')(conv7_2)

    # Prediction from conv8_2
    num_priors = config.PRIORBOX[4]['num_prior']
    name = 'conv8_2_mbox_conf'
    if config.NUM_CLASS != 21:
        name += '_{}'.format(config.NUM_CLASS)

    '''
    conv8_2:?,3,3,256
    conv8_2_mbox_loc : ?,3,3,24
    conv8_2_mbox_loc_flat : ?, 3*3*24 => ?, 216
    conv8_2_mbox_conf ： ?,3,3,126
    conv8_2_mbox_conf_flat : ?,3*3*126
    conv8_2_mbox_priorbox : ?,54,8
    '''
    conv8_2_mbox_conf = Conv2D(num_priors * config.NUM_CLASS, (3, 3),padding='same',name=name)(conv8_2)
    conv8_2_mbox_conf_flat = Flatten(name='conv8_2_mbox_conf_flat')(conv8_2_mbox_conf)
    conv8_2_mbox_loc = Conv2D(num_priors * 4, (3, 3),padding='same',name='conv8_2_mbox_loc')(conv8_2)
    conv8_2_mbox_loc_flat = Flatten(name='conv8_2_mbox_loc_flat')(conv8_2_mbox_loc)
    conv8_2_mbox_priorbox = PriorBox(img_size, config.PRIORBOX[4]['min_size'],max_size=config.PRIORBOX[4]['max_size'],
                                     aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2],name='conv8_2_mbox_priorbox')(conv8_2)

    # Prediction from pool6
    num_priors = config.PRIORBOX[5]['num_prior']
    name = 'pool6_mbox_conf_flat'
    if config.NUM_CLASS != 21:
        name += '_{}'.format(config.NUM_CLASS)
    target_shape = (1, 1, 256)

    '''
    pool6:?,256
    pool6_mbox_loc_flat : ?, 24 => ?, 216
    conv8_2_mbox_conf ： ?,126
    pool6_mbox_priorbox : ?,6,8
    '''
    pool6_mbox_loc_flat = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(pool6)
    pool6_mbox_conf_flat = Dense(num_priors * config.NUM_CLASS, name=name)(pool6)
    pool6_reshaped = Reshape(target_shape,
                             name='pool6_reshaped')(pool6)
    pool6_mbox_priorbox = PriorBox(img_size,  config.PRIORBOX[5]['min_size'], max_size= config.PRIORBOX[5]['max_size'],
                                   aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2],name='pool6_mbox_priorbox')(pool6_reshaped)

    # Gather all predictions
    # mbox_loc => 17328 + 8664 + 2400 + 600 + 216 + 24 = 29232 => 7308 * 4
    mbox_loc = concatenate([conv4_3_norm_mbox_loc_flat,
                            fc7_mbox_loc_flat,
                            conv6_2_mbox_loc_flat,
                            conv7_2_mbox_loc_flat,
                            conv8_2_mbox_loc_flat,
                            pool6_mbox_loc_flat],
                           axis=1,name='mbox_loc')

    # mbox_conf => 90972 + 45486 + 12600 + 3150 + 1134 + 126 = 153468 => 7308 * 21
    mbox_conf = concatenate([conv4_3_norm_mbox_conf_flat,
                             fc7_mbox_conf_flat,
                             conv6_2_mbox_conf_flat,
                             conv7_2_mbox_conf_flat,
                             conv8_2_mbox_conf_flat,
                             pool6_mbox_conf_flat],
                            axis=1,name='mbox_conf')

    #mbox_priorbox => 4332 + 2166 + 600 + 150 + 54 + 6 = 7308
    mbox_priorbox = concatenate([conv4_3_norm_mbox_priorbox,
                                 fc7_mbox_priorbox,
                                 conv6_2_mbox_priorbox,
                                 conv7_2_mbox_priorbox,
                                 conv8_2_mbox_priorbox,
                                 pool6_mbox_priorbox],
                                axis=1,name='mbox_priorbox')

    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4

    mbox_loc = Reshape((num_boxes, 4),name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, config.NUM_CLASS),name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax',name='mbox_conf_final')(mbox_conf)
    predictions = concatenate([
        mbox_loc,# => ?,7308,4
        mbox_conf,# => ?,7308*21
        mbox_priorbox# => ?,7308*8
    ],axis=2,name='predictions') # ?,7308,33

    model = Model(inputs=input_layer, outputs=predictions)
    return model

NUM_CLASSES = len(config.VOC_CLASSES) + 1
input_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNEL)

model = SSD300v2(input_shape)

loss = MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss
model.compile(optimizer='Adadelta', loss=loss)

# plot_model(model, to_file="./assets/model.png", show_shapes=True)
# Image('./assets/model.png')

priors = pickle.load(open('../common/prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

pascal_voc_07_parser = XML_preprocessor(data_path='../data/VOC2007/Annotations/')
# len(pascal_voc_07_parser.data) = 5011

# pascal_voc_07_parser.data['000007.jpg']
# array([[ 0.282     ,  0.15015015,  1.        ,  0.99099099,  0.        ,
#          0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#          1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#          0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
#          0.        ,  0.        ,  0.        ,  0.        ]])

keys = list(pascal_voc_07_parser.data.keys())
train_num = int(0.7 * len(keys))
train_keys = keys[:train_num]
val_keys = keys[train_num:]

gen = Generator(gt=pascal_voc_07_parser.data, bbox_util=bbox_util,
                batch_size=config.BATCH_SIZE, path_prefix='../data/VOC2007/JPEGImages/',
                train_keys=train_keys, val_keys=val_keys, image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE))

EPOCHS = config.EPOCH_NUM

# tensorboard = TensorBoard(log_dir=LOG_DIR, write_images=True)
# checkpoint = ModelCheckpoint(filepath=LOG_FILE_PATH, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
probar = ProgressBarCallback()
history = model.fit_generator(generator=gen.generate(True), steps_per_epoch=int(gen.train_batches / 4),
                              validation_data=gen.generate(False), validation_steps=int(gen.val_batches / 4),
                              epochs=EPOCHS, verbose=0, callbacks=[early_stopping,probar])
model.save_weights("ssd.h5")
print("Saved model to disk")

