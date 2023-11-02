import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential

def get_model(hyper_params):
    """Generating rpn model for given hyper params.
    inputs:
        hyper_params = dictionary

    outputs:
        rpn_model = tf.keras.model
        feature_extractor = feature extractor layer from the base model
    """
    img_size = hyper_params["img_size"]
    base_model = ResNet50(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet')
    feature_extractor = base_model.get_layer("conv4_block6_out")
    output = Conv2D(512, (3, 3), activation="relu", padding="same", name="rpn_conv")(feature_extractor.output)
    rpn_cls_output = Conv2D(hyper_params["anchor_count"], (1, 1), activation="sigmoid", name="rpn_cls")(output)
    rpn_reg_output = Conv2D(hyper_params["anchor_count"] * 4, (1, 1), activation="linear", name="rpn_reg")(output)
    rpn_model = Model(inputs=base_model.input, outputs=[rpn_reg_output, rpn_cls_output])

    print('[INFO BACKBONE]')
    base_model.trainable = False
    base_model.summary()
    return rpn_model, feature_extractor

def init_model(model):
    """Initializing model with dummy data for load weights with optimizer state and also graph construction.
    inputs:
        model = tf.keras.model
    """
    model(tf.random.uniform((1, 500, 500, 3)))
