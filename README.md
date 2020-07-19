# DarkflowYs

1)Downloading weights file and placing it in weights folder
wget https://pjreddie.com/media/files/yolov3.weights

2)Run convert_weights.py 
You should get the following files after running:
  1)checkpoint
  2)yolov3_weights.tf.data-00000-of-00001
  3)yolov3_weights.tf.index

3)Open image.py and change the image path before running it.

4)Open video.py and change the video path before running it.


Importing the necessary packages

Open the yolov3.py and import TensorFlow and Keras Model. We also import the layers from Keras, they are Conv2D, Input, ZeroPadding2D, LeakyReLU, and UpSampling2D. We’ll use them all when we build the YOLOv3 network.

Copy the following lines to the top of the file yolov3.py.

    #yolov3.py
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import BatchNormalization, Conv2D, \
        Input, ZeroPadding2D, LeakyReLU, UpSampling2D

Parsing the configuration file

The code below is a function called parse_cfg() with a parameter named cfgfile used to parse the YOLOv3 configuration fileyolov3.cfg.

    def parse_cfg(cfgfile):
        with open(cfgfile, 'r') as file:
            lines = [line.rstrip('\n') for line in file if line != '\n' and line[0] != '#']
        holder = {}
        blocks = []
        for line in lines:
            if line[0] == '[':
                line = 'type=' + line[1:-1].rstrip()
                if len(holder) != 0:
                    blocks.append(holder)
                    holder = {}
            key, value = line.split("=")
            holder[key.rstrip()] = value.lstrip()
        blocks.append(holder)
        return blocks

Let’s explain this code.

Lines 34-35, we open the cfgfile and read it, then remove unnecessary characters like ‘\n’ and ‘#’.

The variable lines in line 35 is now holding all the lines of the file yolov3.cfg. So, we need to loop over it in order to read every single line from it.

Lines 38-46, loop over the variable lines and read every single attribute from it and store them all in the list blocks. This process is performed by reading the attributes block per block. The block’s attributes and their values are firstly stored as the key-value pairs in a dictionary holder. After reading each block, all attributes are then appended to the list blocks and the holder is then made empty and ready to read another block. Loop until all blocks are read before returning the content of the list blocks.

All right!..we just finished a small piece of code. The next step is to create the YOLOv3 network function. Let’s do it..
Building the YOLOv3 Network

We’re still working on the file yolov3.py, the following is the code for the YOLOv3 network function, called the YOLOv3Net. We pass a parameter named cfgfile. So, Just copy and paste the following lines under the previous function parse_cfg().

    def YOLOv3Net(cfgfile, model_size, num_classes):
        blocks = parse_cfg(cfgfile)
        outputs = {}
        output_filters = []
        filters = []
        out_pred = []
        scale = 0
        inputs = input_image = Input(shape=model_size)
        inputs = inputs / 255.0

Let’s look at it…

Line 63, we first call the function parse_cfg() and store all the return attributes in a variable blocks. Here, the variable blocks contains all the attributes read from the file yolov3.cfg.

Lines 69-70, we define the input model using Keras function and divided by 255 to normalize it to the range of 0–1.

Next…

YOLOv3 has 5 layers types in general, they are: “convolutional layer”, “upsample layer”, “route layer”, “shortcut layer”, and “yolo layer”.

The following code performs an iteration over the list blocks. For every iteration, we check the type of the block which corresponds to the type of layer.

        for i, block in enumerate(blocks[1:]):

Convolutional Layer

In YOLOv3, there are 2 convolutional layer types, i.e with and without batch normalization layer. The convolutional layer followed by a batch normalization layer uses a leaky ReLU activation layer, otherwise, it uses the linear activation. So, we must handle them for every single iteration we perform.

This is the code to perform the convolutional layer.

            # If it is a convolutional layer
            if (block["type"] == "convolutional"):
                activation = block["activation"]
                filters = int(block["filters"])
                kernel_size = int(block["size"])
                strides = int(block["stride"])
                if strides > 1:
                    inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
                inputs = Conv2D(filters,
                                kernel_size,
                                strides=strides,
                                padding='valid' if strides > 1 else 'same',
                                name='conv_' + str(i),
                                use_bias=False if ("batch_normalize" in block) else True)(inputs)
                if "batch_normalize" in block:
                    inputs = BatchNormalization(name='bnorm_' + str(i))(inputs)
                    inputs = LeakyReLU(alpha=0.1, name='leaky_' + str(i))(inputs)

Line 93, we check whether the type of the block is a convolutional block, if it is true then read the attributes associated with it, otherwise, go check for another type ( we’ll be explaining after this). In the convolutional block, you’ll find the following attributes: batch_normalize, activation, filters, pad, size, and stride. For more details, what attributes are in the convolutional blocks, you can open the file yolov3.cfg.

Lines 98-99, verify whether the strideis greater than 1, if it is true, then downsampling is performed, so we need to adjust the padding.

Lines 106-108, if we find batch_normalizein a block, then add layers BatchNormalization and LeakyReLU, otherwise, do nothing.
Upsample Layer

Now, we’re going to continue if..else case above. Here, we’re going to check for the upsample layer. The upsample layer performs upsampling of the previous feature map by a factor of stride. To do this, YOLOv3 uses bilinear upsampling method.
So, if we find upsample block, retrieve the stride value and add a layer UpSampling2D by specifying the stride value.

The following is the code for that.

            elif (block["type"] == "upsample"):
                stride = int(block["stride"])
                inputs = UpSampling2D(stride)(inputs)

Route Layer

The route block contains an attribute layers which holds one or two values. For more details, please look at the file yolov3.cfg and point to lines 619-634. There, you will find the following lines.

    [route]
    layers = -4
    [convolutional]
    batch_normalize=1
    filters=256
    size=1
    stride=1
    pad=1
    activation=leaky
    [upsample]
    stride=2
    [route]
    layers = -1, 61

I’ll explain a little bit about the above lines of yolov3.cfg.

In the line 131 above, the attribute layers holds a value of -4 which means that if we are in this route block, we need to backward 4 layers and then output the feature map from that layer. However, for the case of the route block whose attribute layers has 2 values like in lines 141-142, layers contains -1 and 61, we need to concatenate the feature map from a previous layer (-1) and the feature map from layer 61. So, the following is the code for the route layer.

            # If it is a route layer
            elif (block["type"] == "route"):
                block["layers"] = block["layers"].split(',')
                start = int(block["layers"][0])
                if len(block["layers"]) > 1:
                    end = int(block["layers"][1]) - i
                    filters = output_filters[i + start] + output_filters[end]  # Index negatif :end - index
                    inputs = tf.concat([outputs[i + start], outputs[i + end]], axis=-1)
                else:
                    filters = output_filters[i + start]
                    inputs = outputs[i + start]

Shortcut Layer

In this layer, we perform skip connection. If we look at the file yolov3.cfg, this block contains an attribute from as shown below.

    [shortcut]
    from=-3
    activation=linear

What we’re going to do in this layer block is to backward 3 layers (-3) as indicated in from value, then take the feature map from that layer, and add it with the feature map from the previous layer. Here is the code for that.

            elif block["type"] == "shortcut":
                from_ = int(block["from"])
                inputs = outputs[i - 1] + outputs[i + from_]

Yolo Layer

Here, we perform our detection and do some refining to the bounding boxes. If you have any difficulty understanding or have a problem with this part, just check out my previous post (part-1 of this tutorial).

As we did to other layers, just check whether we’re in the yolo layer.

            # Yolo detection layer
            elif block["type"] == "yolo":

If it is true, then take all the necessary attributes associated with it. In this case, we just need mask and anchors attributes.

                mask = block["mask"].split(",")
                mask = [int(x) for x in mask]
                anchors = block["anchors"].split(",")
                anchors = [int(a) for a in anchors]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]
                n_anchors = len(anchors)

Then we need to reshape the YOLOv3 output to the form of [None, B * grid size * grid size, 5 + C]. The B is the number of anchors and C is the number of classes.

                out_shape = inputs.get_shape().as_list()
                inputs = tf.reshape(inputs, [-1, n_anchors * out_shape[1] * out_shape[2], \
               5 + num_classes])

Then access all boxes attributes by this way:

                box_centers = inputs[:, :, 0:2]
                box_shapes = inputs[:, :, 2:4]
                confidence = inputs[:, :, 4:5]
                classes = inputs[:, :, 5:num_classes + 5]

Refine Bounding Boxes

As I mentioned in part 1 that after the YOLOv3 network outputs the bounding boxes prediction, we need to refine them in order to the have the right positions and shapes.

Use the sigmoid function to convert box_centers, confidence, and classes values into range of 0 – 1.

                box_centers = tf.sigmoid(box_centers)
                confidence = tf.sigmoid(confidence)
                classes = tf.sigmoid(classes)

Then convert box_shapes as the following:

                anchors = tf.tile(anchors, [out_shape[1] * out_shape[2], 1])
                box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)

Use a meshgrid to convert the relative positions of the center boxes into the real positions.

                x = tf.range(out_shape[1], dtype=tf.float32)
                y = tf.range(out_shape[2], dtype=tf.float32)
                cx, cy = tf.meshgrid(x, y)
                cx = tf.reshape(cx, (-1, 1))
                cy = tf.reshape(cy, (-1, 1))
                cxy = tf.concat([cx, cy], axis=-1)
                cxy = tf.tile(cxy, [1, n_anchors])
                cxy = tf.reshape(cxy, [1, -1, 2])
                strides = (input_image.shape[1] // out_shape[1], \
                           input_image.shape[2] // out_shape[2])
                box_centers = (box_centers + cxy) * strides

Then, concatenate them all together.

                prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

Big note: Just to remain you that YOLOv3 does 3 predictions across the scale. We do as it is.

Take the prediction result for each scale and concatenate it with the others.

                if scale:
                    out_pred = tf.concat([out_pred, prediction], axis=1)
                else:
                    out_pred = prediction
                    scale = 1

Since the route and shortcut layers need output feature maps from previous layers, so for every iteration, we always keep the track of the feature maps and output filters.

            outputs[i] = inputs
            output_filters.append(filters)

Finally, we can return our model.

        model = Model(input_image, out_pred)
        model.summary()
        return model
        



convert_weights.py

Open the file convert_weights.py, then copy and paste the following code to the top of it. Here, we import NumPy library and the two functions that we’ve created previously in part 2, YOLOv3Net and parse_cfg.

    #convert_weights.py
    import numpy as np
    from yolov3 import YOLOv3Net
    from yolov3 import parse_cfg

Now, let’s create a function called load_weights(). This function has 3 parameters, model, cfgfile, and weightfile. The parameter model is a returning parameters of the network’s model after calling the function YOLOv3Net. Thecfgfile and weightfile are respectively refer to the files yolov3.cfg and yolov3.weights.

    def load_weights(model,cfgfile,weightfile):

Open the file yolov3.weights and read the first 5 values. These values are the header information. So, we can skip them all.

        # Open the weights file
        fp = open(weightfile, "rb")
        # The first 5 values are header information
        np.fromfile(fp, dtype=np.int32, count=5)

Then call parse_cfg() function.

        blocks = parse_cfg(cfgfile)

As we did when building the YOLOv3 network, we need to loop over the blocks and search for the convolutional layer. Don’t forget to check whether the convolutional is with batch normalization or not. If it is true, go get the relevant values (gamma, beta, means, and variance), and re-arrange them to the TensorFlow weights order. Otherwise, take the bias values. After that take the convolutional weights and set these weights to the convolutional layer depending on the convolutional type.

        for i, block in enumerate(blocks[1:]):
            if (block["type"] == "convolutional"):
                conv_layer = model.get_layer('conv_' + str(i))
                print("layer: ",i+1,conv_layer)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]
                if "batch_normalize" in block:
                    norm_layer = model.get_layer('bnorm_' + str(i))
                    print("layer: ",i+1,norm_layer)
                    size = np.prod(norm_layer.get_weights()[0].shape)
                    bn_weights = np.fromfile(fp, dtype=np.float32, count=4 * filters)
                    # tf [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                else:
                    conv_bias = np.fromfile(fp, dtype=np.float32, count=filters)
                # darknet shape (out_dim, in_dim, height, width)
                conv_shape = (filters, in_dim, k_size, k_size)
                conv_weights = np.fromfile(
                    fp, dtype=np.float32, count=np.product(conv_shape))
                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(
                    conv_shape).transpose([2, 3, 1, 0])
                if "batch_normalize" in block:
                    norm_layer.set_weights(bn_weights)
                    conv_layer.set_weights([conv_weights])
                else:
                    conv_layer.set_weights([conv_weights, conv_bias])

Alert if the reading has failed. Then, close the file whether the reading was successful or not.

        assert len(fp.read()) == 0, 'failed to read all data'
        fp.close()

The last part of this code is the main function. Copy and paste the following code of the main function just right after the function load_weights().

    def main():
        weightfile = "weights/yolov3.weights"
        cfgfile = "cfg/yolov3.cfg"
        model_size = (416, 416, 3)
        num_classes = 80
        model=YOLOv3Net(cfgfile,model_size,num_classes)
        load_weights(model,cfgfile,weightfile)
        try:
            model.save_weights('weights/yolov3_weights.tf')
            print('\nThe file \'yolov3_weights.tf\' has been saved successfully.')
        except IOError:
            print("Couldn't write the file \'yolov3_weights.tf\'.")
            
          

utils.py

The file utils.py will contain the useful functions that we’ll be creating soon, they are: non_maximum_suppression(), resize_image(), output_boxes(), and draw_output().

Open the file utils.py and import the necessary packages as the following:

    import tensorflow as tf
    import numpy as np
    import cv2

non_max_suppression()

Now, we’re going to create the function non_max_suppression(). If you forget about what the non-maximum suppression is, just go back to our first part of this tutorial and read it carefully.

Here, we’re not going to develop NMS algorithm from scratch. Instead, we leverage the TensorFlow’s built-in NMS function, tf.image.combined_non_max_suppression.

Here is the code for the non_max_suppression() function:

    def non_max_suppression(inputs, model_size, max_output_size, 
                            max_output_size_per_class, iou_threshold, confidence_threshold):
        bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
        bbox=bbox/model_size[0]
        scores = confs * class_probs
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
            scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
            max_output_size_per_class=max_output_size_per_class,
            max_total_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=confidence_threshold
        )
        return boxes, scores, classes, valid_detections

resize_image()

We resize the image to fit with the model’s size.

    def resize_image(inputs, modelsize):
        inputs= tf.image.resize(inputs, modelsize)
        return inputs

load_class_names()

The following is the code for function load_class_names().

    def load_class_names(file_name):
        with open(file_name, 'r') as f:
            class_names = f.read().splitlines()
        return class_names

output_boxes()

This function is used to convert the boxes into the format of (top-left-corner, bottom-right-corner), following by applying the NMS function and returning the proper bounding boxes.

    def output_boxes(inputs,model_size, max_output_size, max_output_size_per_class, 
                     iou_threshold, confidence_threshold):
        center_x, center_y, width, height, confidence, classes = \
            tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
        top_left_x = center_x - width / 2.0
        top_left_y = center_y - height / 2.0
        bottom_right_x = center_x + width / 2.0
        bottom_right_y = center_y + height / 2.0
        inputs = tf.concat([top_left_x, top_left_y, bottom_right_x,
                            bottom_right_y, confidence, classes], axis=-1)
        boxes_dicts = non_max_suppression(inputs, model_size, max_output_size, 
                                          max_output_size_per_class, iou_threshold, confidence_threshold)
        return boxes_dicts

draw_outputs()

Finally, we create a function to draw the output.

    def draw_outputs(img, boxes, objectness, classes, nums, class_names):
        boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
        boxes=np.array(boxes)
        for i in range(nums):
            x1y1 = tuple((boxes[i,0:2] * [img.shape[1],img.shape[0]]).astype(np.int32))
            x2y2 = tuple((boxes[i,2:4] * [img.shape[1],img.shape[0]]).astype(np.int32))
            img = cv2.rectangle(img, (x1y1), (x2y2), (255,0,0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(
                class_names[int(classes[i])], objectness[i]),
                              (x1y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            return img
            
 
 
 
