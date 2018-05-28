import numpy as np
import cv2
import tensorflow as tf


# Threhold used for prediction
PREDICT_THRESHOLD = 0.3


def get_model_graph(model_path):
    """
    Load the downloaded Tensorflow model into memory.
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def get_tf_tensors(graph):
    """
    Get handles to input and output tensors.
    """
    ops = graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['detection_boxes', 'detection_scores', 'detection_classes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = graph.get_tensor_by_name(
                tensor_name)
    input_image_tensor = graph.get_tensor_by_name('image_tensor:0')
    return tensor_dict, input_image_tensor


def get_graph_tensors(model_path):
    """
    Load model into memory and get the inputs.
    """
    graph = get_model_graph(model_path)
    tensor_dict, input_image_tensor = get_tf_tensors(graph)
    return graph, tensor_dict, input_image_tensor


def run_tf_inference(image, session,
                     tensor_dict, input_image_tensor):
    """
    Run tf inference to detect bounding boxes
    """
    output_dict = session.run(
        tensor_dict,
        feed_dict={input_image_tensor: np.expand_dims(image, 0)})
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict


def process_image(image, session, tensor_dict,
                  input_image_tensor):
    """
    Process a single frame for bounding box detection.
    """
    output_dict = run_tf_inference(image, session, tensor_dict, input_image_tensor)
    for i in range(100):
        # Assume predictions are ordered by probability
        if output_dict['detection_scores'][i] < PREDICT_THRESHOLD:
            break
        ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i,:]
        ymin_pix = int(ymin*image.shape[0])
        xmin_pix = int(xmin*image.shape[1])
        ymax_pix = int(ymax*image.shape[0])
        xmax_pix = int(xmax*image.shape[1])
        cv2.rectangle(image, (xmin_pix,ymin_pix), (xmax_pix,ymax_pix), (0,255,0), 3)
    return image
