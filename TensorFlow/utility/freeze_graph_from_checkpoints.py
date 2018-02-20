import os
import argparse

import tensorflow as tf
from tensorflow.python.framework import graph_util

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_checkpoint_graph(output_node_names, checkpoint_model_folder, output_graph_filename):
    # retrieve the checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(checkpoint_model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    # precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split("/")[:-1])

    # before exporting out graph - we need to find out what is our output node
    # we can have multiple output nodes
    # output_node_names = "Accuracy/predictions"

    # we clear devices, to allow tensorflow to control on the loading, where it wants operations to be calculated
    clear_devices = True

    # we import a meta graph and retrieve a saver
    # the checkpoint directory has - .meta and .data i.e. weights file to be retrieved
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # retrieve protobuf graph definition
    # returns the default graph of the current thread - will be the innermost graph
    # on which Graph.as_default() context has been entered - global_default_graph if non has been explicitly created
    graph = tf.get_default_graph()
    
    # retrieve graph def for a grpah
    input_graph_def = graph.as_graph_def()

    # print the output nodes 
    output_node_list = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # start the session and restore the weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        
        # in order to freeze the graph - need to export the variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
                sess,   # session have weights stored
                input_graph_def,
                output_node_names.split(",")
        )

        # convert variable to constant - possible to describe the network in a single graphdef - removes lot of operations such as loading and saving the variables
        # inputs are:
        #    1. sess : active tensorflow session containing the variables
        #    2. input_graph_def : graphdef holding the network
        #    3. output_node_names : list of name strings for the result nodes in the graph
        #    4. variable_names_whitelist : list of variables to convert
        #    5. variable_names_blacklist : list of variables not to convert

        # finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph_filename, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        print("[FREEZE_INFO] ", len(output_graph_def.node), " ops in the final graph.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_model_dir", type=str, help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, help="Output node names")
    parser.add_argument("--output_graph_filename", type=str, help="Frozen output file name")
    args = parser.parse_args()
    print("Freezing Checkpoint Model file with following configuration")
    print("Checkpoint Model Directory : ", args.checkpoint_model_dir)
    print("Output Node Names : ", args.output_node_names)
    print("Output Graph Filename : ", args.output_graph_filename)
    
    try:
        freeze_checkpoint_graph(args.output_node_names, args.checkpoint_model_dir, args.output_graph_filename)
        print("#### Success. Saved frozen graph")
    except:
        print("#### Failure. Unable to save frozen graph")
