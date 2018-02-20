import argparse
from tensorflow.python.tools import freeze_graph
from tensorflow.python.saved_model import tag_constants

def freeze_saved_model_graph(output_node_names, input_saved_model_dir, output_graph_filename):
    input_binary = False
    input_saver_def_path = False
    restore_op_name = None
    filename_tensor_name = None
    clear_devices = False
    input_meta_graph = False
    checkpoint_path = None
    input_graph_filename = None
    saved_model_tags = tag_constants.SERVING
    freeze_graph.freeze_graph(input_graph_filename, 
                              input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_filename, clear_devices, "", "", "",
                              input_meta_graph, input_saved_model_dir,
                              saved_model_tags)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", type=str, help="Saved Model Directory")
    parser.add_argument("--output_node_names", type=str, help="Output node names")
    parser.add_argument("--output_graph_filename", type=str, help="Frozen output file name")
    args = parser.parse_args()
    print("Freezing Saved Model file with following configuration")
    print("Saved Model Directory : ", args.saved_model_dir)
    print("Output Node Names : ", args.output_node_names)
    print("Output Graph Filename : ", args.output_graph_filename)
    
    try:
        freeze_saved_model_graph(args.output_node_names, args.saved_model_dir, args.output_graph_filename)
        print("#### Success. Saved frozen graph")
    except:
        print("#### Failure. Unable to save frozen graph")
