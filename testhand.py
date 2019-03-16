import os
import cv2
import time
import argparse
import numpy as np
import subprocess as sp
import json
import tensorflow as tf
import scipy.misc
import operator


from queue import Queue
from threading import Thread
from utils.app_utils import FPS, HLSVideoStream, WebcamVideoStream, draw_boxes_and_labels
from object_detection.utils import label_map_util


from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_2d, plot_hand_3d
from pose.DeterminePositions import create_known_finger_poses, determine_position, get_position_name_with_pose_id
from pose.utils.FingerPoseEstimate import FingerPoseEstimate


known_finger_poses = None
sess = None

def parse_args():
    parser = argparse.ArgumentParser(description = 'Classify hand gestures from the set of images in folder')
    parser.add_argument('data_path', help = 'Path of folder containing images', type = str)
    parser.add_argument('--output-path', dest = 'output_path', type = str, default = None,
                        help = 'Path of folder where to store the evaluation result')
    parser.add_argument('--plot-fingers', dest = 'plot_fingers', help = 'Should fingers be plotted.(1 = Yes, 0 = No)', 
                        default = 1, type = int)
    # Threshold is used for confidence measurement of Geometry and Neural Network methods
    parser.add_argument('--thresh', dest = 'threshold', help = 'Threshold of confidence level(0-1)', default = 0.45,
                        type = float)
    parser.add_argument('--solve-by', dest = 'solve_by', default = 0, type = int,
                        help = 'Solve the keypoints of Hand3d by which method: (0=Geometry, 1=Neural Network, 2=SVM)')
    # If solving by neural network, give the path of PB file.
    parser.add_argument('--pb-file', dest = 'pb_file', type = str, default = None,
                        help = 'Path where neural network graph is kept.')
    # If solving by SVM, give the path of svc pickle file.
    parser.add_argument('--svc-file', dest = 'svc_file', type = str, default = None,
                        help = 'Path where SVC pickle file is kept.')         
    parser.add_argument('-strin', '--stream-input', dest="stream_in", action='store', type=str, default=None)
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=640, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=480, help='Height of the frames in the video stream.')
    parser.add_argument('-strout','--stream-output', dest="stream_out", help='The URL to send the livestreamed object detection to.')
             
    args = parser.parse_args()
    return args

def prepare_input(data_path, output_path):
    data_path = os.path.abspath(data_path)
    data_files = os.listdir(data_path)
    data_files = [os.path.join(data_path, data_file) for data_file in data_files]

    # If output path is not given, output will be stored in input folder.
    if output_path is None:
        output_path = data_path
    else:
        output_path = os.path.abspath(output_path)

    return data_files, output_path

def predict_by_geometry(keypoint_coord3d_v, known_finger_poses, threshold):
    fingerPoseEstimate = FingerPoseEstimate(keypoint_coord3d_v)
    fingerPoseEstimate.calculate_positions_of_fingers(print_finger_info = True)
    obtained_positions = determine_position(fingerPoseEstimate.finger_curled, 
                                        fingerPoseEstimate.finger_position, known_finger_poses,
                                        threshold * 10)

    score_label = 'Undefined'
    if len(obtained_positions) > 0:
        max_pose_label = max(obtained_positions.items(), key=operator.itemgetter(1))[0]
        if obtained_positions[max_pose_label] >= threshold:
            score_label = max_pose_label
    
    print(obtained_positions)
    return score_label

def worker(input_q, output_q):
    
    fps = FPS().start()
    image_tf = tf.placeholder(tf.float32, shape = (1, 240, 320, 3))
    hand_side_tf = tf.constant([[1.0, 1.0]])  # Both left and right hands included
    evaluation = tf.placeholder_with_default(True, shape = ())

    # build network
    net = ColorHandPose3DNetwork()
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
        keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    net.init(sess)
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_raw = scipy.misc.imresize(frame, (240, 320))
        image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)
        keypoint_coord3d_v = sess.run(keypoint_coord3d_tf, feed_dict = {image_tf: image_v})
        output_q.put(predict_by_geometry(keypoint_coord3d_v, known_finger_poses, 0.45))

    fps.stop()
    sess.close()


if __name__ == '__main__':
    args = parse_args()

    input_q = Queue(1)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    if (args.stream_in):
        print('Reading from hls stream.')
        video_capture = HLSVideoStream(src=args.stream_in).start()
    else:
        print('Reading from webcam.')
        video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()

    data_files, output_path = prepare_input(args.data_path, args.output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    known_finger_poses = create_known_finger_poses()

    # network input
    

    # Start TF
    
    net = ColorHandPose3DNetwork()
    # initialize network
    
    fps = FPS().start()

    while True:
        frame = video_capture.read()
        input_q.put(frame)

        t = time.time()

        if output_q.empty():
            pass  # fill up queue
        else:
            
            data = output_q.get()
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, data, (10, 200), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
            if args.stream_out:
                print('Streaming elsewhere!')
            else:
                cv2.imshow('Video', frame)

        fps.update()

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    video_capture.stop()
    cv2.destroyAllWindows()
