from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import time
import numpy as np
import picamera
import math
from threading import Thread
import socket
import select
import multiprocessing

from PIL import Image
from tflite_runtime.interpreter import Interpreter

running = True
conected_socket: socket.socket = None


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def load_labels():
    return {0: 'error', 1: 'ok'}


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    pred = sigmoid(output)
    label_id = 0 if pred < 0.5 else 1

    return label_id, pred


def worker(master_queue: multiprocessing.Queue, worker_queue: multiprocessing.Queue):
    labels = load_labels()
    interpreter = Interpreter('model.tflite')
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    worker_queue.put((width, height))

    while True:
        image = master_queue.get(True, None)
        if image == False:
            break
        start_time = time.time()
        label_id, prob = classify_image(interpreter, image)
        elapsed_ms = (time.time() - start_time) * 1000
        worker_queue.put((labels[label_id], prob, elapsed_ms))


def main():
    global running, conected_socket

    master_queue = multiprocessing.Queue()
    worker_queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(
        master_queue, worker_queue))
    p.start()

    width, height = worker_queue.get(True, None)

    try:
        with picamera.PiCamera(resolution=(width, height), framerate=30) as camera:
            stream = io.BytesIO()
            for _ in camera.capture_continuous(
                    stream, format='jpeg', use_video_port=True):
                if not running:
                    break
                stream.seek(0)
                image = Image.open(stream).convert('RGB')
                stream.seek(0)
                stream.truncate()
                image_bytes = image.tobytes()
                if conected_socket:
                    try:
                        conected_socket.sendall(image_bytes)
                    except (BrokenPipeError, ConnectionResetError):
                        conected_socket = None
                    except Exception as e:
                        print(e)
                master_queue.put(image)
                if worker_queue.qsize != 0:
                    label, prob, elapsed_ms = worker_queue.get()
                    print("{0} {1}\n{2}ms".format(label, prob, elapsed_ms))
    finally:
        master_queue.put(False)


def server_main():
    global conected_socket, running
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', 8840))
    s.listen(1)

    while running:
        readable, writable, errored = select.select([s], [], [], 1)
        if len(readable) != 1:
            continue
        conn, addr = s.accept()
        conected_socket = conn
    s.close()


if __name__ == '__main__':
    t: Thread = None
    try:
        t = Thread(target=server_main, args=())
        t.start()
        main()
    except KeyboardInterrupt:
        running = False
