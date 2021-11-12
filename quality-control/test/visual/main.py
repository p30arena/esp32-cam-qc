import socket
import eventlet
import socketio
import tensorflow as tf

closed = False
sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})


@sio.event
def connect(sid, environ):
    global closed
    pkt_len = 172800
    print('connect ', sid)
    browser_data = bytes()
    while not closed:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3.0)
            s.connect(('192.168.1.92', 8840))
            try:
                data = s.recv(pkt_len)
                n_want = min(pkt_len - len(browser_data), len(data))
                browser_data += data[0:n_want]

                if len(browser_data) == pkt_len:
                    sio.emit('image_data', data=browser_data, to=sid)

                if len(browser_data) == pkt_len:
                    browser_data = bytes()
                    if n_want < len(data):
                        browser_data += data[n_want:]
            except BaseException as err:
                if isinstance(err, socket.timeout):
                    continue
                else:
                    closed = True


@sio.event
def disconnect(sid):
    global closed
    print('disconnect ', sid)
    closed = True


if __name__ == '__main__':
    try:
        eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 8080)), app)
    except KeyboardInterrupt:
        closed = True
