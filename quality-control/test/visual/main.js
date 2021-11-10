const net = require('net');
const express = require('express');
const app = express();
const http = require('http');
const server = http.createServer(app);
const { Server } = require("socket.io");
const io = new Server(server);

let browser_socket;
let browser_data = Buffer.from([]);

const client = new net.Socket();
client.setTimeout(8000);

client.connect(8840, '192.168.1.92', () => {
  console.log('Connected');
});

client.on('close', () => {
  console.log('Connection closed');
});

client.on('data', (data) => {
  const pkt_len = 57600;

  const n_want = Math.min(pkt_len - browser_data.length, data.length);
  browser_data = Buffer.concat([browser_data, data.slice(0, n_want)]);

  if (browser_socket && browser_data.length == pkt_len) {
    browser_socket.emit('image_data', browser_data);
  }

  if (browser_data.length == pkt_len) {
    browser_data = Buffer.from([]);

    if (n_want < data.length) {
      browser_data = Buffer.concat([browser_data, data.slice(n_want)]);
    }
  }
});

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');
});

io.on('connection', (socket) => {
  console.log('a user connected');
  browser_socket = socket;
});

server.listen(8080, '127.0.0.1', () => {
  console.log('listening on http://127.0.0.1:8080');
});
