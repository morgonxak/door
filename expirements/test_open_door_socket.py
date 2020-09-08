import socket

sock = socket.socket()
sock.connect(('192.168.0.196', 9091))

# message = input("Сообщения ")

# sock.send(bytes('open', encoding='utf8'))
sock.send(bytes('disable_door', encoding='utf8'))
# sock.send(bytes('enable_door', encoding='utf8'))
data = sock.recv(1024)

sock.close()
