
import socket

host = '' 
port = 50000 
backlog = 5 

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

s.bind((host,port))

s.listen(backlog) 

client, address = s.accept() 

print "HELLO"



