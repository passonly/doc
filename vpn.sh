#!/bin/bash
## 关闭selinux
setenforce 0
sed -i "s/SELINUX=enforcing/SELINUX=disabled/" /etc/selinux/config
## 开启内核转发
grep -qF "net.ipv4.ip_forward" /etc/sysctl.conf  || echo "net.ipv4.ip_forward = 1" >> /etc/sysctl.conf
sysctl -p
## 关闭防火墙
systemctl stop firewalld
systemctl disable firewalld
## 安装openvpn
yum install -y epel-release
yum update -y
yum install -y openssl lzo pam openssl-devel lzo-devel pam-devel
yum install -y easy-rsa
yum install -y openvpn
## 配置openvpn服务器
mkdir -p /var/log/openvpn/
mkdir -p /etc/openvpn/server/user
chown openvpn:openvpn /var/log/openvpn
## 生成证书
cp -rf /usr/share/easy-rsa/3.0.8 /etc/openvpn/server/easy-rsa
## 为服务器创建证书
cd /etc/openvpn/server/easy-rsa
## 进行初始化，会在当前目录创建PKI目录，用来存储一些中间变量和最终生成的证书
./easyrsa init-pki
## 创建证书
./easyrsa build-ca nopass
## 生成服务器端证书
./easyrsa build-server-full server nopass
## 确保key可以穿越不安全网络的命令
./easyrsa gen-dh
## 创建ta.key够加强认证方式，防止攻击
openvpn --genkey --secret ta.key
## 将证书放到对应的目录
cp -a pki/ca.crt /etc/openvpn/server/
cp -a pki/private/server.key /etc/openvpn/server
cp -a pki/issued/server.crt /etc/openvpn/server
cp -a pki/dh.pem /etc/openvpn/server
cp -a ta.key /etc/openvpn/server

## 创建客户端证书
## 生成无密码的客户端证书
./easyrsa build-client-full zhangsan nopass

## 配置server.conf
echo "local 0.0.0.0
port 1194
proto tcp
dev tun
ca /etc/openvpn/server/ca.crt
cert /etc/openvpn/server/server.crt
key /etc/openvpn/server/server.key
dh /etc/openvpn/server/dh.pem
server 10.8.0.0 255.255.255.0
ifconfig-pool-persist ipp.txt
push \"route 10.10.10.0 255.255.255.0\"
push \"redirect-gateway def1 bypass-dhcp\"
push \"dhcp-option DNS 8.8.8.8\"
push \"dhcp-option DNS 8.8.4.4\"
push \"dhcp-option DNS 1.1.1.1\"
keepalive 10 120
tls-auth /etc/openvpn/server/ta.key 0
cipher AES-256-CBC
push \"compress lz4-v2\"
user nobody
group nobody
persist-key
persist-tun
status openvpn-status.log
log /var/log/openvpn.log
verb 3" > /etc/openvpn/server/server.conf

## 启动openvpn并配置开机启动
## 修改服务文件的名称
cp /usr/lib/systemd/system/openvpn-server\@.service /usr/lib/systemd/system/openvpn.service
## 编辑服务文件名称
echo "[Unit]
Description=OpenVPN service for %I
After=syslog.target network-online.target
Wants=network-online.target
Documentation=man:openvpn(8)
Documentation=https://community.openvpn.net/openvpn/wiki/Openvpn24ManPage
Documentation=https://community.openvpn.net/openvpn/wiki/HOWTO

[Service]
Type=notify
PrivateTmp=true
WorkingDirectory=/etc/openvpn/server
ExecStart=/usr/sbin/openvpn --status %t/openvpn-server/status-%i.log --status-version 2 --suppress-timestamps --config server.conf
CapabilityBoundingSet=CAP_IPC_LOCK CAP_NET_ADMIN CAP_NET_BIND_SERVICE CAP_NET_RAW CAP_SETGID CAP_SETUID CAP_SYS_CHROOT CAP_DAC_OVERRIDE CAP_AUDIT_WRITE
LimitNPROC=10
DeviceAllow=/dev/null rw
DeviceAllow=/dev/net/tun rw
ProtectSystem=true
ProtectHome=true
KillMode=process
RestartSec=5s
Restart=on-failure

[Install]
WantedBy=multi-user.target
" > /usr/lib/systemd/system/openvpn.service 

## 启动openvpn 
systemctl start openvpn
systemctl enable openvpn


##配置防火墙

iptables -t nat -A POSTROUTING -s 10.8.0.0/24 -o eth0 -j MASQUERADE
iptables-save > /etc/sysconfig/iptables
iptables -L -n -t nat

iptables -t nat -A POSTROUTING -s 10.10.10.0/24 -o eth0 -j MASQUERADE


