#!/bin/bash

echo "Setting up instance..."
echo "Installing docker..."
sudo yum update -y
sudo amazon-linux-extras install docker
sudo yum install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
echo "Done!"
