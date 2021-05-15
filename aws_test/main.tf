provider "aws" {
  profile = "fourthbrain"
  region  = "us-west-2"
}


data "aws_ami" "amazon-linux-2" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}


resource "aws_instance" "kbchatter-t2microtest" {
  ami                         = data.aws_ami.amazon-linux-2.id
  instance_type               = "t2.micro"
  key_name                    = "flaskDeployment1"
  associate_public_ip_address = true
  security_groups             = ["kbchatter"]
  iam_instance_profile        = "EC2S3"
  user_data                   = file("user_data.sh")

  root_block_device {
    delete_on_termination = true
    volume_size           = 32
    volume_type           = "standard"
  }

  tags = {
    application = "knowledge base chatter bot"
  }
}


output "instance_public_dns" {
  value = aws_instance.kbchatter-t2microtest.public_dns
}
