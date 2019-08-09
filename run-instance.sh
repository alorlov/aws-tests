#!/bin/bash

aws ec2 run-instances \
    --image-id ami-0ddba16a97b1dcda5 \
    --security-group-ids sg-00cb73578637361a2 \
    --count 1 \
    --instance-type t2.micro \
    --key-name keras-keypair2 \
    --subnet-id subnet-d69d94af \
#    --query Instances[0].InstanceId

