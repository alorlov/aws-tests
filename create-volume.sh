#!/bin/bash

aws ec2 create-volume \
        --size 5 \
        --region us-west-2 \
        --availability-zone us-west-2b \
        --volume-type gp2 \
        --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=DL-datasets-checkpoints}]'
