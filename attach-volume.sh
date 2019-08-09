#!/bin/bash

aws ec2 attach-volume \
    --volume-id vol-01809c446642cf8ca \
    --instance-id i-08b954879653bafe6 \
    --device /dev/sdf
