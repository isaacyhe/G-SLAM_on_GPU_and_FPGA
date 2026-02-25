#!/bin/bash

{ sleep 3; } &
FOO=$!
{ sudo powerstat -R 0.5; } &
wait $FOO
sudo pkill -P $$
