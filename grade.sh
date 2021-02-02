#!/usr/bin/env bash

# Runs the requested autograder:
#   ./grade.sh 1: hw1 autograder
#   ./grade.sh m: hw1 mnist
#   ./grade.sh 2: hw2 autograder
#   ./grade.sh 3: hw3 autograder
#   ./grade.sh 4: hw4 autograder
#   ./grade.sh 1b: hw1 bonus autograder
#   ./grade.sh 2b: hw2 bonus autograder
#   ./grade.sh 3b: hw3 bonus autograder
#   ./grade.sh 4b: hw4 bonus autograder

if [ -z $1 ]; then
    echo "./grade.sh 1: hw1 autograder"
    echo "./grade.sh m: hw1 mnist"
    echo "./grade.sh 1b: hw1 bonus autograder"
    echo "./grade.sh 2: hw2 autograder"
    echo "./grade.sh 2b: hw2 bonus autograder"
    echo "./grade.sh 3: hw3 autograder"
    echo "./grade.sh 3b: hw3 bonus autograder"
    echo "./grade.sh 4: hw4 autograder"
    echo "./grade.sh 4b: hw4 bonus autograder"
    exit
fi

if [ "$1" == "1" ]; then
    python3 ./autograder/hw1_autograder/runner.py
elif [ "$1" == "m" ]; then
    python3 ./autograder/hw1_autograder/test_mnist.py
elif [ "$1" == "1b" ]; then
    python3 ./autograder/hw1_bonus_autograder/runner.py
elif [ "$1" == "2" ]; then
    python3 ./autograder/hw2_autograder/runner.py
elif [ "$1" == "2b" ]; then
    python3 ./autograder/hw2_bonus_autograder/runner.py
elif [ "$1" == "3" ]; then
    python3 ./autograder/hw3_autograder/runner.py
elif [ "$1" == "3b" ]; then
    python3 ./autograder/hw3_bonus_autograder/runner.py
elif [ "$1" == "4" ]; then
    python3 ./autograder/hw4_autograder/runner.py
elif [ "$1" == "4b" ]; then
    python3 ./autograder/hw4_bonus_autograder/runner.py
fi
