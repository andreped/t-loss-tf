#!/bin/bash
isort --sl t_loss
black --line-length 120 t_loss
flake8 t_loss