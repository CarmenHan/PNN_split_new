#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 21:37:36 2020

@author: haoxuanwang
"""


import argparse


class LengthCheckAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) != namespace.layers:
            msg = "Sizes must have length L (number of layers). L={}, got {} values"
            parser.error(msg.format(namespace.layers, len(values)))

        setattr(namespace, self.dest, values)