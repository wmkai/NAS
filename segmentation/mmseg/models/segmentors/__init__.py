# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
from .dynamic_encoder_decoder import DynamicEncoderDecoder
from .static_encoder_decoder import StaticEncoderDecoder
__all__ = ['BaseSegmentor', 'EncoderDecoder', 'DynamicEncoderDecoder', 'StaticEncoderDecoder']
