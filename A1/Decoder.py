import numpy as np

from GlobalConfig import Config
from Block import pad_frame, Block
from YUV_formater import YUV_Operator
config = Config()

class Decoder:
    def __init__(self, config:Config) -> None:
        self.config = config
        self.yuv_operator = YUV_Operator(config)

    def _construct_virtual_reference_frame(self, frame_shape, value):
        virtual_frame = np.full(frame_shape, value, dtype=np.uint8)
        return virtual_frame

    def _reconstruct_block_(self, predict_block:Block, residual_block:Block):
        return Block(residual_block.x, residual_block.y, residual_block.block_size, \
                     None, predict_block.block_data+residual_block.block_data)

    def decode(self, input_dir = 'Assignment/encoder_ws' ,ws = 'Assignment/decoder_ws'):
        block_size = self.config.block_size
        width = self.config.width
        height = self.config.height
        reference_frame = self._construct_virtual_reference_frame((self.config.height, self.config.width), 128)
        reference_frame = pad_frame(reference_frame, block_size)
        frames = []
        k = 0
        with open(f'{input_dir}/mv.13', 'rb') as f_mv:
            with open(f'{input_dir}/residual_frames.13', 'rb') as f_residaul_frames:
                while True:
                    residual_frame = f_residaul_frames.read(reference_frame.size)
                    if not residual_frame:
                        break
                    residual_data = np.frombuffer(residual_frame, dtype=np.uint8).reshape(reference_frame.shape)
                    reconstructed_frame = np.zeros(reference_frame.shape, dtype=np.uint8)
                    for i in range(0, self.config.height, block_size):
                        for j in range(0, self.config.width, block_size):
                            motion_vector = np.frombuffer(f_mv.read(2), dtype=np.int8)
                            reference_block = Block(i+motion_vector[0], j+motion_vector[1], block_size, reference_frame)
                            residual_block = residual_data[i:i+block_size, j:j+block_size]
                            reconstructed_block = Block(i, j, block_size, None, reference_block.block_data+residual_block)
                            reconstructed_frame[i:i+block_size, j:j+block_size] = reconstructed_block.block_data
                            # self.yuv_operator.visualize_YUV_in_text(reference_block.block_data)
                    reference_frame = reconstructed_frame
                    frames.append(reconstructed_frame)
                    self.yuv_operator.convert_Y_to_png(reconstructed_frame, f'{ws}/png_sequence/Y_frame_reconstructed{k}.png')
                    k += 1

if __name__ == '__main__':
    decoder = Decoder(config)
    decoder.decode()