import numpy as np

from GlobalConfig import Config
from Block import pad_frame, Block
from YUV_formater import YUV_Operator
from scipy.fftpack import idct
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
    
    def idct_transform(self, block):
        return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    def dequantize(self, quantized_coefficients, QP, i):
        Q = self.calculate_quantization_matrix(QP, i)
        return quantized_coefficients * Q
    
    def decode_i_frame(self, i_frame_data):
        frame_height = self.config.height
        frame_width = self.config.width
        block_size = self.config.block_size
        frames = []

        # TODO: 从i_frame_data中接收QP值，现在设置个default值
        QP = 0

        for i in range(0, frame_height, block_size):
            for j in range(0, frame_width, block_size):
                quantized_coefficients = i_frame_data.pop(0)  # 从帧数据中取出一个块的量化系数

                # 2. 对每个块执行反量化操作
                # TODO: 使用接收到的QP值进行反量化操作

                dct_coefficients = self.dequantize(quantized_coefficients, QP, block_size)

                # 3. 对每个块执行IDCT逆变换
                reconstructed_block = self.idct_transform(dct_coefficients)

                # 4. 将块重构为帧
                frames.append(reconstructed_block)

        # 5. TODO: 保存QP值以备将来使用

        return frames

    def calculate_quantization_matrix(self, QP, i):
        Q = np.zeros((i, i), dtype=float)
        for x in range(i):
            for y in range(i):
                if (x + y) < (i - 1):
                    Q[x][y] = 2 * QP
                elif (x + y) == (i - 1):
                    Q[x][y] = 2 * (QP + 1)
                else:
                    Q[x][y] = 2 * (QP + 2)
        return Q

    def decode(self, input_dir='Assignment/encoder_ws', ws='Assignment/decoder_ws'):
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
                    # TODO: 读取QP值，现在设置个default值
                    QP = 0

                    residual_frame = f_residaul_frames.read(reference_frame.size)
                    if not residual_frame:
                        break
                    residual_data = np.frombuffer(residual_frame, dtype=np.uint8).reshape(reference_frame.shape)
                    reconstructed_frame = np.zeros(reference_frame.shape, dtype=np.uint8)
                    i_frame_data = []  # 用于存储I帧的帧数据
                    for i in range(0, self.config.height, block_size):
                        for j in range(0, self.config.width, block_size):
                            motion_vector = np.frombuffer(f_mv.read(2), dtype=np.int8)
                            reference_block = Block(i + motion_vector[0], j + motion_vector[1], block_size, reference_frame)
                            residual_block = residual_data[i:i + block_size, j:j + block_size]
                            reconstructed_block = Block(i, j, block_size, None, reference_block.block_data + residual_block)
                            reconstructed_frame[i:i + block_size, j:j + block_size] = reconstructed_block.block_data
                            i_frame_data.append(reconstructed_block.block_data)

                    # 调用decode_i_frame方法，传递帧数据和QP值
                    i_frames = self.decode_i_frame(i_frame_data, QP)

                    # TODO: 使用i_frames进行进一步的处理或保存
                    
                    reference_frame = reconstructed_frame
                    frames.append(reconstructed_frame)
                    self.yuv_operator.convert_Y_to_png(reconstructed_frame, f'{ws}/png_sequence/Y_frame_reconstructed{k}.png')
                    k += 1

if __name__ == '__main__':
    decoder = Decoder(config)
    decoder.decode()