import numpy as np

from GlobalConfig import Config
from YUV_formater import YUV_Operator
from Block import Block, pad_frame, PSNR, SSIM
from scipy.fftpack import dct, idct
config = Config()

class Encoder:
    def __init__(self) -> None:
        self.config = config
        self.yuv_operator = YUV_Operator(self.config)
    

    def _construct_virtual_reference_frame(self, frame_shape, value):
        virtual_frame = np.full(frame_shape, value, dtype=np.uint8)
        return virtual_frame
    

    def _simple_predict_Y_(self, block:Block, reference_frame, search_range):
        best_mae = float('inf')
        motion_vector = np.array([0, 0], dtype=np.int8)
        best_matching_block = Block(block.x, block.y, block.block_size, reference_frame)
        frame_height, frame_width = reference_frame.shape
        for dx in range(-search_range, search_range+1):
            for dy in range(-search_range, search_range+1):
                temp_x = block.x + dx 
                temp_y = block.y + dy
                # frame boundary
                if temp_x >= 0 and temp_x <= frame_height - block.block_size \
                    and temp_y >= 0 and temp_y <= frame_width-block.block_size:
                    temp_block = Block(temp_x, temp_y, block.block_size, reference_frame)
                    temp_mae = np.abs(block.block_data - temp_block.block_data).mean()

                    if temp_mae < best_mae:
                        best_mae = temp_mae
                        motion_vector = np.array([dx, dy], dtype=np.int8)
        best_matching_block = Block(block.x+motion_vector[0], block.y+motion_vector[1], block.block_size, reference_frame)
        residual_block = Block(block.x, block.y, block.block_size, None, block.block_data-best_matching_block.block_data)
        return motion_vector, best_matching_block, residual_block
    

    def _approximate_residual_block(self, residual_block:Block):
        factor = 2**self.config.residual_n
        residual_block.block_data = np.uint8(np.round(residual_block.block_data/factor) * factor)
        return residual_block


    def _reconstruct_block_(self, predict_block:Block, residual_block:Block):
        return Block(residual_block.x, residual_block.y, residual_block.block_size, \
                     None, predict_block.block_data+residual_block.block_data)
    
    
    def dct_transform(self, block):
        return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    

    def idct_transform(self, block):
        return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    

    def quantize(self, coefficients, QP, i):
        Q = self.calculate_quantization_matrix(QP, i)
        return np.round(coefficients / Q)
    

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
    

    def encode_i_frame(self, frame_data, QP):
        frame_height = self.config.height
        frame_width = self.config.width
        block_size = self.config.block_size
        frames = []

        # 1. 将帧数据分成(i x i)块
        for i in range(0, frame_height, block_size):
            for j in range(0, frame_width, block_size):
                block = Block(i, j, block_size, frame_data)

                # 2. 对每个块执行DCT变换
                dct_coefficients = self.dct_transform(block.block_data)

                # 3. 量化DCT系数
                quantized_coefficients = self.quantize(dct_coefficients, QP, block_size)

                # 4. TODO 保存QP用于传输给解码器
                # 可以将QP和quantized_coefficients一起存储到一个数据结构中，然后添加到frames列表中
                frame_info = {'QP': QP, 'coefficients': quantized_coefficients}
                frames.append(frame_info)

        return frames
    

    def encode_p_frame(self, Y_frame, reference_frame, motion_vectors, QP):
        frame_height = self.config.height
        frame_width = self.config.width
        block_size = self.config.block_size
        frames = []

        for i in range(0, frame_height, block_size):
            for j in range(0, frame_width, block_size):
                block = Block(i, j, block_size, Y_frame)

                # 获取运动矢量
                motion_vector = motion_vectors.pop(0)

                # 使用运动矢量和参考帧进行运动补偿
                reference_block = Block(i + motion_vector[0], j + motion_vector[1], block_size, reference_frame)
                residual_block = block.block_data - reference_block.block_data

                # 执行DCT变换
                dct_coefficients = self.dct_transform(residual_block)

                # 量化DCT系数
                quantized_coefficients = self.quantize(dct_coefficients, QP, block_size)

                # TODO: 保存QP值以备将来使用

                frames.append(quantized_coefficients)

        return frames
    

    # 实现I帧的内部预测（水平或垂直）
    def encode_i_frame_intra(self, frame_data, QP):
        frame_height = self.config.height
        frame_width = self.config.width
        block_size = self.config.block_size
        frames = []

        for i in range(0, frame_height, block_size):
            for j in range(0, frame_width, block_size):
                block = Block(i, j, block_size, frame_data)

                # 选择水平或垂直模式
                horizontal_mode = self.horizontal_intra_prediction(block)
                vertical_mode = self.vertical_intra_prediction(block)

                # 选择MAE成本最低的模式
                if horizontal_mode['MAE'] < vertical_mode['MAE']:
                    selected_mode = horizontal_mode
                else:
                    selected_mode = vertical_mode

                # 使用选定的模式进行内部预测
                intra_predicted_block = self.perform_intra_prediction(selected_mode, block)

                # 对内部预测块进行DCT、量化等操作
                dct_coefficients = self.dct_transform(intra_predicted_block.block_data)
                quantized_coefficients = self.quantize(dct_coefficients, QP, block_size)

                # 将QP和quantized_coefficients一起存储到一个数据结构中，然后添加到frames列表中
                frame_info = {'QP': QP, 'coefficients': quantized_coefficients}
                frames.append(frame_info)

        return frames

    # 实现水平内部预测
    def horizontal_intra_prediction(self, block):
        # 水平预测逻辑：使用左边的重建样本进行预测
        left_border = np.full((block.block_size,), 128, dtype=np.uint8)  # 使用128填充缺失的样本
        predicted_block = np.tile(left_border, (block.block_size, 1))
        mae_cost = np.abs(block.block_data - predicted_block).mean()
        return {'mode': 0, 'MAE': mae_cost}


    # 实现垂直内部预测
    def vertical_intra_prediction(self, block):
        # 垂直预测逻辑：使用顶部的重建样本进行预测
        top_border = np.full((1, block.block_size), 128, dtype=np.uint8)  # 使用128填充缺失的样本
        predicted_block = np.tile(top_border, (block.block_size, 1))
        mae_cost = np.abs(block.block_data - predicted_block).mean()
        return {'mode': 1, 'MAE': mae_cost}


    def encode(self, input_video_path, ws='Assignment/encoder_ws'):
        # Read from video, default 420p
        frames = self.yuv_operator.read_yuv(input_video_path)
        block_size = self.config.block_size
        width = self.config.width
        height = self.config.height
        residual_frames = []
        motion_vectors = []
        # Construct first reference frame
        reference_frame = self._construct_virtual_reference_frame((self.config.height, self.config.width), 128)
        reference_frame = pad_frame(reference_frame, block_size)
        Y_previous = self._construct_virtual_reference_frame((self.config.height, self.config.width), 128)
        Y_previous = pad_frame(Y_previous, block_size)
        is_i_frame = True # 标志是否是I帧
        # Start encoding
        for k, frame in enumerate(frames):
            # Focus on Y only
            if self.config.Y_only_mode == True:
                Y_frame, _, _ = self.yuv_operator.get_YUV_from_frame(frame)
                Y_frame = pad_frame(Y_frame, block_size)
                reconstructed_frame = np.zeros(Y_frame.shape, dtype=np.uint8)
                residual_frame = np.zeros(Y_frame.shape, dtype=np.uint8)
                # split to blocks
                for i in range(0, self.config.height, block_size):
                    for j in range(0, self.config.width, block_size):
                        block = Block(i, j, block_size, Y_frame)
                        motion_vector, best_matching_block, residual_block = self._simple_predict_Y_(block, reference_frame, self.config.search_range)
                        residual_block = self._approximate_residual_block(residual_block)
                        reconstructed_block = self._reconstruct_block_(best_matching_block, residual_block)

                        if is_i_frame:
                            # 6. 编码I帧
                            QP = 0  # TODO: 读取QP值，现在设置个default值，我觉得可以写到GlobalConfig里
                            i_frame_data = self.encode_i_frame_intra(Y_frame, QP)  # 传递QP值给encode_i_frame方法
                            is_i_frame = False
                            # TODO: 将i_frame_data保存到文件或传输给解码器
                        else:
                            # 编码P帧
                            QP = 0  # TODO: 从配置文件或其他途径获取QP值
                            p_frame_data = self.encode_p_frame(Y_frame, reference_frame, motion_vectors, QP)

                        # dump blocks
                        reconstructed_frame[i:i+block_size, j:j+block_size] = reconstructed_block.block_data
                        residual_frame[i:i+block_size, j:j+block_size] = residual_block.block_data
                        motion_vectors.append(motion_vector)
                        # self.yuv_operator.visualize_YUV_in_text(best_matching_block.block_data)

                # post process
                reference_frame = reconstructed_frame.copy()
                # self.yuv_operator.convert_Y_to_png(Y_frame, f'{ws}/png_sequence/Y_frame{k}.png')
                # self.yuv_operator.convert_Y_to_png(residual_frame, f'Assignment/temp_output/png_sequence/Y_frame_residual{k}.png')
                self.yuv_operator.convert_Y_to_png(reconstructed_frame, f'{ws}/png_sequence/Y_frame_reconstructed{k}.png')
                # self.yuv_operator.convert_Y_to_png(Y_frame-Y_previous, f'Assignment/temp_output/png_sequence/Y_frame_diff{k}.png')
                print(f"Frame{k}:  PSNR: {PSNR(Y_frame[:height, :width], reconstructed_frame[:height, :width])}, \tSSIM: {SSIM(Y_frame[:height, :width], reconstructed_frame[:height, :width])}")
                Y_previous = Y_frame
                residual_frames.append(residual_frame)

        # save to files
        with open(f'{ws}/residual_frames.13', 'wb') as f:
            for residual_frame in residual_frames:
                f.write(residual_frame.tobytes())
        with open(f'{ws}/mv.13', 'wb') as f:
            for mv in motion_vectors:
                f.write(mv.tobytes())

        return input_video_path
    

if __name__ == '__main__':
    encoder = Encoder()
    encoder.encode('Videos/foreman_420p.yuv')
    
    