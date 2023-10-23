import numpy as np

from GlobalConfig import Config
from YUV_formater import YUV_Operator
from Block import Block, pad_frame, PSNR, SSIM
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

    
    def encode(self, input_video_path, ws = 'Assignment/encoder_ws'):
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
                        # print(best_matching_block.block_data)
                        # print('--------')
                        # print(residual_block.block_data[0][0], end=' ')

                        # print(reconstructed_block.block_data)
                        # print('***********************************')
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
    
    