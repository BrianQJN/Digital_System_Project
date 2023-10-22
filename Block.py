import numpy as np
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

from YUV_formater import YUV_Operator
from GlobalConfig import Config

config = Config()

# init parameter for a block
# coordinates of top-left pixel: x, y
# block size
# raw data of Y(or U V)
class Block:
    def __init__(self, x, y, block_size, Y_frame: np.ndarray, Y_data: np.ndarray = None) -> None:
        self.x = x
        self.y = y
        self.block_size = block_size
        if Y_data is None: 
            self.block_data = np.array(Y_frame[x:x+block_size, y:y+block_size], dtype=np.uint8)
        else:
            self.block_data = Y_data[:block_size*block_size].reshape((block_size, block_size))
    

    def _calculate_approximate_average_(self):
        average = np.mean(self.block_data)
        return np.uint8(round(average))
    

    def average_block(self):
        return np.full((self.block_size, self.block_size), self._calculate_approximate_average_())


    def average_differential_block(self, magnify_factor = 2):
        return np.abs(self.block_data - self._calculate_approximate_average_())*(2**magnify_factor)


def pad_frame(frame, block_size):
    height, width = frame.shape
    padding_needed = False

    # calculate the width and height to see if padding is needed
    if width % block_size != 0:
        padding_width = block_size - (width % block_size)
        padding_needed = True
    else:
        padding_width = 0

    if height % block_size != 0:
        padding_height = block_size - (height % block_size)
        padding_needed = True
    else:
        padding_height = 0

    # if padding is needed, pad the frame with gray(128)
    if padding_needed:
        frame = np.pad(frame, ((0, padding_height), (0, padding_width)), 'constant', constant_values=128)
    return frame

if __name__ == '__main__':
    operator = YUV_Operator(config=config)
    frames = operator.read_yuv('Videos/foreman_420p.yuv')
    block_size = config.block_size
    width = config.width
    height = config.height

    # Only focus on Y
    for k, frame in enumerate(frames):
        Y_frame, _, _ = operator.get_YUV_from_frame(frame)
        Y_frame = pad_frame(Y_frame, block_size=config.block_size)
        operator.convert_Y_to_png(Y_frame, f'Assignment/temp_output/png_sequence/Y_frame{k}.png')
        Y_frame_average = np.zeros(Y_frame.shape, dtype=np.uint8)
        Y_frame_average = pad_frame(Y_frame_average, block_size=config.block_size)
        Y_frame_average_diff = np.zeros(Y_frame.shape, dtype=np.uint8)
        Y_frame_average_diff = pad_frame(Y_frame_average_diff, block_size=config.block_size)
        # split to blocks
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                
                block = Block(i, j, block_size, Y_frame)
                average_block = block.average_block()
                average_block_diff = block.average_differential_block()
                Y_frame_average[i:i+block_size, j:j+block_size] = average_block
                Y_frame_average_diff[i:i+block_size, j:j+block_size] = average_block_diff
        operator.convert_Y_to_png(Y_frame_average, f'Assignment/temp_output/png_sequence/Y_frame_average{k}.png')
        operator.convert_Y_to_png(Y_frame_average_diff, f'Assignment/temp_output/png_sequence/Y_frame_average_diff{k}.png')
        print(f'Frame{k}: PSNR:{PSNR(Y_frame, Y_frame_average)} SSIM: {SSIM(Y_frame, Y_frame_average)}')