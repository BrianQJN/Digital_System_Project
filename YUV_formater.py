import numpy as np
from scipy.ndimage import zoom
from PIL import Image

from GlobalConfig import Config

class YUV_Operator():
    def __init__(self, config: Config) -> None:
        self.config = config
    

    # Read frames in raw data 
    def read_yuv(self, filename, pix_fmt = '420p'):
        width = self.config.width
        height = self.config.height
        if pix_fmt == '420p':
            frame_size = width*height+2*(width//2)*(height//2)
        elif pix_fmt == '444p':
            frame_size = width*height*3
        frames = []
        with open(filename, 'rb') as f:
            while True:
                try:
                    frame_data = f.read(frame_size)
                    if not frame_data:
                        break
                    frames.append(frame_data)
                except:
                    raise IOError
        return frames
    

    # default 420p, get YUV from a single frame
    def get_YUV_from_frame(self, frame_data, pix_fmt = '420p'):
        width = self.config.width
        height = self.config.height
        Y_count = width*height
        if pix_fmt == '420p':
            scale = 2
        elif pix_fmt == '444p':
            scale = 1
        UV_count = (height//scale)*(width//scale)
        Y_frame = np.frombuffer(frame_data, dtype = np.uint8, count = Y_count, offset=0).reshape((height, width))
        U_frame = np.frombuffer(frame_data, dtype = np.uint8, count = UV_count, offset=Y_count).reshape((height//scale, width//scale))
        V_frame = np.frombuffer(frame_data, dtype = np.uint8, count = UV_count, offset=Y_count+UV_count).reshape((height//scale, width//scale))
        return Y_frame, U_frame, V_frame


    def upscale_to_444(self, Y:np.ndarray, U:np.ndarray, V:np.ndarray):
        """Upscale U, V planes to 4:4:4."""
        U_upscaled = np.repeat(np.repeat(U, 2, axis=0), 2, axis=1)
        V_upscaled = np.repeat(np.repeat(V, 2, axis=0), 2, axis=1)
        frame_data = Y.tobytes() + U_upscaled.tobytes() + V_upscaled.tobytes()
        return frame_data, Y, U_upscaled, V_upscaled


    def _yuv444_to_rgb_(self, y, u, v):
        """Convert YUV values to RGB."""
        matrix = np.array([
            [1.164, 0.000, 1.596],
            [1.164, -0.392, -0.813],
            [1.164, 2.017, 0.000]
        ])
        yuv = np.array([y - 16, u - 128, v - 128])
        rgb = np.dot(matrix, yuv).clip(0, 255).astype(np.uint8)
        return tuple(rgb)


    def convert_frame_data_to_rgb_png(self, frame_data, output_file, pix_fmt='420p'):
        if pix_fmt == '420p':
            Y_frame, U_frame, V_frame = self.get_YUV_from_frame(frame_data)
            _, Y_frame, U_frame, V_frame = self.upscale_to_444(Y_frame, U_frame, V_frame)
        elif pix_fmt == '444p':
            Y_frame, U_frame, V_frame = self.get_YUV_from_frame(frame_data, pix_fmt='444p')
        print(len(Y_frame), len(Y_frame[0]), len(U_frame), len(U_frame[0]), len(V_frame), len(V_frame[0]))
        height = self.config.height
        width = self.config.width
        rgb_frame = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                rgb_frame[i, j] = self._yuv444_to_rgb_(Y_frame[i, j], U_frame[i, j], V_frame[i, j])
            
        # Save the RGB frame as .png
        img = Image.fromarray(rgb_frame, 'RGB')
        img.save(f"{output_file}")
        return output_file
    

    def convert_Y_to_png(self, Y, output_file):
        width = self.config.width
        height = self.config.height
        img = Image.fromarray(Y[:height, :width], 'L')
        img.save(f'{output_file}')
        return output_file


    def visualize_YUV_in_text(self, Y):
        block_size = self.config.block_size
        row_count = 0
        for i, row in enumerate(Y):
            col_count = 0
            for pix in row:
                print(f'{pix},', end='')
                col_count += 1
                if col_count % block_size == 0:
                    print('|', end='')
            print('\n', end='')
            row_count += 1
            if row_count % block_size == 0:
                print('------------------------------------------------------')

            
# EX1
if __name__ == '__main__':
    config = Config()
    operator = YUV_Operator(config)
    frames = operator.read_yuv(filename='Videos/foreman_420p.yuv')
    print(len(frames))
    for i, frame in enumerate(frames):
        # output = operator.convert_frame_data_to_rgb_png(frame, f'Assignment/temp_output/png_sequence/frame_{i}.png', pix_fmt='420p')
        Y_frame, U_frame, V_frame = operator.get_YUV_from_frame(frame_data=frame)
        frame_data, _, _, _ = operator.upscale_to_444(Y_frame, U_frame, V_frame)
        output = operator.convert_frame_data_to_rgb_png(frame_data, f'Assignment/temp_output/png_sequence/frame_{i}.png', pix_fmt='444p')
        output = operator.convert_Y_to_png(Y_frame, f'Assignment/temp_output/png_sequence/Y_only_frame_{i}.png')