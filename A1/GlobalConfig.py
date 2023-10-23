# single instance for global config
class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.init_config()  # Initialize the configuration
        return cls._instance

    # Add new global config here
    def init_config(self) -> None:
        self.width = 352
        self.height = 288
        self.block_size = 16
        self.search_range = 4
        self.residual_n = 4
        self.fps = 30
        # True when doing ex3
        self.Y_only_mode = True
        self.I_Period = 10  # 设置I帧的周期，例如每隔10个帧一个I帧