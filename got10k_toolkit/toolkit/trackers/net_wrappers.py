from lib.models.model import SiamBC
from lib.utils.utils import load_pretrain
from tracking.set_config_by_type import set_config_by_type


class NetWrapper:
    """Used for wrapping networks in pytracking.
    Network modules and functions can be accessed directly as if they were members of this class."""
    _rec_iter = 0

    def __init__(self, net_path, use_gpu=True, dataset='OTB100', align=True, type=None):
        self.net_path = net_path
        self.use_gpu = use_gpu
        self.dataset = dataset
        self.align = align
        self.type = type
        self.net = None
        self.load_network()

    def __getattr__(self, name):
        if self._rec_iter > 0:
            self._rec_iter = 0
            return None
        self._rec_iter += 1
        try:
            ret_val = getattr(self.net, name)
        except Exception as e:
            self._rec_iter = 0
            raise e
        self._rec_iter = 0
        return ret_val

    def load_network(self):
        print("load network")
        # 设置参数
        if self.online:
            align = False
        else:
            align = True if 'VOT' in self.dataset and self.align == 'True' else False

        # 根据类型设置配置项
        set_config_by_type(self.type)

        self.net = SiamBC(align=align)
        self.net = load_pretrain(self.net, self.net_path)
        if self.use_gpu:
            self.cuda()
        self.eval()


class NetWithBackbone(NetWrapper):
    """Wraps a network with a common backbone.
    Assumes the network have a 'extract_backbone_features(image)' function."""

    def __init__(self, net_path, use_gpu=True, **kwargs):
        super().__init__(net_path, use_gpu, **kwargs)

    def initialize(self):
        super().initialize()

    def template(self, z):
        self.net.template(z)

    def track(self, image):
        return self.net.track(image)
