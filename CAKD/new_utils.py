# =============================================================================
# new_utils.py — HỘP ĐỒ NGHỀ (toolbox) cho hệ thống CAKD
# -----------------------------------------------------------------------------
# File này KHÔNG chạy training, nó chỉ định nghĩa các class/hàm phụ trợ để các
# file train (dist_train_cakd.py, dist_train_student.py, dist_train_logits.py)
# import và gọi ra dùng.
#
# =============================================================================
# MỤC LỤC — VAI TRÒ CỦA TỪNG THÀNH PHẦN
# =============================================================================
#
# ----- NHÓM 1: GAN ĐỐI KHÁNG (ép student bắt chước teacher qua "giám khảo") -----
#   class GANLoss                -> Tính loss cho GAN. Che giấu việc tạo tensor nhãn
#                                   (toàn 1=thật / toàn 0=giả); hỗ trợ 3 loại:
#                                   vanilla, lsgan, wgangp. Chỉ cần gọi criterion(pred, True/False).
#   class NLayerDiscriminator    -> "Giám khảo" kiểu PatchGAN (nhiều lớp Conv). Nhận
#                                   attention map, chấm điểm thật/giả theo từng vùng.
#                                   Đây là discriminator CAKD thực tế dùng.
#   class PixelDiscriminator     -> Giám khảo PatchGAN cỡ 1x1 (chấm từng pixel). Bản
#                                   nhẹ hơn, không nhìn ngữ cảnh vùng. (Định nghĩa sẵn, tùy chọn.)
#
# ----- NHÓM 2: LẤY MẪU DỮ LIỆU & TĂNG CƯỜNG (chuẩn bị ảnh đầu vào) -----
#   class RASampler              -> Sampler cho train đa GPU + "repeated augmentation":
#                                   lặp mỗi ảnh nhiều lần (augment khác nhau), chia cho các GPU.
#   class ClassificationPresetTrain -> Gói chuỗi biến đổi ảnh khi TRAIN (crop/lật/augment mạnh
#                                   + chuẩn hóa). Gọi như hàm: preset(img).
#   class ClassificationPresetEval  -> Gói biến đổi ảnh khi TEST (resize + crop giữa + chuẩn hóa,
#                                   KHÔNG augment ngẫu nhiên -> kết quả tất định).
#
# ----- NHÓM 3: GHI LOG & ĐO LƯỜNG (theo dõi quá trình train) -----
#   class SmoothedValue          -> Theo dõi 1 chuỗi số (vd loss) và cho ra giá trị "làm mượt"
#                                   (trung vị/trung bình cửa sổ, trung bình toàn cục).
#   class MetricLogger           -> Quản lý NHIỀU SmoothedValue cùng lúc + in tiến độ train,
#                                   ước lượng thời gian còn lại (ETA). Bọc quanh data_loader.
#   class ExponentialMovingAverage -> Giữ bản sao trọng số model làm mượt theo EMA (thường cho
#                                   độ chính xác cao & ổn định hơn khi đánh giá).
#
# ----- HÀM DÙNG CHUNG (định nghĩa NGOÀI class, gọi trực tiếp new_utils.ten_ham) -----
#   accuracy(output, target, topk)      -> Tính độ chính xác top-k (%). Vd top1, top5.
#   mkdir(path)                         -> Tạo thư mục, bỏ qua nếu đã tồn tại.
#   setup_for_distributed(is_master)    -> Khi đa GPU, chỉ tiến trình chủ được print (tránh log lặp).
#   is_dist_avail_and_initialized()     -> Kiểm tra môi trường phân tán đã sẵn sàng chưa.
#   get_world_size()                    -> Số tiến trình (GPU) đang tham gia (không phân tán -> 1).
#   get_rank()                          -> Số thứ tự tiến trình hiện tại (không phân tán -> 0).
#   is_main_process()                   -> Có phải tiến trình chủ (rank 0) không.
#   save_on_master(*a, **k)             -> Chỉ tiến trình chủ mới lưu file (tránh N GPU ghi đè).
#   init_distributed_mode(args)         -> Khởi tạo môi trường train đa GPU (đọc rank/world_size,
#                                          lập nhóm tiến trình NCCL).
#   average_checkpoints(inputs)         -> Trung bình trọng số nhiều checkpoint (ensemble nhẹ).
#   store_model_weights(model, ...)     -> Xuất file trọng số "sạch" để phát hành (kèm hash SHA256).
#   reduce_across_processes(val)        -> Cộng dồn 1 giá trị qua tất cả GPU (all_reduce) -> metric toàn cục.
#   set_weight_decay(model, ...)        -> Chia tham số model thành nhóm để áp weight_decay khác nhau
#                                          (vd lớp norm/bias không bị phạt). Trả param_groups cho optimizer.
# =============================================================================

# ----- Thư viện chuẩn của Python -----
import math          # hàm toán học: ceil, floor... (dùng ở RASampler)
import copy          # copy.deepcopy: sao chép sâu 1 object (dùng khi lưu model)
import datetime      # định dạng thời gian (in ETA — thời gian còn lại)
import errno         # mã lỗi hệ điều hành (dùng ở hàm mkdir để bắt lỗi "đã tồn tại")
import hashlib       # băm SHA256 (đặt tên file trọng số theo hash)
import os            # thao tác file/thư mục & đọc biến môi trường
import time          # đo thời gian chạy mỗi vòng lặp
from collections import defaultdict, deque, OrderedDict
#   defaultdict: dict tự tạo giá trị mặc định khi key chưa có
#   deque:       hàng đợi 2 đầu, giới hạn độ dài -> lưu cửa sổ trượt các giá trị
#   OrderedDict: dict giữ đúng thứ tự thêm vào (dùng cho state_dict của model)
from typing import List, Optional, Tuple   # chú thích kiểu (type hints), không ảnh hưởng chạy
import functools     # functools.partial: "đóng gói" 1 hàm kèm sẵn tham số

# ----- Thư viện PyTorch (deep learning) -----
import torch                       # thư viện tensor & autograd cốt lõi
import torch.nn as nn              # các lớp mạng nơ-ron (Conv, Loss, Module...)
from torch.nn import init          # các hàm khởi tạo trọng số
from torch.optim import lr_scheduler   # điều chỉnh learning rate theo epoch
import torch.distributed as dist   # train phân tán trên nhiều GPU/máy
from torchvision.transforms import autoaugment, transforms   # phép biến đổi ảnh
from torchvision.transforms.functional import InterpolationMode   # kiểu nội suy khi resize


# =============================================================================
# NHÓM 1 — GAN ĐỐI KHÁNG
# =============================================================================

class GANLoss(nn.Module):
    """Định nghĩa các loại hàm mất mát (objective) cho GAN.
    Class này che giấu việc phải tự tay tạo tensor nhãn (label) có cùng kích
    thước với đầu vào — bạn chỉ cần truyền True (real) hoặc False (fake).
    """

    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):
        """ Khởi tạo GANLoss.
        Tham số:
            gan_mode (str)         -- loại GAN: 'vanilla', 'lsgan', hoặc 'wgangp'
            target_real_label      -- nhãn cho ảnh THẬT (mặc định 1.0)
            target_fake_label      -- nhãn cho ảnh GIẢ  (mặc định 0.0)
        Lưu ý: KHÔNG đặt sigmoid ở lớp cuối của Discriminator.
        LSGAN không cần sigmoid; vanilla GAN dùng BCEWithLogitsLoss (đã gồm sigmoid).
        """
        super(GANLoss, self).__init__()   # gọi hàm khởi tạo của lớp cha nn.Module (bắt buộc)
        # register_buffer: đăng ký tensor là "buffer" — KHÔNG tính gradient (nhãn là
        # hằng số), nhưng TỰ ĐỘNG chuyển theo device khi .to(device)/.cuda().
        # Nhờ vậy nhãn luôn cùng GPU/CPU với prediction -> tránh lỗi khác device.
        self.register_buffer('real_label', torch.tensor(target_real_label))  # giá trị 1.0
        self.register_buffer('fake_label', torch.tensor(target_fake_label))  # giá trị 0.0
        self.gan_mode = gan_mode          # nhớ lại loại GAN để dùng trong __call__
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()          # LSGAN: phạt theo bình phương sai lệch
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()# GAN gốc: sigmoid + binary cross-entropy gộp 1 (ổn định số học)
        elif gan_mode in ['wgangp']:
            self.loss = None                  # WGAN-GP không dùng loss có sẵn, tính tay ở dưới
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)  # loại lạ -> báo lỗi ngay

    def get_target_tensor(self, prediction, target_is_real):
        """Tạo tensor nhãn có CÙNG kích thước với đầu vào (đầu ra discriminator).
        Tham số:
            prediction (tensor)    -- đầu ra của discriminator
            target_is_real (bool)  -- nhãn là cho ảnh thật hay giả
        Trả về:
            Tensor nhãn (toàn 1 hoặc toàn 0) đúng shape của prediction.
        """

        if target_is_real:
            target_tensor = self.real_label   # chọn scalar 1.0
        else:
            target_tensor = self.fake_label   # chọn scalar 0.0
        # expand_as: "phình" scalar ra đúng shape prediction MÀ KHÔNG cấp phát bộ nhớ
        # mới (chỉ tạo view broadcast) -> tiết kiệm RAM. Cần vì discriminator kiểu
        # PatchGAN trả về cả 1 map dự đoán, không phải 1 số.
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Tính loss từ đầu ra Discriminator và nhãn ground-truth.
        Nhờ định nghĩa __call__, có thể gọi object như hàm: loss = criterion(pred, True)
        Tham số:
            prediction (tensor)    -- đầu ra discriminator
            target_is_real (bool)  -- nhãn thật/giả
        Trả về:
            Giá trị loss (tensor scalar).
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)  # tạo tensor nhãn
            loss = self.loss(prediction, target_tensor)   # so prediction với nhãn qua MSE/BCE
        elif self.gan_mode == 'wgangp':
            # WGAN-GP: loss = trung bình điểm số (xấp xỉ khoảng cách Wasserstein)
            if target_is_real:
                loss = -prediction.mean()  # muốn điểm real CAO -> tối thiểu số âm
            else:
                loss = prediction.mean()   # muốn điểm fake THẤP
        return loss


class NLayerDiscriminator(nn.Module):
    """Định nghĩa discriminator kiểu PatchGAN.
    PatchGAN không chấm điểm cả ảnh 1 lần, mà chấm từng "vùng nhỏ" (patch) rồi
    tổng hợp -> nhạy với chi tiết cục bộ. Trong CAKD nó nhận attention map của
    student (giả) / teacher (thật) và học phân biệt hai bên.
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Xây dựng discriminator PatchGAN.
        Tham số:
            input_nc (int)  -- số kênh ảnh đầu vào (CAKD dùng 1 vì attention map 1 kênh)
            ndf (int)       -- số bộ lọc (filter) ở lớp conv đầu tiên
            n_layers (int)  -- số lớp conv trong discriminator
            norm_layer      -- lớp chuẩn hóa (BatchNorm/InstanceNorm)
        """
        super(NLayerDiscriminator, self).__init__()
        # Nếu dùng InstanceNorm thì conv nên có bias; nếu BatchNorm thì không cần
        # (vì BatchNorm đã có tham số affine đảm nhiệm vai trò dịch/scale).
        if type(norm_layer) == functools.partial:   # norm_layer bị gói bởi partial -> lấy hàm gốc ra so
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4       # kernel width: kích thước cửa sổ conv = 4x4
        padw = 1     # padding: đệm 1 pixel quanh viền
        # Lớp đầu: Conv hạ kích thước (stride=2) + LeakyReLU (không dùng norm ở lớp đầu).
        # LeakyReLU(0.2): với giá trị âm vẫn cho qua 0.2*x (tránh "chết nơ-ron").
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1        # hệ số nhân số filter hiện tại
        nf_mult_prev = 1   # hệ số nhân số filter lớp trước
        for n in range(1, n_layers):   # tăng dần số filter qua từng lớp
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)   # nhân đôi mỗi lớp nhưng chặn trần ở 8 lần (2->4->8)
            sequence += [
                # Conv hạ kích thước (stride=2), số kênh vào = ndf*prev, ra = ndf*mult
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),   # chuẩn hóa đầu ra
                nn.LeakyReLU(0.2, True)       # kích hoạt phi tuyến
            ]

        # Thêm 1 lớp conv nữa nhưng stride=1 (giữ nguyên kích thước, tăng độ sâu đặc trưng)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Lớp cuối: conv về 1 kênh -> ra "bản đồ dự đoán" (mỗi ô = điểm thật/giả 1 patch)
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 kênh
        sequence += [nn.AdaptiveAvgPool2d(1)]   # gộp trung bình cả map -> 1 con số duy nhất cho mỗi ảnh
        self.model = nn.Sequential(*sequence)   # nối tất cả các lớp thành 1 mạng tuần tự

    def forward(self, input):
        """Lượt truyền xuôi chuẩn: đưa input qua mạng, trả về điểm dự đoán."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Discriminator PatchGAN cỡ 1x1 (còn gọi PixelGAN).
    Toàn bộ dùng conv kernel 1x1 -> chấm điểm thật/giả cho TỪNG pixel độc lập,
    không nhìn ngữ cảnh vùng xung quanh. Nhẹ hơn NLayerDiscriminator.
    """

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Xây dựng PixelDiscriminator.
        Tham số:
            input_nc (int)  -- số kênh ảnh đầu vào
            ndf (int)       -- số filter ở lớp conv cuối
            norm_layer      -- lớp chuẩn hóa
        """
        super(PixelDiscriminator, self).__init__()
        # Cùng logic chọn bias như trên: InstanceNorm -> cần bias.
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Chuỗi 3 lớp conv 1x1: input_nc -> ndf -> ndf*2 -> 1 (điểm dự đoán mỗi pixel)
        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)   # nối thành mạng tuần tự

    def forward(self, input):
        """Lượt truyền xuôi chuẩn."""
        return self.net(input)


# =============================================================================
# NHÓM 2 — LẤY MẪU DỮ LIỆU & TĂNG CƯỜNG (AUGMENTATION)
# =============================================================================

class RASampler(torch.utils.data.Sampler):
    """Sampler chia dữ liệu cho train phân tán, KÈM "repeated augmentation" (RA):
    mỗi ảnh được lặp lại nhiều lần (mỗi lần augment khác nhau) và các bản khác nhau
    được phân cho các GPU khác nhau. Kỹ thuật này giúp train hội tụ tốt hơn.
    Dựa nhiều trên torch.utils.data.DistributedSampler; mượn từ repo DeiT:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3):
        # num_replicas = số tiến trình (số GPU). Nếu không truyền thì tự lấy từ hệ phân tán.
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            num_replicas = dist.get_world_size()
        # rank = số thứ tự GPU hiện tại (0,1,2...). Không truyền thì tự lấy.
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            rank = dist.get_rank()
        self.dataset = dataset               # bộ dữ liệu gốc
        self.num_replicas = num_replicas     # tổng số GPU
        self.rank = rank                     # GPU này là số mấy
        self.epoch = 0                       # epoch hiện tại (dùng để trộn ngẫu nhiên có kiểm soát)
        # num_samples: số mẫu mỗi GPU xử lý = ceil(N * repetitions / số GPU)
        self.num_samples = int(math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas   # tổng số mẫu (chia đều được)
        # num_selected_samples: cắt bớt về bội số của 256 để batch chia chẵn
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle               # có trộn ngẫu nhiên hay không
        self.seed = seed                     # hạt giống ngẫu nhiên (để tái lập kết quả)
        self.repetitions = repetitions       # số lần lặp lại mỗi ảnh

    def __iter__(self):
        # Hàm này trả về iterator các CHỈ SỐ (index) ảnh mà GPU này sẽ nạp.
        if self.shuffle:
            # Trộn ngẫu nhiên nhưng "tất định" theo epoch: cùng seed+epoch -> cùng thứ tự
            # (đảm bảo tất cả GPU trộn giống nhau, tránh trùng/thiếu dữ liệu).
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))   # không trộn -> 0,1,2,...

        # Lặp mỗi index 'repetitions' lần: [a,b] -> [a,a,a, b,b,b] (repeated augmentation)
        indices = [ele for ele in indices for i in range(self.repetitions)]
        # Đệm thêm vài phần tử đầu danh sách cho đủ total_size (chia chẵn cho các GPU)
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size   # kiểm tra đủ số lượng

        # Lấy phần dành riêng cho GPU này: bắt đầu từ rank, nhảy bước num_replicas
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # Chỉ trả về num_selected_samples đầu (cắt cho chẵn batch)
        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        # Số mẫu mà GPU này duyệt trong 1 epoch
        return self.num_selected_samples

    def set_epoch(self, epoch):
        # Vòng train gọi hàm này đầu mỗi epoch để đổi cách trộn (xem __iter__)
        self.epoch = epoch


class ClassificationPresetTrain:
    """Gói sẵn chuỗi biến đổi ảnh khi TRAIN (augment mạnh để model tổng quát hơn)."""
    def __init__(
        self,
        *,                       # dấu * ép mọi tham số sau phải gọi bằng tên (keyword-only)
        crop_size,               # kích thước ảnh sau khi cắt
        mean=(0.485, 0.456, 0.406),   # trung bình chuẩn hóa (chuẩn ImageNet, mỗi kênh RGB)
        std=(0.229, 0.224, 0.225),    # độ lệch chuẩn để chuẩn hóa
        interpolation=InterpolationMode.BILINEAR,   # kiểu nội suy khi resize
        hflip_prob=0.5,          # xác suất lật ngang ảnh
        auto_augment_policy=None,# chính sách auto-augment (nếu có)
        ra_magnitude=9,          # cường độ RandAugment
        augmix_severity=3,       # độ mạnh AugMix
        random_erase_prob=0.0,   # xác suất xóa ngẫu nhiên 1 vùng ảnh
    ):
        # Bắt đầu chuỗi biến đổi: cắt ngẫu nhiên rồi resize về crop_size
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))   # lật ngang ngẫu nhiên
        if auto_augment_policy is not None:
            # Chọn 1 trong các chiến lược auto-augment
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                trans.append(autoaugment.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),                  # ảnh PIL -> tensor (số nguyên 0-255)
                transforms.ConvertImageDtype(torch.float), # ép về float, đưa về [0,1]
                transforms.Normalize(mean=mean, std=std),  # chuẩn hóa: (x - mean)/std
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))  # xóa 1 mảng ngẫu nhiên (regularization)

        self.transforms = transforms.Compose(trans)   # gộp toàn bộ thành 1 pipeline

    def __call__(self, img):
        # Cho phép gọi object như hàm: img_da_bien_doi = preset(img)
        return self.transforms(img)


class ClassificationPresetEval:
    """Gói biến đổi ảnh khi ĐÁNH GIÁ/TEST (KHÔNG augment ngẫu nhiên, chỉ resize+crop giữa)."""
    def __init__(
        self,
        *,
        crop_size,               # kích thước cắt giữa
        resize_size=256,         # resize về trước khi cắt
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),  # resize cạnh nhỏ về 256
                transforms.CenterCrop(crop_size),          # cắt chính giữa (tất định, không ngẫu nhiên)
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)


# =============================================================================
# NHÓM 3 — GHI LOG & ĐO LƯỜNG
# =============================================================================

class SmoothedValue:
    """Theo dõi 1 chuỗi giá trị (vd loss theo từng batch) và cho ra giá trị "làm mượt"
    trên 1 cửa sổ trượt, hoặc trung bình toàn cục. Dùng để log số liệu đỡ nhiễu.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"   # định dạng in mặc định: trung vị (trung bình toàn cục)
        self.deque = deque(maxlen=window_size)   # hàng đợi giữ tối đa window_size giá trị gần nhất
        self.total = 0.0    # tổng dồn tất cả giá trị (cho trung bình toàn cục)
        self.count = 0      # số lần cập nhật
        self.fmt = fmt      # chuỗi định dạng khi in

    def update(self, value, n=1):
        # Thêm 1 giá trị mới; n = số mẫu (vd batch_size) để tính trung bình có trọng số
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Đồng bộ count & total giữa các GPU (KHÔNG đồng bộ deque)."""
        t = reduce_across_processes([self.count, self.total])  # cộng dồn qua tất cả GPU
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property                    # biến 1 method thành thuộc tính -> gọi obj.median (không có ())
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()   # trung vị cửa sổ (ít bị outlier ảnh hưởng)

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()     # trung bình cửa sổ trượt

    @property
    def global_avg(self):
        return self.total / self.count   # trung bình toàn bộ (từ đầu tới giờ)

    @property
    def max(self):
        return max(self.deque)     # giá trị lớn nhất trong cửa sổ

    @property
    def value(self):
        return self.deque[-1]      # giá trị mới nhất

    def __str__(self):
        # Khi print(obj) -> in ra theo định dạng fmt với các thuộc tính trên
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    """Quản lý NHIỀU SmoothedValue cùng lúc (loss, accuracy, lr...) và in tiến độ train."""
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)   # dict: tên metric -> SmoothedValue (tự tạo khi truy cập key mới)
        self.delimiter = delimiter                 # ký tự ngăn cách khi in các metric

    def update(self, **kwargs):
        # Cập nhật nhiều metric 1 lần: logger.update(loss=..., acc=...)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()             # nếu là tensor -> lấy số Python
            assert isinstance(v, (float, int))
            self.meters[k].update(v)     # đẩy giá trị vào meter tương ứng

    def __getattr__(self, attr):
        # Cho phép truy cập metric như thuộc tính: logger.loss thay vì logger.meters['loss']
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        # Ghép chuỗi tất cả metric để in
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        # Đồng bộ tất cả metric giữa các GPU
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        # Thêm 1 metric có định dạng riêng
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """Generator bọc quanh data_loader: vừa duyệt batch, vừa in log định kỳ +
        ước lượng thời gian còn lại (ETA). Dùng: for x in logger.log_every(loader, 10, hdr):"""
        i = 0
        if not header:
            header = ""
        start_time = time.time()   # mốc bắt đầu toàn bộ
        end = time.time()          # mốc kết thúc batch trước
        iter_time = SmoothedValue(fmt="{avg:.4f}")   # đo thời gian mỗi vòng lặp
        data_time = SmoothedValue(fmt="{avg:.4f}")   # đo thời gian nạp dữ liệu
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"   # canh lề số đếm cho đẹp
        if torch.cuda.is_available():
            # Mẫu chuỗi log khi có GPU (thêm thông tin bộ nhớ)
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",   # [i/tổng]
                    "eta: {eta}",                    # thời gian còn lại
                    "{meters}",                      # các metric
                    "time: {time}",                  # thời gian/vòng
                    "data: {data}",                  # thời gian nạp data
                    "max mem: {memory:.0f}",         # RAM GPU đỉnh
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0   # số byte trong 1 MB (đổi đơn vị bộ nhớ)
        for obj in iterable:
            data_time.update(time.time() - end)   # đo thời gian nạp batch này
            yield obj                              # trả batch ra cho vòng train dùng, rồi chờ quay lại
            iter_time.update(time.time() - end)   # đo tổng thời gian xử lý batch này
            if i % print_freq == 0:               # cứ print_freq batch thì in 1 lần
                eta_seconds = iter_time.global_avg * (len(iterable) - i)   # ước lượng thời gian còn lại
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,   # RAM GPU đỉnh (MB)
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()   # cập nhật mốc cho batch sau
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")   # in tổng thời gian cả epoch


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Giữ 1 bản sao trọng số model được làm mượt theo trung bình mũ (EMA).
    Công thức: ema = decay * ema + (1 - decay) * trọng_số_hiện_tại.
    Model EMA thường cho độ chính xác cao & ổn định hơn model gốc lúc đánh giá.
    """

    def __init__(self, model, decay, device="cpu"):
        # Hàm định nghĩa cách cập nhật trung bình cho từng tham số
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param
        # Gọi lớp cha, use_buffers=True để cả buffer (vd running_mean của BatchNorm) cũng được EMA
        super().__init__(model, device, ema_avg, use_buffers=True)


def accuracy(output, target, topk=(1,)):
    """Tính độ chính xác top-k: dự đoán đúng nếu nhãn thật nằm trong k lớp có điểm cao nhất.
    Trả về danh sách accuracy (%) tương ứng từng k trong topk. Vd topk=(1,5) -> [top1, top5]."""
    with torch.inference_mode():   # tắt autograd (chỉ suy luận, nhanh & tiết kiệm RAM)
        maxk = max(topk)           # k lớn nhất cần xét
        batch_size = target.size(0)
        if target.ndim == 2:       # nếu nhãn dạng one-hot/soft (2 chiều) -> đổi về chỉ số lớp
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)   # lấy maxk lớp điểm cao nhất mỗi mẫu
        pred = pred.t()                              # chuyển vị -> shape (maxk, batch)
        correct = pred.eq(target[None])              # so khớp với nhãn thật -> ma trận True/False

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)  # đếm số đúng trong top-k
            res.append(correct_k * (100.0 / batch_size))               # đổi ra phần trăm
        return res


# =============================================================================
# NHÓM 4 — TIỆN ÍCH HỆ THỐNG, TRAIN ĐA GPU & LƯU MODEL
# =============================================================================

def mkdir(path):
    """Tạo thư mục; nếu đã tồn tại thì bỏ qua (không báo lỗi)."""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:   # chỉ nuốt lỗi "đã tồn tại"; lỗi khác thì ném lại
            raise


def setup_for_distributed(is_master):
    """Khi train nhiều GPU, chỉ cho tiến trình "chủ" (master) được print, các tiến trình
    còn lại im lặng -> tránh log bị nhân bản N lần. Truyền force=True để in cưỡng bức."""
    import builtins as __builtin__

    builtin_print = __builtin__.print   # giữ lại hàm print gốc

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print   # ghi đè print toàn cục bằng phiên bản có kiểm soát


def is_dist_avail_and_initialized():
    """Kiểm tra môi trường phân tán có sẵn VÀ đã khởi tạo chưa."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Số tiến trình (GPU) tham gia; nếu không phân tán thì trả 1."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Số thứ tự tiến trình hiện tại; nếu không phân tán thì trả 0."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Có phải tiến trình chủ (rank 0) không -> dùng để chỉ lưu/log 1 lần."""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """Chỉ tiến trình chủ mới thực sự lưu file (tránh N GPU ghi đè cùng 1 file)."""
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """Khởi tạo môi trường train đa GPU. Đọc thông tin rank/world_size từ biến môi
    trường (do torchrun hoặc SLURM đặt sẵn), rồi thiết lập nhóm tiến trình NCCL."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Trường hợp khởi chạy bằng torchrun / torch.distributed.launch
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        # Trường hợp chạy trên cụm máy dùng SLURM
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass   # đã tự set rank sẵn -> dùng luôn
    else:
        # Không có thông tin phân tán -> chạy 1 GPU/CPU bình thường
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)       # gán GPU cho tiến trình này
    args.dist_backend = "nccl"            # backend giao tiếp GPU của NVIDIA (nhanh nhất)
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    # Khởi tạo nhóm tiến trình để các GPU "bắt tay" và đồng bộ với nhau
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()           # chặn: chờ TẤT CẢ tiến trình tới đây rồi mới đi tiếp
    setup_for_distributed(args.rank == 0) # chỉ rank 0 được print


def average_checkpoints(inputs):
    """Nạp nhiều file checkpoint và trả về 1 model có trọng số là TRUNG BÌNH của chúng.
    (Kỹ thuật ensemble nhẹ, thường tăng độ chính xác.) Nguồn gốc từ fairseq.
    Tham số:
      inputs (List[str]): danh sách đường dẫn các checkpoint cần trung bình.
    Trả về:
      Một dict trạng thái, trong đó key 'model' là OrderedDict {tên tham số -> tensor}.
    """
    params_dict = OrderedDict()   # tích lũy tổng các tham số
    params_keys = None            # danh sách tên tham số (để kiểm tra các checkpoint khớp nhau)
    new_state = None              # giữ cấu trúc state từ checkpoint đầu tiên
    num_models = len(inputs)      # số checkpoint -> chia trung bình
    for fpath in inputs:
        with open(fpath, "rb") as f:
            # Nạp checkpoint, ép mọi tensor về CPU cho an toàn bộ nhớ
            state = torch.load(
                f,
                map_location=(lambda s, _: torch.serialization.default_restore_location(s, "cpu")),
            )
        # Sao cấu hình từ checkpoint đầu tiên (các key ngoài 'model')
        if new_state is None:
            new_state = state
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            # Các checkpoint phải cùng kiến trúc (cùng danh sách tham số)
            raise KeyError(
                f"For checkpoint {f}, expected list of params: {params_keys}, but found: {model_params_keys}"
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()          # đổi float16 -> float32 để cộng chính xác
            if k not in params_dict:
                params_dict[k] = p.clone()   # lần đầu: sao chép (clone tránh sửa nhầm tham số chia sẻ)
            else:
                params_dict[k] += p          # cộng dồn
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)   # tham số số thực -> chia trung bình
        else:
            averaged_params[k] //= num_models     # tham số nguyên -> chia lấy nguyên
    new_state["model"] = averaged_params   # gắn trọng số trung bình vào lại
    return new_state


def store_model_weights(model, checkpoint_path, checkpoint_key="model", strict=True):
    """Chuẩn bị file trọng số "sạch" để phát hành: nạp checkpoint vào model để kiểm
    chứng khớp, loại bỏ phần thừa (vd n_averaged của EMA), rồi lưu lại kèm hash SHA256
    trong tên file (chuẩn đặt tên của torchvision).

    Ví dụ:
        from torchvision import models as M
        model = M.mobilenet_v3_large(weights=None)
        print(store_model_weights(model, './class.pth'))
        ... (các ví dụ khác giữ nguyên trong docstring gốc)

    Tham số:
        model            -- kiến trúc model để nạp trọng số vào kiểm tra.
        checkpoint_path  -- đường dẫn checkpoint cần nạp.
        checkpoint_key   -- key chứa trọng số trong checkpoint (mặc định "model").
        strict           -- có bắt buộc khớp CHÍNH XÁC mọi key không.
    Trả về:
        output_path (str): nơi file trọng số được lưu.
    """
    # Lưu file mới cạnh checkpoint gốc
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.dirname(checkpoint_path)

    # Copy sâu để không làm thay đổi object model gốc bên ngoài
    model = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Nạp trọng số để kiểm chứng chạy được, đồng thời bỏ phần không cần (auxiliary...)
    if checkpoint_key == "model_ema":
        del checkpoint[checkpoint_key]["n_averaged"]   # bỏ biến đếm của EMA (không phải trọng số)
        # Bỏ tiền tố "module." (do DistributedDataParallel thêm vào) nếu có
        nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint[checkpoint_key], "module.")
    model.load_state_dict(checkpoint[checkpoint_key], strict=strict)

    tmp_path = os.path.join(output_dir, str(model.__hash__()))   # file tạm tên theo hash object
    torch.save(model.state_dict(), tmp_path)

    # Tính SHA256 của file để đưa 8 ký tự đầu vào tên (đảm bảo tính toàn vẹn/định danh)
    sha256_hash = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        # Đọc theo khối 4KB để không ngốn RAM với file lớn
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hh = sha256_hash.hexdigest()

    output_path = os.path.join(output_dir, "weights-" + str(hh[:8]) + ".pth")
    os.replace(tmp_path, output_path)   # đổi tên file tạm thành tên cuối cùng

    return output_path


def reduce_across_processes(val):
    """Cộng dồn 1 giá trị qua TẤT CẢ các GPU (all_reduce). Dùng để gộp count/total khi
    tính metric toàn cục. Nếu không phân tán thì chỉ đổi val thành tensor và trả về."""
    if not is_dist_avail_and_initialized():
        # Không phân tán: không cần đồng bộ, nhưng vẫn trả tensor cho nhất quán kiểu dữ liệu.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()        # chờ mọi tiến trình sẵn sàng
    dist.all_reduce(t)    # cộng t của tất cả GPU rồi phát kết quả về mọi GPU
    return t


def set_weight_decay(
    model: nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    """Chia tham số model thành các NHÓM để áp weight_decay (phạt trọng số lớn) KHÁC nhau.
    Thực tế: các lớp chuẩn hóa (BatchNorm...) và bias thường KHÔNG nên bị weight_decay ->
    hàm này tách chúng ra nhóm riêng với hệ số riêng. Trả về danh sách param_groups để
    đưa thẳng vào optimizer."""
    if not norm_classes:
        # Danh sách mặc định các lớp chuẩn hóa cần đối xử riêng
        norm_classes = [
            nn.modules.batchnorm._BatchNorm,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.modules.instancenorm._InstanceNorm,
            nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)   # đổi sang tuple để dùng với isinstance

    # Hai nhóm cơ bản: "norm" (lớp chuẩn hóa) và "other" (phần còn lại)
    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,        # hệ số cho nhóm thường
        "norm": norm_weight_decay,    # hệ số riêng cho lớp norm (có thể = 0)
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        # Cho phép chỉ định hệ số riêng theo TÊN tham số cụ thể
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        # Duyệt ĐỆ QUY toàn bộ cây module, phân loại từng tham số vào đúng nhóm
        for name, p in module.named_parameters(recurse=False):   # chỉ tham số trực tiếp của module này
            if not p.requires_grad:
                continue                 # bỏ qua tham số bị đóng băng (không học)
            is_custom_key = False
            for key in custom_keys:      # ưu tiên khớp theo tên tùy chỉnh trước
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                # Nếu là lớp chuẩn hóa và có cấu hình norm_weight_decay -> nhóm "norm"
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)   # còn lại -> nhóm thường

        for child_name, child_module in module.named_children():
            # Đệ quy xuống các module con, nối thêm tiền tố tên
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)   # bắt đầu duyệt từ model gốc

    # Đóng gói các nhóm không rỗng thành param_groups cho optimizer
    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups
