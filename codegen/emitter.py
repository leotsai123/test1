from codegen.isa_def import (
    LoadMacroOp,
    LoadEmitOp,
    LoadEmitCIMOp,
    StoreEmitOp,
    XYZ,
    XY,
    MemType
)
import math

CIM_MAC=64
CIM_IC1=128
CIM_IC2=32
CIM_COL = 8

class Emitter:
    def __init__(self,ch_per_group):
        self.ch_per_group = ch_per_group
        self.emitted_op = []
        self.emitted_load = 0
        self.emitted_store = 0
        self.emitted_conv = 0
    
    def emit_load(self, macro_op: LoadMacroOp):
        dst_type = macro_op.payload["dst_type"]
        src_type = macro_op.payload["src_type"]

        if dst_type == MemType.UNI_SRAM and src_type == MemType.BIAS_MEM:
            seg_len         = macro_op.payload["src_len"]
            src_base_addr   = macro_op.payload["src_base_addr"]
            dst_base_addr   = macro_op.payload["dst_base_addr"]
            seg_num = 1
            v_stride = 0
            seg_stride = 0
            self.emitted_op.append(
                LoadEmitOp(
                    src_type=src_type,
                    dst_type=dst_type,
                    src_base_addr=src_base_addr,
                    dst_base_addr=dst_base_addr,
                    seg_num=seg_num,
                    v_stride=v_stride,
                    seg_stride=seg_stride,
                    seg_len=seg_len
                )
            )
            self.emitted_load+=1
        elif dst_type == MemType.UNI_SRAM:
            ifm_shape   = macro_op.payload["ifm_shape"]
            i_tile_shape= macro_op.payload["i_tile_shape"]
            kernel_shape= macro_op.payload["kernel_shape"]
            reduce_size = macro_op.payload["reduce_size"]
            dbyte = macro_op.payload["dbyte"]
            src_base_addr = macro_op.payload["src_base_addr"]
            dst_base_addr = macro_op.payload["dst_base_addr"]
            k_size = macro_op.payload["k_size"]

            reduce_size1, reduce_size2 = reduce_size[0], reduce_size[1]
            k_h, k_w = kernel_shape.h, kernel_shape.w
            i_tile_h, i_tile_w = i_tile_shape.h, i_tile_shape.w
            ifm_h, ifm_w, p_ifm_c = ifm_shape.h, ifm_shape.w, ifm_shape.c
            reduced_ch1_bound = math.ceil(reduce_size1 / (CIM_MAC // k_size))
            if k_h == 1 and k_w == 1:
                reduced_ch2_bound = (CIM_MAC // k_size // self.ch_per_group) if p_ifm_c >= (CIM_MAC // k_size) \
                else (p_ifm_c // self.ch_per_group)
            x_len = i_tile_w * self.ch_per_group * dbyte
            y_len = reduced_ch2_bound * reduced_ch1_bound if k_h == 1 and k_w == 1 else (reduce_size2 // self.ch_per_group)
            z_len = i_tile_h
            xyz_len=XYZ(x_len,y_len,z_len)
            src_x_stride = ifm_w * self.ch_per_group * dbyte
            src_y_stride = p_ifm_c * ifm_w * dbyte
            dst_x_stride=x_len
            dst_y_stride=p_ifm_c * i_tile_w * dbyte
            src_base_addr=src_base_addr
            dst_base_addr=dst_base_addr
            src_stride=XY(src_x_stride,src_y_stride)
            dst_stride=XY(dst_x_stride,dst_y_stride)
            continuous_flag = (p_ifm_c > CIM_IC1 and k_h == 1 and k_w == 1) or (p_ifm_c > CIM_IC2 and k_h != 1 and k_w != 1)

            if continuous_flag:
                for z in range(z_len):
                    seg_num = y_len
                    v_stride = 0
                    seg_stride = src_stride.x
                    seg_len = x_len
                    self.emitted_op.append(
                        LoadEmitOp(
                            src_type=src_type,
                            dst_type=dst_type,
                            src_base_addr=src_base_addr,
                            dst_base_addr=dst_base_addr,
                            seg_num=seg_num,
                            v_stride=v_stride,
                            seg_stride=seg_stride,
                            seg_len=seg_len
                        )
                    )
                    src_base_addr += src_stride.y
                    dst_base_addr += dst_stride.y
                    self.emitted_load+=1
            else:
                seg_num = y_len * z_len
                v_stride = 0
                seg_stride = src_stride.x
                seg_len = x_len
                self.emitted_op.append(
                    LoadEmitOp(
                        src_type=src_type,
                        dst_type=dst_type,
                        src_base_addr=src_base_addr,
                        dst_base_addr=dst_base_addr,
                        seg_num=seg_num,
                        v_stride=v_stride,
                        seg_stride=seg_stride,
                        seg_len=seg_len
                    )
                )
                self.emitted_load+=1
        elif dst_type == MemType.W_SRAM:
            spatial_sizes   = macro_op.payload["spatial_sizes"]
            k_size          = macro_op.payload["k_size"]
            kernel_shape    = macro_op.payload["kernel_shape"]
            ping            = macro_op.payload["ping"]
            dbyte           = macro_op.payload["dbyte"]
            reduce_size     = macro_op.payload["reduce_size"]
            src_base_addr   = macro_op.payload["src_base_addr"]


            spatial_size1, spatial_size2 = spatial_sizes[0], spatial_sizes[1]
            k_h, k_w, p_ifm_c = kernel_shape.h, kernel_shape.w, kernel_shape.c
            reduce_size1, reduce_size2 = reduce_size[0], reduce_size[1]
            # Compute spatial output channel (soc)
            if k_size != 1:
                soc = spatial_size2 // CIM_COL
            else:
                soc = spatial_size1 // CIM_COL
            # Stride for output channels (in bytes)
            oc_stride = CIM_COL * p_ifm_c * k_h * k_w * dbyte
            # Compute spatial input channel (sic)
            if k_size != 1:
                sic = reduce_size2 // (CIM_MAC // k_size)
            else:
                sic = reduce_size1 // (CIM_MAC // k_size)
            # Stride for input channels (in bytes)
            ic_stride = (CIM_MAC // k_size) * k_h * k_w * CIM_COL * dbyte
            # Effective input channels (depending on padding size)
            eic = min(p_ifm_c, CIM_MAC // k_size)
            # Kernel stride (in bytes)
            k_stride = k_h * k_w * CIM_COL * dbyte
            # Compute source length for DMA
            if k_size != 1:
                src_len = k_size * CIM_COL * dbyte
            else:
                # For k_size == 1, kernel shape may be pointwise
                len_eic = min(p_ifm_c, CIM_MAC // k_size)
                if k_h == 1 and k_w == 1:
                    src_len = len_eic * CIM_COL * dbyte
                else:
                    src_len = k_size * CIM_COL * dbyte
            self.emitted_op.append(
                LoadEmitCIMOp(
                    soc=soc,
                    oc_stride=oc_stride,
                    sic=sic,
                    ic_stride=ic_stride,
                    eic=eic,
                    k_stride=k_stride,
                    src_base_addr=src_base_addr,
                    src_len=src_len,
                    ping=ping,
                    k_size=k_size,
                    kernel_hw=(k_h,k_w),
                )
            )
            self.emitted_load+=1

