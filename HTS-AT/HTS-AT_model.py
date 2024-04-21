import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # For tf.data and preprocessing only.
import keras
from keras import layers
import math

num_classes = 12
input_shape = (157, 60, 3)

patch_size = (4, 4)  # 2-by-2 sized patches
dropout_rate = 0.03  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64  # Embedding dimension
num_mlp = 256  # MLP layer size
qkv_bias = True
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window

input_window_size = 4 # Divide Input

def window_partition(x,H,W,C,window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """

    H_pad = 0
    W_pad = 0

    B = tf.shape(x)[0]

    if H % window_size != 0:
        ratio = H / window_size - H // window_size
        to_add = (1-ratio) * window_size

        if (math.ceil(to_add) + H) % window_size == 0:
            H_pad = math.ceil(to_add)
        else:
            H_pad = math.floor(to_add)

    if W % window_size != 0:
        ratio = W / window_size - W // window_size
        to_add = (1-ratio) * window_size

        if (math.ceil(to_add) + W) % window_size == 0:
            W_pad = math.ceil(to_add)
        else:
            W_pad = math.floor(to_add)
    
    x = tf.pad(x,[[0,0],[H_pad,0],[W_pad,0],[0,0]],mode='REFLECT')  
    x = tf.reshape(x, (B, x.shape[1]// window_size, window_size, x.shape[2]// window_size, window_size, C))
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, (-1, window_size, window_size, C))
    return windows


def combine_windows(windows, H, W, C, window_size,B):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        H (int): original height of x
        W (int): original width of x
        C (int): number of channels in x
        window_size (int): window size
    Returns:
        x: (B, H, W, C)
    """

    H_pad = 0
    W_pad = 0


    if H % window_size != 0:
        ratio = H / window_size - H // window_size
        to_add = (1-ratio) * window_size

        if (math.ceil(to_add) + H) % window_size == 0:
            H_pad = math.ceil(to_add)
        else:
            H_pad = math.floor(to_add)

    if W % window_size != 0:
        ratio = W / window_size - W // window_size
        to_add = (1-ratio) * window_size

        if (math.ceil(to_add) + W) % window_size == 0:
            W_pad = math.ceil(to_add)
        else:
            W_pad = math.floor(to_add)

    _H = H + H_pad
    _W = W + W_pad
    windows = tf.reshape(windows, (B, _H// window_size, _W// window_size, window_size, window_size, C))
    windows = tf.transpose(windows, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(windows,(B, (_H//window_size) * window_size, (_W//window_size) * window_size, C))
    x = x[:, H_pad:_H, W_pad:_W, :]
    x = tf.reshape(x, (-1, H, W, C))
    return x





class WindowAttention(layers.Layer):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        relative_position_index = tf.convert_to_tensor(relative_position_index)

        self.relative_position_index = tf.Variable(
            initial_value=relative_position_index,
            shape=relative_position_index.shape,
            trainable=False,
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, (-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, (2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, (0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(self.relative_position_index, (-1,))
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            relative_position_index_flat,
            axis=0,
        )
        relative_position_bias = tf.reshape(
            relative_position_bias,
            (num_window_elements, num_window_elements, -1),
        )
        relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0),
                "float32",
            )
            attn = tf.reshape(attn, (-1, nW, self.num_heads, size, size)) + mask_float
            attn = tf.reshape(attn, (-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, (0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, (-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv

class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array,mask_array.shape[1],mask_array.shape[2],mask_array.shape[3],self.window_size)
            mask_windows = tf.reshape(
                mask_windows, [-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask,
                shape=attn_mask.shape,
                dtype=attn_mask.dtype,
                trainable=False,
            )

    def call(self, x, training=False):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, (-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x,shifted_x.shape[1],shifted_x.shape[2],shifted_x.shape[3],self.window_size)
        x_windows = tf.reshape(
            x_windows, (-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows,
            (-1, self.window_size, self.window_size, channels),
        )
        shifted_x = combine_windows(
            attn_windows, height, width, channels, self.window_size,tf.shape(x_skip)[0]
        )
        

        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, (-1, height * width, channels))
        x = self.drop_path(x, training=training)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch)
        return self.proj(patch) + self.pos_embed(pos)


class PatchMerging(keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super().__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def pad_input(self,x):
        H,W = self.num_patch
        H_pad = 0
        W_pad = 0

        B = tf.shape(x)[0]

        if H % window_size != 0:
            ratio = H / window_size - H // window_size
            to_add = (1-ratio) * window_size

            if (math.ceil(to_add) + H) % window_size == 0:
                H_pad = math.ceil(to_add)
            else:
                H_pad = math.floor(to_add)

        if W % window_size != 0:
            ratio = W / window_size - W // window_size
            to_add = (1-ratio) * window_size

            if (math.ceil(to_add) + W) % window_size == 0:
                W_pad = math.ceil(to_add)
            else:
                W_pad = math.floor(to_add)
        
        x = tf.pad(x,[[0,0],[H_pad,0],[W_pad,0],[0,0]],mode='REFLECT')  

        return x

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.shape
        B = tf.shape(x)[0]
        x = tf.reshape(x, (-1, height, width, C))
        x = self.pad_input(x)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, (B, -1 , 4 * C))
        x = self.linear_trans(x)
        #x = tf.reshape(x, (B, (x.shape[1] // 2) * (x.shape[2] // 2), -1, 4 * C))

        return x


def patch_extract(images):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images=images,
        sizes=(1, patch_size[0], patch_size[1], 1),
        strides=(1, patch_size[0], patch_size[1], 1),
        rates=(1, 1, 1, 1),
        padding="SAME",
    )
    patch_dim = patches.shape[-1]
    patch_num = patches.shape[1]
    return tf.reshape(patches, (batch_size, patches.shape[1] * patches.shape[2], patch_dim)),patches.shape[1],patches.shape[2]

def get_adjusted_values(num_patch_x,num_patch_y,window_size):
    H_pad = 0
    W_pad = 0
    H = num_patch_x
    W = num_patch_y

    if H % window_size != 0:
        ratio = H / window_size - H // window_size
        to_add = (1-ratio) * window_size

        if (math.ceil(to_add) + H) % window_size == 0:
            H_pad = math.ceil(to_add)
        else:
            H_pad = math.floor(to_add)

    if W % window_size != 0:
        ratio = W / window_size - W // window_size
        to_add = (1-ratio) * window_size

        if (math.ceil(to_add) + W) % window_size == 0:
            W_pad = math.ceil(to_add)
        else:
            W_pad = math.floor(to_add)

    num_patch_x = num_patch_x + H_pad
    num_patch_y = num_patch_y + W_pad

    return (num_patch_x,num_patch_y)

def reshape_image(Input,window_size = input_window_size):
    H_pad = 0
    W_pad = 0
    H = Input.shape[1]
    W = Input.shape[2]

    if H % window_size != 0:
        ratio = H / window_size - H // window_size
        to_add = (1-ratio) * window_size

        if (math.ceil(to_add) + H) % window_size == 0:
            H_pad = math.ceil(to_add)
        else:
            H_pad = math.floor(to_add)

    if W % window_size != 0:
        ratio = W / window_size - W // window_size
        to_add = (1-ratio) * window_size

        if (math.ceil(to_add) + W) % window_size == 0:
            W_pad = math.ceil(to_add)
        else:
            W_pad = math.floor(to_add)
    
    reshaped_image = layers.ZeroPadding2D((H_pad,W_pad),name='reshape_pad')(Input)
    return reshaped_image

def apply_input_window(X):
    
    patches = None
    start = 0
    end = input_window_size
    final_num_patch_x = None
    final_num_patch_y = None
    num_patch_x = None
    num_patch_y = None

    while end <= X.shape[1]:
        slices = tf.gather(X,tf.range(start,end),axis=1)

        if patches is None:
            patches,num_patch_x,num_patch_y = patch_extract(slices)
            final_num_patch_x = num_patch_x
            final_num_patch_y = num_patch_y
        else:
            temp_patches,num_patch_x,num_patch_y = patch_extract(slices)
            final_num_patch_x += num_patch_x
            patches = tf.concat([patches,temp_patches],1)
        start += input_window_size
        end += input_window_size

    return patches,final_num_patch_x,final_num_patch_y

def get_pad_values(val,target_val):

    pad_val = 0

    if val % target_val != 0:
        ratio = val / target_val - val // target_val
        to_add = (1-ratio) * target_val

        if (math.ceil(to_add) + val) % window_size == 0:
            pad_val = math.ceil(to_add)
        else:
            pad_val = math.floor(to_add)

    return pad_val






if __name__ == '__main__':

    #X = np.random.random((32,157,60,1))
    num_layers = 4
    input_shape = (157,60,1)
    Input = tf.keras.layers.Input(input_shape)

    for i in range(num_layers):

        if  i == 0:
            X = reshape_image(Input)
            x,num_patch_x,num_patch_y = apply_input_window(X)

            x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)
            x = SwinTransformer(
            dim=embed_dim,
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            num_mlp=num_mlp,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            )(x)


            x = SwinTransformer(
            dim=embed_dim,
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            num_mlp=num_mlp,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            )(x)

            x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)

        elif i == (num_layers - 1):
            _num_patch_x,_num_patch_y = get_adjusted_values(num_patch_x // (2 ** i),num_patch_y // (2 ** i),window_size)

            embed_dim = embed_dim * 2
            
            x = SwinTransformer(
            dim= embed_dim,
            num_patch=(_num_patch_x,_num_patch_y),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            num_mlp=num_mlp,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            )(x)


            x = SwinTransformer(
            dim=embed_dim,
            num_patch=(_num_patch_x, _num_patch_y),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            num_mlp=num_mlp,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            )(x)

            x = tf.reshape(x,(-1,_num_patch_x,_num_patch_y,x.shape[-1]))
        else:

            _num_patch_x,_num_patch_y = get_adjusted_values(num_patch_x // (2 ** i),num_patch_y // (2 ** i),window_size)

            embed_dim = embed_dim * 2
            
            x = SwinTransformer(
            dim= embed_dim,
            num_patch=(_num_patch_x,_num_patch_y),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            num_mlp=num_mlp,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            )(x)


            x = SwinTransformer(
            dim=embed_dim,
            num_patch=(_num_patch_x, _num_patch_y),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            num_mlp=num_mlp,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            )(x)

            x = PatchMerging((_num_patch_x, _num_patch_y), embed_dim=embed_dim)(x)


    x = layers.ZeroPadding2D(padding = (1,0),name='Final_padding_layer')(x)
    x = layers.Conv2D(num_classes,kernel_size=(3,x.shape[2]),padding = 'VALID')(x)
    x = layers.Activation('relu',name='relu')(x)
    x = tf.squeeze(x,axis=2)

    output = layers.GlobalAveragePooling1D()(x)
    output = layers.Activation('sigmoid',name='sigmoid')(output)

    model = tf.keras.models.Model(Input,output)
    print(model.summary())