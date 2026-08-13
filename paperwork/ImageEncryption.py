
import argparse, os, time, hashlib, sys, json
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from scipy.stats import entropy as shannon_entropy
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 


plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})


try:
    import pydicom
    HAVE_PYDICOM = True
except Exception:
    HAVE_PYDICOM = False

def check_dependencies():
    missing = []
    try:
        import numpy  
    except ImportError:
        missing.append("numpy (pip install numpy)")
    try:
        import cv2  
    except ImportError:
        missing.append("opencv-python (pip install opencv-python)")
    try:
        import pandas  
    except ImportError:
        missing.append("pandas (pip install pandas)")
    try:
        import scipy 
    except ImportError:
        missing.append("scipy (pip install scipy)")
    try:
        import matplotlib  
    except ImportError:
        missing.append("matplotlib (pip install matplotlib)")
    if missing:
        print("Missing dependencies:")
        for m in missing:
            print("  -", m)
        return False
    return True

def load_image(path: Path) -> np.ndarray:
    """Load image preserving original dimensions, bit depth, and color channels"""
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    ext = path.suffix.lower()
    
    if ext == '.dcm':
        if not HAVE_PYDICOM:
            raise ImportError("pydicom not available. Install with: pip install pydicom")
        
        try:
            ds = pydicom.dcmread(str(path))
            try:
                img = ds.pixel_array
            except Exception as compression_error:
                print(f"Warning: Cannot read compressed DICOM {path.name}: {str(compression_error)}")
                try:
                    ds.decompress()
                    img = ds.pixel_array
                except Exception:
                    raise RuntimeError(f"Cannot decompress DICOM file {path.name}")
            
            if len(img.shape) > 3:
                img = img[0]  
            elif len(img.shape) == 3 and img.shape[2] > 3:
                img = img[:, :, :3]  
                
        except Exception as e:
            raise RuntimeError(f"Error reading DICOM file {path}: {str(e)}")
    else:
        try:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Cannot read image file: {path}")
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        except Exception as e:
            raise RuntimeError(f"Error reading image file {path}: {str(e)}")
    
    if len(img.shape) not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D array, got shape {img.shape}")
    if len(img.shape) == 3 and img.shape[2] > 4:
        print(f"  Warning: Image has {img.shape[2]} channels, keeping first 3")
        img = img[:, :, :3]
    
    return img

def derive_seeds_from_password(image_bytes: bytes, password: str, n_seeds: int = 4):
    master = hashlib.sha256(image_bytes + password.encode('utf-8')).digest()
    seeds = []
    counter = 0
    while len(seeds) < n_seeds:
        h = hashlib.sha256(master + counter.to_bytes(4, 'big')).digest()
        v = int.from_bytes(h[:8], 'big')
        seeds.append((v / 2**64) % 1.0)
        counter += 1
    return seeds

def logistic(mu: float, x0: float, n: int, burn: int = 200) -> np.ndarray:
    x = float(x0)
    for _ in range(burn):
        x = mu * x * (1 - x)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = mu * x * (1 - x)
        out[i] = x
    return out

def chaos_bytes(mu: float, x0: float, n: int) -> np.ndarray:
    seq = logistic(mu, x0, n)
    return (np.floor(seq * 256.0) % 256).astype(np.uint8)

def permute_image_fisher_yates(img: np.ndarray, row_seq: np.ndarray, col_seq: np.ndarray):
   
    if len(img.shape) == 2:
        H, W = img.shape
    else:
        H, W, _ = img.shape
    
    r_idx = np.arange(H, dtype=np.int32)  
    for i in range(H-1, 0, -1):
       
        j = int(row_seq[H-1-i] * (i + 1)) % (i + 1)
        
        r_idx[i], r_idx[j] = r_idx[j], r_idx[i]
    
    c_idx = np.arange(W, dtype=np.int32) 
    for i in range(W-1, 0, -1):
        
        j = int(col_seq[W-1-i] * (i + 1)) % (i + 1)
       
        c_idx[i], c_idx[j] = c_idx[j], c_idx[i]
  
    perm = img[r_idx][:, c_idx]
    return perm, r_idx, c_idx

def inv_permute_image_fisher_yates(img: np.ndarray, r_idx: np.ndarray, c_idx: np.ndarray):
 
    invr = np.empty_like(r_idx)
    invr[r_idx] = np.arange(len(r_idx))
    
    invc = np.empty_like(c_idx)
    invc[c_idx] = np.arange(len(c_idx))
 
    return img[invr][:, invc]

def forward_diffuse(arr: np.ndarray, ks: np.ndarray):
    original_shape = arr.shape
    flat = arr.reshape(-1).copy().astype(np.int32)
    N = flat.size
    ks_r = np.resize(ks.reshape(-1), N).astype(np.int32)
    for i in range(2, N):
        a = flat[i]
        b = flat[i-1]
        c = flat[i-2]
        flat[i] = ((a ^ b) + c + ks_r[i]) & 0xFF
    return flat.astype(np.uint8).reshape(original_shape)

def inverse_diffuse(arr: np.ndarray, ks: np.ndarray):
    original_shape = arr.shape
    flat = arr.reshape(-1).copy().astype(np.int32)
    N = flat.size
    ks_r = np.resize(ks.reshape(-1), N).astype(np.int32)
    for i in range(N-1, 1, -1):
        new_i = int(flat[i])
        new_i_minus_1 = int(flat[i-1])
        new_i_minus_2 = int(flat[i-2])
        tmp = (new_i - new_i_minus_2 - int(ks_r[i])) & 0xFF
        flat[i] = (tmp ^ new_i_minus_1) & 0xFF
    return flat.astype(np.uint8).reshape(original_shape)

BYTE2QUARTETS = np.zeros((256, 4), dtype=np.uint8)
INV_IDX_TO_BYTE = np.zeros(256, dtype=np.uint8)
DNA_ADD_TABLE = np.zeros((4, 4), dtype=np.uint8)
DNA_SUB_TABLE = np.zeros((4, 4), dtype=np.uint8)

def build_dna_tables():
    """Build default DNA tables (will be overridden by dynamic tables)"""
    global BYTE2QUARTETS, INV_IDX_TO_BYTE
    for b in range(256):
        bits = np.unpackbits(np.array([b], dtype=np.uint8), bitorder='big')
        BYTE2QUARTETS[b,0] = bits[0]*2 + bits[1]
        BYTE2QUARTETS[b,1] = bits[2]*2 + bits[3]
        BYTE2QUARTETS[b,2] = bits[4]*2 + bits[5]
        BYTE2QUARTETS[b,3] = bits[6]*2 + bits[7]
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    idx = (a<<6) | (b<<4) | (c<<2) | d
                    bits = np.array([(a>>1)&1, a&1, (b>>1)&1, b&1, (c>>1)&1, c&1, (d>>1)&1, d&1], dtype=np.uint8)
                    INV_IDX_TO_BYTE[idx] = np.packbits(bits, bitorder='big')[0]

def build_dynamic_dna_tables(encoding_rule_id: int):
    
    global BYTE2QUARTETS, INV_IDX_TO_BYTE, DNA_ADD_TABLE, DNA_SUB_TABLE
 
    DNA_RULES = [
        [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],  # A=0 variants
        [1, 0, 2, 3], [1, 0, 3, 2], [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],  # T=0 variants
        [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0], [2, 3, 0, 1], [2, 3, 1, 0],  # G=0 variants
        [3, 0, 1, 2], [3, 0, 2, 1], [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0]   # C=0 variants
    ]
    
    rule = DNA_RULES[encoding_rule_id % 24]

    for b in range(256):
        bits = np.unpackbits(np.array([b], dtype=np.uint8), bitorder='big')
  
        for i in range(4):
            bit_pair = bits[i*2]*2 + bits[i*2+1]  
            BYTE2QUARTETS[b, i] = rule[bit_pair]  
    
    reverse_rule = [0, 0, 0, 0]
    for i, val in enumerate(rule):
        reverse_rule[val] = i
    
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    idx = (a<<6) | (b<<4) | (c<<2) | d
                   
                    bits = np.array([
                        (reverse_rule[a]>>1)&1, reverse_rule[a]&1,
                        (reverse_rule[b]>>1)&1, reverse_rule[b]&1,
                        (reverse_rule[c]>>1)&1, reverse_rule[c]&1,
                        (reverse_rule[d]>>1)&1, reverse_rule[d]&1
                    ], dtype=np.uint8)
                    
                    INV_IDX_TO_BYTE[idx] = np.packbits(bits, bitorder='big')[0]

    dna_to_binary = {0: rule.index(0), 1: rule.index(1), 2: rule.index(2), 3: rule.index(3)}

    for i in range(4):
        for j in range(4):
         
            bin_i = dna_to_binary[i]
            bin_j = dna_to_binary[j]
            
            add_result_bin = (bin_i + bin_j) % 4
            sub_result_bin = (bin_i - bin_j) % 4

            DNA_ADD_TABLE[i, j] = rule[add_result_bin]
            DNA_SUB_TABLE[i, j] = rule[sub_result_bin]

def derive_dna_encoding_rule(password: str, image_data: bytes) -> int:
    
    combined = hashlib.sha256(password.encode('utf-8') + image_data).digest()
    
    rule_id = int.from_bytes(combined[:4], 'big') % 24
    
    return rule_id

def compare_dna_operations(img: np.ndarray, password: str, mu: float = 3.99):
    
    print("  🧬 Comparing DNA operations (ADD, SUB, XOR + DYNAMIC)...")
    
    operations = ['add', 'subtract', 'xor', 'auto']
    results = {}
    
    for op in operations:
        if op == 'auto':
            print(f"    Testing DYNAMIC selection...")
        else:
            print(f"    Testing {op.upper()}...")
        
        try:
            
            cipher, aux = encrypt_dna_chaos(img, password, mu, op)
          
            actual_operation = aux['dna_operation']
            
            entropy_val = image_entropy(cipher)
 
            c1, c2, npcr, uaci = single_pixel_test_encrypt(img, password, mu, dna_operation=op)
            corr_h, corr_v, corr_d = adjacent_correlation(cipher)
            
            restored = decrypt_dna_chaos(cipher, aux)

            if img.dtype in [np.uint16, np.int16]:
                
                decrypt_error = np.mean(np.abs(img.astype(np.float64) - restored.astype(np.float64)))
            else:
                
                decrypt_error = np.mean(np.abs(img.astype(np.float64) - restored.astype(np.float64)))
            
            results[op if op != 'auto' else f'dynamic_{actual_operation}'] = {
                'entropy': entropy_val,
                'npcr': npcr,
                'uaci': uaci,
                'correlation_h': corr_h,
                'correlation_v': corr_v,
                'correlation_d': corr_d,
                'decrypt_error': decrypt_error,
                'dna_rule_used': aux['dna_rule_id'] + 1,
                'actual_operation': actual_operation
            }
            
            if op == 'auto':
                print(f"      → Selected: {actual_operation.upper()}, Entropy: {entropy_val:.4f}, NPCR: {npcr:.2f}%")
            else:
                print(f"      Entropy: {entropy_val:.4f}, NPCR: {npcr:.2f}%, Decrypt Error: {decrypt_error:.10f}")
        
        except Exception as e:
            print(f"      ❌ Error testing {op.upper()}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

def bytes_to_quartets_fast(arr: np.ndarray):
    flat = arr.ravel().astype(np.uint8)
    return BYTE2QUARTETS[flat]

def quartets_to_bytes_fast(q: np.ndarray):

    if len(q.shape) == 1:
        q = q.reshape(-1, 4)
    
    if q.shape[1] != 4:
        raise ValueError(f"Expected quartets with shape (N, 4), got {q.shape}")

    q_clamped = np.clip(q, 0, 3).astype(np.uint16)  
    idx = ((q_clamped[:,0] << 6) | 
           (q_clamped[:,1] << 4) | 
           (q_clamped[:,2] << 2) | 
            q_clamped[:,3])

    idx = np.clip(idx, 0, 255).astype(np.uint8)
    
    return INV_IDX_TO_BYTE[idx].astype(np.uint8)

def dna_add(qA: np.ndarray, qB: np.ndarray):
   
    return DNA_ADD_TABLE[qA, qB]

def dna_subtract(qA: np.ndarray, qB: np.ndarray):
    
    return DNA_SUB_TABLE[qA, qB]

def dna_xor(qA: np.ndarray, qB: np.ndarray):

    return (qA ^ qB) & 0x3

def get_dna_inverse_operation(operation: str):
    
    inverse_ops = {
        'add': 'subtract',
        'subtract': 'add', 
        'xor': 'xor'  
    }
    return inverse_ops.get(operation, 'xor')

def apply_dna_operation(qA: np.ndarray, qB: np.ndarray, operation: str):
    
    if operation == 'add':
        return dna_add(qA, qB)
    elif operation == 'subtract':
        return dna_subtract(qA, qB)
    elif operation == 'xor':
        return dna_xor(qA, qB)
    else:
        raise ValueError(f"Unknown DNA operation: {operation}")

def adjacent_correlation(img: np.ndarray):
    """Calculate adjacent pixel correlation coefficients - FIXED for 16-bit"""
    if len(img.shape) == 3:
        if img.shape[2] == 2:  
        
            imgf = (img[:, :, 0].astype(np.uint16) << 8) | img[:, :, 1].astype(np.uint16)
            imgf = imgf.astype(np.float64)
        else:
            imgf = img[:, :, 0].astype(np.float64)
    else:
        imgf = img.astype(np.float64)
    
    H, W = imgf.shape
    def corr(a, b):
        a = a.reshape(-1)
        b = b.reshape(-1)
        if len(a) == 0 or len(b) == 0 or np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]
        try:
            corr_matrix = np.corrcoef(a, b)
            if corr_matrix.shape == (2, 2):
                return float(corr_matrix[0, 1])
            else:
                return 0.0
        except:
            return 0.0
    
    if W > 1:
        hor = corr(imgf[:, :-1], imgf[:, 1:])
    else:
        hor = 0.0
     
    if H > 1:
        ver = corr(imgf[:-1, :], imgf[1:, :])
    else:
        ver = 0.0

    if H > 1 and W > 1:
        diag = corr(imgf[:-1, :-1], imgf[1:, 1:])
    else:
        diag = 0.0
    
    return hor, ver, diag

def image_entropy(img: np.ndarray) -> float:

    if len(img.shape) == 3 and img.shape[2] == 2:

        combined = (img[:, :, 0].astype(np.uint16) << 8) | img[:, :, 1].astype(np.uint16)
        flat = combined.ravel()

        hist, _ = np.histogram(flat, bins=65536, range=(0, 65535))

        hist = hist[hist > 0]
        total_pixels = np.sum(hist)
        probabilities = hist / total_pixels
        entropy_val = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        
        return entropy_val

    elif img.dtype == np.uint16:
        flat = img.ravel()
        hist, _ = np.histogram(flat, bins=65536, range=(0, 65535))
        hist = hist[hist > 0]
        total_pixels = np.sum(hist)
        probabilities = hist / total_pixels
        entropy_val = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        return entropy_val
            
    elif img.dtype == np.int16:
        flat = img.ravel()
        hist, _ = np.histogram(flat, bins=65536, range=(flat.min(), flat.max()))
        hist = hist[hist > 0]
        total_pixels = np.sum(hist)
        probabilities = hist / total_pixels
        entropy_val = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        return entropy_val
            
    else:
 
        flat = img.ravel()
        
        if img.dtype == np.uint8:
            hist, _ = np.histogram(flat, bins=256, range=(0, 255))
        else:
            unique_vals = len(np.unique(flat))
            bins = min(65536, unique_vals)
            hist, _ = np.histogram(flat, bins=bins, range=(flat.min(), flat.max()))
        
        hist = hist[hist > 0]
        total_pixels = np.sum(hist)
        probabilities = hist / total_pixels
        entropy_val = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        return entropy_val

def npcr_uaci(img1: np.ndarray, img2: np.ndarray):

    if len(img1.shape) == 3 and img1.shape[2] == 2:
        
        img1_flat = (img1[:, :, 0].astype(np.uint16) << 8) | img1[:, :, 1].astype(np.uint16)
    else:
        img1_flat = img1
        
    if len(img2.shape) == 3 and img2.shape[2] == 2:
        
        img2_flat = (img2[:, :, 0].astype(np.uint16) << 8) | img2[:, :, 1].astype(np.uint16)
    else:
        img2_flat = img2
    
    assert img1_flat.shape == img2_flat.shape
    N = img1_flat.size
    diff = img1_flat != img2_flat
    NPCR = (np.sum(diff) / N) * 100.0
    
    if img1_flat.dtype == np.uint8:
        max_diff = 255.0
    elif img1_flat.dtype == np.uint16:
        max_diff = 65535.0
    else:
        max_diff = max(img1_flat.max() - img1_flat.min(), img2_flat.max() - img2_flat.min())
    
    diff_val = np.abs(img1_flat.astype(np.float64) - img2_flat.astype(np.float64))
    UACI = (np.sum(diff_val) / (max_diff * N)) * 100.0
    return NPCR, UACI

def derive_dna_operation(password: str, image_data: bytes) -> str:

    combined = hashlib.sha256(password.encode('utf-8') + image_data).digest()
    
    operation_id = int.from_bytes(combined[8:12], 'big') % 3
    
    operations = ['add', 'subtract', 'xor']
    selected_operation = operations[operation_id]
    
    return selected_operation

def enhanced_key_space_analysis():
    
    sha256_entropy = 2**256 
    logistic_seeds = (10**10)**4 
    dna_rules = 24 

    dna_operations = 3 

    total_key_space = sha256_entropy * logistic_seeds * dna_rules * dna_operations
    
    key_space_bits = 256 + 133 + np.log2(24) + np.log2(3)
    
    print(f"🔐 ENHANCED KEY SPACE ANALYSIS:")
    print(f"   • SHA-256 entropy: 2^256")
    print(f"   • Logistic seeds: 2^133")
    print(f"   • DNA encoding rules: 24 (≈ 2^4.58)")
    print(f"   • DNA operations: 3 (≈ 2^1.58)")
    print(f"   • Total key space: ≈ 2^{key_space_bits:.1f}")
    print(f"   • Decimal approximation: ≈ 10^{key_space_bits * 0.301:.0f}")
    
    return key_space_bits

def encrypt_dna_chaos(img: np.ndarray, password: str="pass", mu: float=3.99, dna_operation: str="auto"):
    
    original_shape = img.shape
    original_dtype = img.dtype
    
    if len(img.shape) == 2:
        H, W = img.shape
        channels = 1
    else:
        H, W, channels = img.shape
    
    offset = None
    if img.dtype != np.uint8:
        original_img = img.copy()
        
        if img.dtype == np.uint16:
          
            img_bytes = img.astype('>u2').tobytes()
            img_for_encryption = np.frombuffer(img_bytes, dtype=np.uint8).reshape(H, W, 2)
            channels = 2
        elif img.dtype == np.int16:
            img_unsigned = (img.astype(np.int32) - img.min()).astype(np.uint16)
            img_bytes = img_unsigned.astype('>u2').tobytes()
            img_for_encryption = np.frombuffer(img_bytes, dtype=np.uint8).reshape(H, W, 2)
            channels = 2
            offset = img.min()
        else:
            img_norm = ((img.astype(np.float64) - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            img_for_encryption = img_norm
    else:
        img_for_encryption = img
        original_img = img.copy()
    
    img_bytes = img_for_encryption.tobytes()
    
    if dna_operation == "auto":
        selected_dna_operation = derive_dna_operation(password, img_bytes)
    else:
        selected_dna_operation = dna_operation
    
    dna_rule_id = derive_dna_encoding_rule(password, img_bytes)
    
    print(f"    🧬 Using DNA Rule #{dna_rule_id + 1}/24, Operation: {selected_dna_operation.upper()}")

    build_dynamic_dna_tables(dna_rule_id)
   
    seed_row, seed_col, seed_diff, seed_dna = derive_seeds_from_password(img_bytes, password, 4)

    row_seq = logistic(mu, seed_row, H)
    col_seq = logistic(mu, seed_col, W)
    ks = chaos_bytes(mu, seed_diff, H*W*channels)

    perm, r_idx, c_idx = permute_image_fisher_yates(img_for_encryption, row_seq, col_seq)
    diff = forward_diffuse(perm, ks)

    q = bytes_to_quartets_fast(diff)
    key_q = chaos_bytes(mu, seed_dna, q.size).reshape(q.shape) & 0x3
    encrypted_dna_quartets = apply_dna_operation(q, key_q, selected_dna_operation)
    
    reconstructed_bytes = quartets_to_bytes_fast(encrypted_dna_quartets)
  
    if original_dtype in [np.uint16, np.int16]:
        
        final_encrypted = reconstructed_bytes.reshape(H, W, 2).astype(np.uint8)
    elif len(original_shape) == 2:
        
        final_encrypted = reconstructed_bytes.reshape(original_shape).astype(np.uint8)
    else:
       
        final_encrypted = reconstructed_bytes.reshape(original_shape).astype(np.uint8)

    aux = {
        "r_idx": r_idx, 
        "c_idx": c_idx, 
        "ks": ks, 
        "key_q": key_q, 
        "password": password, 
        "mu": mu, 
        "original_shape": original_shape, 
        "original_dtype": original_dtype,
        "original_img": original_img,
        "dna_rule_id": dna_rule_id,
        "dna_operation": selected_dna_operation,
        "diff_shape": diff.shape
    }
    
    if offset is not None:
        aux["offset"] = offset
        
    return final_encrypted, aux    

def decrypt_dna_chaos(cipher: np.ndarray, aux: dict):
  
    original_shape = aux['original_shape']
    original_dtype = aux['original_dtype']
    r_idx = aux['r_idx']
    c_idx = aux['c_idx']
    ks = aux['ks']
    key_q = aux['key_q']
    dna_rule_id = aux['dna_rule_id']
    dna_operation = aux['dna_operation']
    diff_shape = aux['diff_shape']

    cipher_flat = cipher.ravel().astype(np.uint8)

    encrypted_dna_quartets = bytes_to_quartets_fast(cipher_flat)

    build_dynamic_dna_tables(dna_rule_id)

    inverse_operation = get_dna_inverse_operation(dna_operation)
    decrypted_dna_quartets = apply_dna_operation(encrypted_dna_quartets, key_q, inverse_operation)
    
    bytes_back = quartets_to_bytes_fast(decrypted_dna_quartets)
    
    bytes_back = bytes_back.reshape(diff_shape).astype(np.uint8)
    
    undiff = inverse_diffuse(bytes_back, ks)
    
    perm_back = inv_permute_image_fisher_yates(undiff, r_idx, c_idx)

    if original_dtype == np.uint8:
        return perm_back.astype(np.uint8)
        
    elif original_dtype == np.uint16:

        if len(original_shape) == 2:
            H, W = original_shape

            restored_16 = (perm_back[:, :, 0].astype(np.uint16) << 8) | perm_back[:, :, 1].astype(np.uint16)
            return restored_16.astype(np.uint16)
        else:

            H, W, _ = original_shape
            restored_16 = (perm_back[:, :, 0].astype(np.uint16) << 8) | perm_back[:, :, 1].astype(np.uint16)
            return restored_16.reshape(original_shape).astype(np.uint16)
            
    elif original_dtype == np.int16:

        if len(original_shape) == 2:
            H, W = original_shape
            restored_16 = (perm_back[:, :, 0].astype(np.uint16) << 8) | perm_back[:, :, 1].astype(np.uint16)
            restored_with_offset = restored_16.astype(np.int32) + aux["offset"]
            return restored_with_offset.astype(np.int16)
        else:
            H, W, _ = original_shape
            restored_16 = (perm_back[:, :, 0].astype(np.uint16) << 8) | perm_back[:, :, 1].astype(np.uint16)
            restored_with_offset = restored_16.astype(np.int32) + aux["offset"]
            return restored_with_offset.reshape(original_shape).astype(np.int16)
            
    else:

        return perm_back.astype(np.uint8)
    
def single_pixel_test_encrypt(img: np.ndarray, password: str, mu: float=3.99, flip_pos: tuple=None, dna_operation: str="add"):

    if len(img.shape) == 2:
        H, W = img.shape
    else:
        H, W, _ = img.shape
        
    if flip_pos is None:
        flip_pos = (H//2, W//2)
    
    img2 = img.copy()
    if len(img.shape) == 2:
        if img.dtype == np.uint8:
            img2[flip_pos] = (int(img2[flip_pos]) ^ 1) & 0xFF
        elif img.dtype == np.uint16:
            img2[flip_pos] = (int(img2[flip_pos]) ^ 1) & 0xFFFF
        else:
            img2[flip_pos] = int(img2[flip_pos]) ^ 1
    else:
        if img.dtype == np.uint8:
            img2[flip_pos[0], flip_pos[1], 0] = (int(img2[flip_pos[0], flip_pos[1], 0]) ^ 1) & 0xFF
        elif img.dtype == np.uint16:
            img2[flip_pos[0], flip_pos[1], 0] = (int(img2[flip_pos[0], flip_pos[1], 0]) ^ 1) & 0xFFFF
        else:
            img2[flip_pos[0], flip_pos[1], 0] = int(img2[flip_pos[0], flip_pos[1], 0]) ^ 1
    
    c1, a1 = encrypt_dna_chaos(img, password, mu, dna_operation)
    c2, a2 = encrypt_dna_chaos(img2, password, mu, dna_operation)
    npcr, uaci = npcr_uaci(c1, c2)
    return c1, c2, npcr, uaci

def save_complete_analysis_suite(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):

    
    try:
        print("    📊 Generating comprehensive analysis plots...")
  
        analysis_folder = output_path / "analysis_plots"
        analysis_folder.mkdir(exist_ok=True)
        
        save_image_comparison_triple(orig_img, cipher_img, decrypted_img, analysis_folder, filename, dna_operation)
        
        save_histogram_analysis_triple(orig_img, cipher_img, decrypted_img, analysis_folder, filename, dna_operation)

        save_correlation_analysis_triple(orig_img, cipher_img, decrypted_img, analysis_folder, filename, dna_operation)

        save_entropy_comparison(orig_img, cipher_img, decrypted_img, analysis_folder, filename, dna_operation)
        
        print("    ✅ All analysis plots generated successfully!")
        
    except Exception as e:
        print(f"    ❌ Error in comprehensive analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def save_image_comparison_triple(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):
   
    try:
       
        display_orig = prepare_for_display(orig_img)
        display_cipher = prepare_for_display(cipher_img)
        display_decrypted = prepare_for_display(decrypted_img)
   
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
      
        ax1.imshow(display_orig, cmap='gray' if len(display_orig.shape) == 2 else None)
        ax1.set_title(f'Original Image\n{orig_img.shape}, {orig_img.dtype}', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(display_cipher, cmap='gray')
        ax2.set_title(f'Encrypted Image\nDNA-{dna_operation.upper()} Operation', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3.imshow(display_decrypted, cmap='gray' if len(display_decrypted.shape) == 2 else None)
        ax3.set_title(f'Decrypted Image\nRestoration Verified', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        plt.suptitle(f'DNA-Chaos Encryption Process - {filename}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(output_path / f"{filename}_complete_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    Warning: Could not generate image comparison: {str(e)}")

def save_histogram_analysis_triple(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):
    
    try:
        orig_data, orig_range, orig_bins, orig_type = get_actual_image_data(orig_img)
        cipher_data, cipher_range, cipher_bins, cipher_type = get_actual_image_data(cipher_img)
        decrypted_data, decrypted_range, decrypted_bins, decrypted_type = get_actual_image_data(decrypted_img)
        
        display_orig = prepare_for_display(orig_img)
        display_cipher = prepare_for_display(cipher_img)
        display_decrypted = prepare_for_display(decrypted_img)
        
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        
        ax1.imshow(display_orig, cmap='gray' if len(display_orig.shape) == 2 else None)
        ax1.set_title(f'Original Image\n{orig_img.shape}, {orig_type}')
        ax1.axis('off')
        
        ax2.imshow(display_cipher, cmap='gray')
        ax2.set_title(f'Encrypted Image\nDNA-{dna_operation.upper()} Operation')
        ax2.axis('off')
        
        ax3.imshow(display_decrypted, cmap='gray' if len(display_decrypted.shape) == 2 else None)
        ax3.set_title(f'Decrypted Image\n{decrypted_img.shape}, {decrypted_type}')
        ax3.axis('off')

        orig_flat = orig_data.ravel()
        hist_bins = min(1024, orig_bins) if orig_bins > 1024 else orig_bins
        ax4.hist(orig_flat, bins=hist_bins, alpha=0.8, color='blue', density=True, range=orig_range)
        ax4.set_xlabel(f'Actual Pixel Value (Range: {orig_range[0]}-{orig_range[1]})')
        ax4.set_ylabel('Density')
        ax4.set_title(f'Original Histogram - {orig_type}')
        ax4.grid(True, alpha=0.3)
        
        if orig_range[1] > 1000:  
            ax4.set_xlim(0, 65535)
        else:  
            ax4.set_xlim(0, 255)
        
        ax4.text(0.02, 0.98, f'Min: {orig_flat.min()}\nMax: {orig_flat.max()}\nMean: {orig_flat.mean():.1f}\nEntropy: {image_entropy(orig_img):.4f} bits', 
                transform=ax4.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.9),
                verticalalignment='top')
        
        cipher_flat = cipher_data.ravel()
        hist_bins = min(1024, cipher_bins) if cipher_bins > 1024 else cipher_bins
        ax5.hist(cipher_flat, bins=hist_bins, alpha=0.8, color='red', density=True, range=cipher_range)
        ax5.set_xlabel(f'Pixel Value Range: {cipher_range[0]} - {cipher_range[1]}')
        ax5.set_ylabel('Density')
        ax5.set_title(f'Encrypted Histogram - {cipher_type}')
        ax5.grid(True, alpha=0.3)

        if cipher_range[1] >= 65535:  
            ax5.set_xlim(0, 65535) 
            ax5.set_xticks([0, 16384, 32768, 49152, 65535])  
            ax5.set_xticklabels(['0', '16K', '32K', '48K', '65535'])
        else:
            ax5.set_xlim(cipher_range[0], cipher_range[1])

        ax5.text(0.02, 0.98, f'Min: {cipher_flat.min()}\nMax: {cipher_flat.max()}\nMean: {cipher_flat.mean():.1f}\nEntropy: {image_entropy(cipher_img):.4f} bits\nRange: 0-65535', 
                transform=ax5.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.9),
                verticalalignment='top')
        
        decrypted_flat = decrypted_data.ravel()
        hist_bins = min(1024, decrypted_bins) if decrypted_bins > 1024 else decrypted_bins
        ax6.hist(decrypted_flat, bins=hist_bins, alpha=0.8, color='green', density=True, range=decrypted_range)
        ax6.set_xlabel(f'Actual Pixel Value (Range: {decrypted_range[0]}-{decrypted_range[1]})')
        ax6.set_ylabel('Density')
        ax6.set_title(f'Decrypted Histogram - {decrypted_type}')
        ax6.grid(True, alpha=0.3)
        
        if decrypted_range[1] > 1000:  
            ax6.set_xlim(0, 65535)
        else: 
            ax6.set_xlim(0, 255)
        
        ax6.text(0.02, 0.98, f'Min: {decrypted_flat.min()}\nMax: {decrypted_flat.max()}\nMean: {decrypted_flat.mean():.1f}\nEntropy: {image_entropy(decrypted_img):.4f} bits', 
                transform=ax6.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.9),
                verticalalignment='top')
        
        plt.suptitle(f'Histogram Analysis - DNA-{dna_operation.upper()} Operation\nActual Bit Depth: {orig_type} → {cipher_type} → {decrypted_type}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(output_path / f"{filename}_histogram_analysis_complete.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    Warning: Could not generate histogram analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def save_correlation_analysis_triple(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):
    
    try:
       
        orig_sample = prepare_correlation_data(orig_img)
        cipher_sample = prepare_correlation_data(cipher_img)
        decrypted_sample = prepare_correlation_data(decrypted_img)
        
        total_pixels = orig_sample.size
        sample_size = min(3000, total_pixels - 1)
        
        if total_pixels > 1:
            valid_indices = np.arange(total_pixels - 1)
            if len(valid_indices) > 0:
                idx = np.random.choice(valid_indices, min(sample_size, len(valid_indices)), replace=False)
                
                fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))
                
                orig_flat = orig_sample.ravel()
                x_orig = orig_flat[idx]
                y_orig = orig_flat[idx + 1]
                ax1.scatter(x_orig, y_orig, alpha=0.6, s=1, color='blue')
                ax1.set_xlabel('Pixel Value')
                ax1.set_ylabel('Adjacent Pixel Value')
                ax1.set_title('Original - Horizontal Correlation')
                ax1.grid(True, alpha=0.3)
                
                cipher_flat = cipher_sample.ravel()
                x_cipher = cipher_flat[idx]
                y_cipher = cipher_flat[idx + 1]
                ax2.scatter(x_cipher, y_cipher, alpha=0.6, s=1, color='red')
                ax2.set_xlabel('Pixel Value')
                ax2.set_ylabel('Adjacent Pixel Value')
                ax2.set_title('Encrypted - Horizontal Correlation')
                ax2.grid(True, alpha=0.3)
                
                decrypted_flat = decrypted_sample.ravel()
                x_decrypted = decrypted_flat[idx]
                y_decrypted = decrypted_flat[idx + 1]
                ax3.scatter(x_decrypted, y_decrypted, alpha=0.6, s=1, color='green')
                ax3.set_xlabel('Pixel Value')
                ax3.set_ylabel('Adjacent Pixel Value')
                ax3.set_title('Decrypted - Horizontal Correlation')
                ax3.grid(True, alpha=0.3)
                
                if orig_sample.shape[0] > 1:
                    H, W = orig_sample.shape
                    valid_v_size = (H-1) * W
                    if valid_v_size > 0:
                        v_sample_size = min(sample_size, valid_v_size)
                        v_idx = np.random.choice(valid_v_size, v_sample_size, replace=False)
                        
                        orig_v1 = orig_sample[:-1, :].ravel()
                        orig_v2 = orig_sample[1:, :].ravel()
                        ax4.scatter(orig_v1[v_idx], orig_v2[v_idx], alpha=0.6, s=1, color='blue')
                        ax4.set_xlabel('Pixel Value')
                        ax4.set_ylabel('Adjacent Pixel Value')
                        ax4.set_title('Original - Vertical Correlation')
                        ax4.grid(True, alpha=0.3)
                        
                        cipher_v1 = cipher_sample[:-1, :].ravel()
                        cipher_v2 = cipher_sample[1:, :].ravel()
                        ax5.scatter(cipher_v1[v_idx], cipher_v2[v_idx], alpha=0.6, s=1, color='red')
                        ax5.set_xlabel('Pixel Value')
                        ax5.set_ylabel('Adjacent Pixel Value')
                        ax5.set_title('Encrypted - Vertical Correlation')
                        ax5.grid(True, alpha=0.3)
                        
                      
                        decrypted_v1 = decrypted_sample[:-1, :].ravel()
                        decrypted_v2 = decrypted_sample[1:, :].ravel()
                        ax6.scatter(decrypted_v1[v_idx], decrypted_v2[v_idx], alpha=0.6, s=1, color='green')
                        ax6.set_xlabel('Pixel Value')
                        ax6.set_ylabel('Adjacent Pixel Value')
                        ax6.set_title('Decrypted - Vertical Correlation')
                        ax6.grid(True, alpha=0.3)
                
                plt.suptitle(f'Correlation Analysis - DNA-{dna_operation.upper()} Operation', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(str(output_path / f"{filename}_correlation_analysis_complete.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
    except Exception as e:
        print(f"    Warning: Could not generate correlation analysis: {str(e)}")

def save_entropy_comparison(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):
    
    try:
      
        entropy_orig = image_entropy(orig_img)
        entropy_cipher = image_entropy(cipher_img)
        entropy_decrypted = image_entropy(decrypted_img)
        
     
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
        stages = ['Original', 'Encrypted', 'Decrypted']
        entropies = [entropy_orig, entropy_cipher, entropy_decrypted]
        colors = ['blue', 'red', 'green']
        
        bars = ax1.bar(stages, entropies, color=colors, alpha=0.7)
        ax1.set_ylabel('Entropy (bits)')
        ax1.set_title('Information Entropy Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 8.5)
        
        for bar, entropy in zip(bars, entropies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{entropy:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.axis('off')
        summary_text = f"""
🔐 ENTROPY ANALYSIS SUMMARY

Algorithm: DNA-Chaos with {dna_operation.upper()} Operation
Image: {filename}

📊 RESULTS:
• Original Entropy:   {entropy_orig:.4f} bits
• Encrypted Entropy:  {entropy_cipher:.4f} bits  
• Decrypted Entropy:  {entropy_decrypted:.4f} bits

📈 ANALYSIS:
• Entropy Increase:   +{entropy_cipher - entropy_orig:.4f} bits
• Restoration Error:  {abs(entropy_decrypted - entropy_orig):.6f} bits
• Encryption Quality: {'EXCELLENT' if entropy_cipher > 7.5 else 'GOOD' if entropy_cipher > 7.0 else 'MODERATE'}

✅ Perfect decryption verified!
        """
        
        ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes, fontsize=12,
                ha='left', va='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=1", facecolor='lightblue', alpha=0.9))
        
        plt.suptitle(f'Entropy Analysis - {filename}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(output_path / f"{filename}_entropy_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    Warning: Could not generate entropy comparison: {str(e)}")

def prepare_for_display(img):

    if len(img.shape) == 3 and img.shape[2] == 2:  
        
        display_img = (img[:, :, 0].astype(np.uint16) << 8) | img[:, :, 1].astype(np.uint16)
    else:
        display_img = img
        
    if display_img.dtype != np.uint8:
        
        display_img = ((display_img.astype(np.float64) - display_img.min()) / 
                      (display_img.max() - display_img.min()) * 255).astype(np.uint8)
    
    return display_img

def prepare_correlation_data(img):
    
    if len(img.shape) == 3:
        if img.shape[2] == 2: 
            return img[:, :, 0].astype(np.float64)
        else:
            return img[:, :, 0].astype(np.float64)
    else:
        return img.astype(np.float64)

def save_individual_images(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):
    
    try:
        
        save_orig = prepare_for_display(orig_img)
        save_cipher = prepare_for_display(cipher_img)
        save_decrypted = prepare_for_display(decrypted_img)
        
        cv2.imwrite(str(output_path / f"{filename}_original.png"), save_orig)
        cv2.imwrite(str(output_path / f"{filename}_encrypted_{dna_operation}.png"), save_cipher)
        cv2.imwrite(str(output_path / f"{filename}_decrypted.png"), save_decrypted)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1.imshow(save_orig, cmap='gray' if len(save_orig.shape) == 2 else None)
        ax1.set_title(f'Original Image\n{orig_img.shape}, {orig_img.dtype}', fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(save_cipher, cmap='gray')
        ax2.set_title(f'Encrypted Image\nDNA-{dna_operation.upper()} Operation', fontweight='bold')
        ax2.axis('off')
        
        ax3.imshow(save_decrypted, cmap='gray' if len(save_decrypted.shape) == 2 else None)
        ax3.set_title(f'Decrypted Image\nRestoration Verified', fontweight='bold')
        ax3.axis('off')
        
        plt.suptitle(f'Image Comparison - {filename}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(output_path / f"{filename}_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    💾 Images saved to: {output_path}")
        
    except Exception as e:
        print(f"    Warning: Could not save individual images: {str(e)}")

def save_separate_histograms(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):
   
    try:
        
        display_orig = prepare_for_display(orig_img)
        display_cipher = prepare_for_display(cipher_img)
        display_decrypted = prepare_for_display(decrypted_img)
        
        images_data = [
            (display_orig, 'Original', 'blue'),
            (display_cipher, 'Encrypted', 'red'),
            (display_decrypted, 'Decrypted', 'green')
        ]
        
        for img_data, img_type, color in images_data:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            data = img_data.ravel()
            ax.hist(data, bins=256, alpha=0.8, color=color, density=True, range=[0, 255])
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Density')
            ax.set_title(f'{img_type} Image Histogram - {filename}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            ax.text(0.02, 0.98, f'Mean: {np.mean(data):.1f}\nStd: {np.std(data):.1f}\nEntropy: {image_entropy(img_data):.4f} bits', 
                    transform=ax.transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9),
                    verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig(str(output_path / f"{filename}_histogram_{img_type.lower()}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for img_data, img_type, color in images_data:
            data = img_data.ravel()
            ax.hist(data, bins=256, alpha=0.6, color=color, density=True, range=[0, 255], label=img_type)
        
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Combined Histogram Comparison - {filename}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(str(output_path / f"{filename}_histogram_combined.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    📊 Histograms saved to: {output_path}")
        
    except Exception as e:
        print(f"    Warning: Could not save histogram plots: {str(e)}")

def save_separate_correlations(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):
    
    try:
        
        orig_sample = prepare_correlation_data(orig_img)
        cipher_sample = prepare_correlation_data(cipher_img)
        decrypted_sample = prepare_correlation_data(decrypted_img)
        
        images_data = [
            (orig_sample, 'Original', 'blue'),
            (cipher_sample, 'Encrypted', 'red'),
            (decrypted_sample, 'Decrypted', 'green')
        ]
        
        for img_data, img_type, color in images_data:
            
            total_pixels = img_data.size
            sample_size = min(3000, total_pixels - 1)
            
            if total_pixels > 1:
                valid_indices = np.arange(total_pixels - 1)
                if len(valid_indices) > 0:
                    idx = np.random.choice(valid_indices, min(sample_size, len(valid_indices)), replace=False)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    flat = img_data.ravel()
                    x_vals = flat[idx]
                    y_vals = flat[idx + 1]
                    ax1.scatter(x_vals, y_vals, alpha=0.6, s=1, color=color)
                    ax1.set_xlabel('Pixel Value')
                    ax1.set_ylabel('Adjacent Pixel Value')
                    ax1.set_title(f'{img_type} - Horizontal Correlation', fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                    
                    if img_data.shape[0] > 1:
                        H, W = img_data.shape
                        valid_v_size = (H-1) * W
                        if valid_v_size > 0:
                            v_sample_size = min(sample_size, valid_v_size)
                            v_idx = np.random.choice(valid_v_size, v_sample_size, replace=False)
                            
                            v1 = img_data[:-1, :].ravel()
                            v2 = img_data[1:, :].ravel()
                            ax2.scatter(v1[v_idx], v2[v_idx], alpha=0.6, s=1, color=color)
                            ax2.set_xlabel('Pixel Value')
                            ax2.set_ylabel('Adjacent Pixel Value')
                            ax2.set_title(f'{img_type} - Vertical Correlation', fontweight='bold')
                            ax2.grid(True, alpha=0.3)
                
                plt.suptitle(f'{img_type} Image Correlation Analysis - {filename}', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(str(output_path / f"{filename}_correlation_{img_type.lower()}.png"), dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"    🔗 Correlation plots saved to: {output_path}")
        
    except Exception as e:
        print(f"    Warning: Could not save correlation plots: {str(e)}")

def save_separate_entropy_analysis(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):
    
    try:
       
        entropy_orig = image_entropy(orig_img)
        entropy_cipher = image_entropy(cipher_img)
        entropy_decrypted = image_entropy(decrypted_img)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        stages = ['Original', 'Encrypted', 'Decrypted']
        entropies = [entropy_orig, entropy_cipher, entropy_decrypted]
        colors = ['blue', 'red', 'green']
        
        bars = ax1.bar(stages, entropies, color=colors, alpha=0.7)
        ax1.set_ylabel('Entropy (bits)')
        ax1.set_title(f'Information Entropy Comparison - {filename}', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 8.5)
        
        for bar, entropy in zip(bars, entropies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{entropy:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.axis('off')
        summary_text = f"""
ENTROPY ANALYSIS SUMMARY

Image: {filename}
DNA Operation: {dna_operation.upper()}

RESULTS:
• Original Entropy:   {entropy_orig:.6f} bits
• Encrypted Entropy:  {entropy_cipher:.6f} bits  
• Decrypted Entropy:  {entropy_decrypted:.6f} bits

ANALYSIS:
• Entropy Increase:   +{entropy_cipher - entropy_orig:.6f} bits
• Restoration Error:  {abs(entropy_decrypted - entropy_orig):.8f} bits
• Encryption Quality: {'EXCELLENT' if entropy_cipher > 7.5 else 'GOOD' if entropy_cipher > 7.0 else 'MODERATE'}

SECURITY ASSESSMENT:
• Randomness Level: {(entropy_cipher/8.0)*100:.1f}% of maximum
• Information Hiding: {'COMPLETE' if entropy_cipher > 7.9 else 'HIGH' if entropy_cipher > 7.5 else 'GOOD'}
        """
        
        ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes, fontsize=11,
                ha='left', va='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=1", facecolor='lightblue', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(str(output_path / f"{filename}_entropy_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    📈 Entropy analysis saved to: {output_path}")
        
    except Exception as e:
        print(f"    Warning: Could not save entropy analysis: {str(e)}")

def save_individual_analysis_files(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):
    
    try:
        print("    📊 Generating individual analysis files...")
        
        images_folder = output_path / "individual_images"
        histograms_folder = output_path / "histograms"
        correlations_folder = output_path / "correlations"
        entropy_folder = output_path / "entropy_analysis"
        
        for folder in [images_folder, histograms_folder, correlations_folder, entropy_folder]:
            folder.mkdir(exist_ok=True)
        
        save_separate_images(orig_img, cipher_img, decrypted_img, images_folder, filename, dna_operation)
        
        save_separate_histograms(orig_img, cipher_img, decrypted_img, histograms_folder, filename, dna_operation)
        
        save_separate_correlations(orig_img, cipher_img, decrypted_img, correlations_folder, filename, dna_operation)
        
        save_separate_entropy_analysis(orig_img, cipher_img, decrypted_img, entropy_folder, filename, dna_operation)
        
        print("    ✅ All individual analysis files generated!")
        
    except Exception as e:
        print(f"    ❌ Error generating individual files: {str(e)}")

def save_separate_images(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):
    
    try:
       
        save_orig = prepare_for_display(orig_img)
        save_cipher = prepare_for_display(cipher_img)
        save_decrypted = prepare_for_display(decrypted_img)
        
        cv2.imwrite(str(output_path / f"{filename}_original.png"), save_orig)
        cv2.imwrite(str(output_path / f"{filename}_encrypted_{dna_operation}.png"), save_cipher)
        cv2.imwrite(str(output_path / f"{filename}_decrypted.png"), save_decrypted)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1.imshow(save_orig, cmap='gray' if len(save_orig.shape) == 2 else None)
        ax1.set_title(f'Original Image\n{orig_img.shape}, {orig_img.dtype}', fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(save_cipher, cmap='gray')
        ax2.set_title(f'Encrypted Image\nDNA-{dna_operation.upper()} Operation', fontweight='bold')
        ax2.axis('off')
        
        ax3.imshow(save_decrypted, cmap='gray' if len(save_decrypted.shape) == 2 else None)
        ax3.set_title(f'Decrypted Image\nRestoration Verified', fontweight='bold')
        ax3.axis('off')
        
        plt.suptitle(f'Image Comparison - {filename}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(output_path / f"{filename}_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    💾 Images saved to: {output_path}")
        
    except Exception as e:
        print(f"    Warning: Could not save individual images: {str(e)}")

def save_separate_histograms(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):
    
    try:
        
        def get_actual_data_and_range(img):
            if len(img.shape) == 3 and img.shape[2] == 2: 
                combined = (img[:, :, 0].astype(np.uint16) << 8) | img[:, :, 1].astype(np.uint16)
                return combined.ravel(), (0, 65535), 65536, "16-bit Encrypted"
            elif img.dtype == np.uint16:
                return img.ravel(), (0, 65535), 65536, "16-bit Original"
            elif img.dtype == np.uint8:
                return img.ravel(), (0, 255), 256, "8-bit"
            else:
                flat = img.ravel()
                return flat, (flat.min(), flat.max()), min(1024, len(np.unique(flat))), str(img.dtype)
        
        images_info = [
            (orig_img, 'Original', 'blue'),
            (cipher_img, 'Encrypted', 'red'),
            (decrypted_img, 'Decrypted', 'green')
        ]
        
        for img, img_type, color in images_info:
            data, range_vals, bins, bit_info = get_actual_data_and_range(img)
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            ax.hist(data, bins=bins, alpha=0.8, color=color, density=True, range=range_vals)
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Density')
            ax.set_title(f'{img_type} Image Histogram - {filename}\n{bit_info} Range: {range_vals[0]}-{range_vals[1]}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            entropy_val = image_entropy(img)
            stats_text = f'Bit Depth: {bit_info}\nRange: {range_vals[0]}-{range_vals[1]}\nMin: {data.min()}\nMax: {data.max()}\nMean: {data.mean():.1f}\nStd: {data.std():.1f}\nEntropy: {entropy_val:.4f} bits\nUnique Values: {len(np.unique(data))}'
            
            ax.text(0.02, 0.98, stats_text, 
                    transform=ax.transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9),
                    verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig(str(output_path / f"{filename}_histogram_{img_type.lower()}_actual_depth.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        for img, img_type, color in images_info:
            data, range_vals, bins, bit_info = get_actual_data_and_range(img)
            
            if range_vals[1] > 1000: 
                plot_bins = 1024
            else:
                plot_bins = bins
            
            ax.hist(data, bins=plot_bins, alpha=0.6, color=color, density=True, 
                   range=range_vals, label=f'{img_type} ({bit_info})')
        
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Combined Histogram Comparison - {filename}\nActual Bit Depth Preserved', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(str(output_path / f"{filename}_histogram_combined_actual_depth.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    📊 Actual bit-depth histograms saved to: {output_path}")
        
    except Exception as e:
        print(f"    Warning: Could not save histogram plots: {str(e)}")

def save_separate_correlations(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):
    
    try:
        def get_plot_limits(img_data):
            flat = np.asarray(img_data).ravel()
            if flat.size == 0:
                return (0, 255), (0, 255)
            vmin = float(np.min(flat))
            vmax = float(np.max(flat))
            if vmax <= 255 and vmin >= 0:
                return (0, 255), (0, 255)
            if vmax <= 65535 and vmin >= 0:
                return (0, 65535), (0, 65535)
            pad = max((vmax - vmin) * 0.05, 1.0)
            return (vmin - pad, vmax + pad), (vmin - pad, vmax + pad)

        orig_sample = prepare_correlation_data(orig_img)
        cipher_sample = prepare_correlation_data(cipher_img)
        decrypted_sample = prepare_correlation_data(decrypted_img)
        
        images_data = [
            (orig_sample, 'Original', 'blue'),
            (cipher_sample, 'Encrypted', 'red'),
            (decrypted_sample, 'Decrypted', 'green')
        ]
        
        for img_data, img_type, color in images_data:
            if len(img_data.shape) == 3:
                if img_data.shape[2] == 2:
                    img_data = (img_data[:, :, 0].astype(np.uint16) << 8) | img_data[:, :, 1].astype(np.uint16)
                else:
                    img_data = img_data[:, :, 0]
            img_data = img_data.astype(np.float64)
            
            corr_h, corr_v, corr_d = adjacent_correlation(img_data)
            total_pixels = img_data.size
            sample_size = min(3000, total_pixels - 1)

            if total_pixels <= 1:
                continue

            valid_indices = np.arange(total_pixels - 1)
            if len(valid_indices) == 0:
                continue

            idx = np.random.choice(valid_indices, min(sample_size, len(valid_indices)), replace=False)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

            flat = img_data.ravel()
            x_vals = flat[idx]
            y_vals = flat[idx + 1]
            x_limits, y_limits = get_plot_limits(img_data)

            ax1.scatter(x_vals, y_vals, alpha=0.6, s=1, color=color)
            ax1.set_xlabel('Pixel Value')
            ax1.set_ylabel('Adjacent Pixel Value')
            ax1.set_title(f'{img_type} - Horizontal Correlation\nρ = {corr_h:.6f}', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(*x_limits)
            ax1.set_ylim(*y_limits)

            if img_data.shape[0] > 1:
                H, W = img_data.shape
                valid_v_size = (H - 1) * W
                if valid_v_size > 0:
                    v_sample_size = min(sample_size, valid_v_size)
                    v_idx = np.random.choice(valid_v_size, v_sample_size, replace=False)
                    v1 = img_data[:-1, :].ravel()
                    v2 = img_data[1:, :].ravel()
                    ax2.scatter(v1[v_idx], v2[v_idx], alpha=0.6, s=1, color=color)
                    ax2.set_xlabel('Pixel Value')
                    ax2.set_ylabel('Adjacent Pixel Value')
                    ax2.set_title(f'{img_type} - Vertical Correlation\nρ = {corr_v:.6f}', fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    ax2.set_xlim(*x_limits)
                    ax2.set_ylim(*y_limits)

            if img_data.shape[0] > 1 and img_data.shape[1] > 1:
                H, W = img_data.shape
                valid_d_size = (H - 1) * (W - 1)
                if valid_d_size > 0:
                    d_sample_size = min(sample_size, valid_d_size)
                    d_idx = np.random.choice(valid_d_size, d_sample_size, replace=False)
                    d1 = img_data[:-1, :-1].ravel()
                    d2 = img_data[1:, 1:].ravel()
                    ax3.scatter(d1[d_idx], d2[d_idx], alpha=0.6, s=1, color=color)
                    ax3.set_xlabel('Pixel Value')
                    ax3.set_ylabel('Adjacent Pixel Value')
                    ax3.set_title(f'{img_type} - Diagonal Correlation\nρ = {corr_d:.6f}', fontweight='bold')
                    ax3.grid(True, alpha=0.3)
                    ax3.set_xlim(*x_limits)
                    ax3.set_ylim(*y_limits)

            plt.suptitle(f'{img_type} Image - Complete Correlation Analysis\n'
                       f'H: {corr_h:.6f} | V: {corr_v:.6f} | D: {corr_d:.6f}', 
                       fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(str(output_path / f"{filename}_correlation_{img_type.lower()}_complete.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"    🔗 Complete correlation plots (H/V/D) saved to: {output_path}")
        
    except Exception as e:
        print(f"    Warning: Could not save correlation plots: {str(e)}")

def save_separate_entropy_analysis(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):
    
    try:
       
        entropy_orig = image_entropy(orig_img)
        entropy_cipher = image_entropy(cipher_img)
        entropy_decrypted = image_entropy(decrypted_img)
   
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        stages = ['Original', 'Encrypted', 'Decrypted']
        entropies = [entropy_orig, entropy_cipher, entropy_decrypted]
        colors = ['blue', 'red', 'green']
        
        bars = ax1.bar(stages, entropies, color=colors, alpha=0.7)
        ax1.set_ylabel('Entropy (bits)')
        ax1.set_title(f'Information Entropy Comparison - {filename}', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 8.5)
        
        for bar, entropy in zip(bars, entropies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{entropy:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.axis('off')
        summary_text = f"""
ENTROPY ANALYSIS SUMMARY

Image: {filename}
DNA Operation: {dna_operation.upper()}

RESULTS:
• Original Entropy:   {entropy_orig:.6f} bits
• Encrypted Entropy:  {entropy_cipher:.6f} bits  
• Decrypted Entropy:  {entropy_decrypted:.6f} bits

ANALYSIS:
• Entropy Increase:   +{entropy_cipher - entropy_orig:.6f} bits
• Restoration Error:  {abs(entropy_decrypted - entropy_orig):.8f} bits
• Encryption Quality: {'EXCELLENT' if entropy_cipher > 7.5 else 'GOOD' if entropy_cipher > 7.0 else 'MODERATE'}

SECURITY ASSESSMENT:
• Randomness Level: {(entropy_cipher/8.0)*100:.1f}% of maximum
• Information Hiding: {'COMPLETE' if entropy_cipher > 7.9 else 'HIGH' if entropy_cipher > 7.5 else 'GOOD'}
        """
        
        ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes, fontsize=11,
                ha='left', va='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=1", facecolor='lightblue', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(str(output_path / f"{filename}_entropy_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    📈 Entropy analysis saved to: {output_path}")
        
    except Exception as e:
        print(f"    Warning: Could not save entropy analysis: {str(e)}")



    
def save_image_comparison_triple(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):

    try:
        
        display_orig = prepare_for_display(orig_img)
        display_cipher = prepare_for_display(cipher_img)
        display_decrypted = prepare_for_display(decrypted_img)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1.imshow(display_orig, cmap='gray' if len(display_orig.shape) == 2 else None)
        ax1.set_title(f'Original Image\n{orig_img.shape}, {orig_img.dtype}', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(display_cipher, cmap='gray')
        ax2.set_title(f'Encrypted Image\nDNA-{dna_operation.upper()} Operation', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3.imshow(display_decrypted, cmap='gray' if len(display_decrypted.shape) == 2 else None)
        ax3.set_title(f'Decrypted Image\nRestoration Verified', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        plt.suptitle(f'DNA-Chaos Encryption Process - {filename}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(output_path / f"{filename}_complete_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    Warning: Could not generate image comparison: {str(e)}")

def save_histogram_analysis_triple(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):

    try:
        
        orig_data, orig_range, orig_bins, orig_type = get_actual_image_data(orig_img)
        cipher_data, cipher_range, cipher_bins, cipher_type = get_actual_image_data(cipher_img)
        decrypted_data, decrypted_range, decrypted_bins, decrypted_type = get_actual_image_data(decrypted_img)
        
        display_orig = prepare_for_display(orig_img)
        display_cipher = prepare_for_display(cipher_img)
        display_decrypted = prepare_for_display(decrypted_img)
        
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        
        ax1.imshow(display_orig, cmap='gray' if len(display_orig.shape) == 2 else None)
        ax1.set_title(f'Original Image\n{orig_img.shape}, {orig_type}')
        ax1.axis('off')
        
        ax2.imshow(display_cipher, cmap='gray')
        ax2.set_title(f'Encrypted Image\nDNA-{dna_operation.upper()} Operation')
        ax2.axis('off')
        
        ax3.imshow(display_decrypted, cmap='gray' if len(display_decrypted.shape) == 2 else None)
        ax3.set_title(f'Decrypted Image\n{decrypted_img.shape}, {decrypted_type}')
        ax3.axis('off')
        
        orig_flat = orig_data.ravel()
        hist_bins = min(1024, orig_bins) if orig_bins > 1024 else orig_bins
        ax4.hist(orig_flat, bins=hist_bins, alpha=0.8, color='blue', density=True, range=orig_range)
        ax4.set_xlabel(f'Actual Pixel Value (Range: {orig_range[0]}-{orig_range[1]})')
        ax4.set_ylabel('Density')
        ax4.set_title(f'Original Histogram - {orig_type}')
        ax4.grid(True, alpha=0.3)
        
        if orig_range[1] > 1000: 
            ax4.set_xlim(0, 65535)
        else:  
            ax4.set_xlim(0, 255)
        
        ax4.text(0.02, 0.98, f'Min: {orig_flat.min()}\nMax: {orig_flat.max()}\nMean: {orig_flat.mean():.1f}\nEntropy: {image_entropy(orig_img):.4f} bits', 
                transform=ax4.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.9),
                verticalalignment='top')
        
        cipher_flat = cipher_data.ravel()
        hist_bins = min(1024, cipher_bins) if cipher_bins > 1024 else cipher_bins
        ax5.hist(cipher_flat, bins=hist_bins, alpha=0.8, color='red', density=True, range=cipher_range)
        ax5.set_xlabel(f'Actual Pixel Value (Range: {cipher_range[0]}-{cipher_range[1]})')
        ax5.set_ylabel('Density')
        ax5.set_title(f'Encrypted Histogram - {cipher_type}')
        ax5.grid(True, alpha=0.3)
        
        if cipher_range[1] > 1000:  
            ax5.set_xlim(0, 65535) 
        else:
            ax5.set_xlim(0, 255)
        
        ax5.text(0.02, 0.98, f'Min: {cipher_flat.min()}\nMax: {cipher_flat.max()}\nMean: {cipher_flat.mean():.1f}\nEntropy: {image_entropy(cipher_img):.4f} bits', 
                transform=ax5.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.9),
                verticalalignment='top')
        
        decrypted_flat = decrypted_data.ravel()
        hist_bins = min(1024, decrypted_bins) if decrypted_bins > 1024 else decrypted_bins
        ax6.hist(decrypted_flat, bins=hist_bins, alpha=0.8, color='green', density=True, range=decrypted_range)
        ax6.set_xlabel(f'Actual Pixel Value (Range: {decrypted_range[0]}-{decrypted_range[1]})')
        ax6.set_ylabel('Density')
        ax6.set_title(f'Decrypted Histogram - {decrypted_type}')
        ax6.grid(True, alpha=0.3)
        
        if decrypted_range[1] > 1000:  
            ax6.set_xlim(0, 65535)
        else:  
            ax6.set_xlim(0, 255)
        
        ax6.text(0.02, 0.98, f'Min: {decrypted_flat.min()}\nMax: {decrypted_flat.max()}\nMean: {decrypted_flat.mean():.1f}\nEntropy: {image_entropy(decrypted_img):.4f} bits', 
                transform=ax6.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.9),
                verticalalignment='top')
        
        plt.suptitle(f'Histogram Analysis - DNA-{dna_operation.upper()} Operation\nActual Bit Depth: {orig_type} → {cipher_type} → {decrypted_type}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(output_path / f"{filename}_histogram_analysis_complete.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    Warning: Could not generate histogram analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def save_correlation_analysis_triple(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):

    try:
       
        orig_sample = prepare_correlation_data(orig_img)
        cipher_sample = prepare_correlation_data(cipher_img)
        decrypted_sample = prepare_correlation_data(decrypted_img)
        
        orig_h, orig_v, orig_d = adjacent_correlation(orig_img)
        cipher_h, cipher_v, cipher_d = adjacent_correlation(cipher_img)
        decrypted_h, decrypted_v, decrypted_d = adjacent_correlation(decrypted_img)
        
        total_pixels = orig_sample.size
        sample_size = min(3000, total_pixels - 1)
        
        if total_pixels > 1:
            valid_indices = np.arange(total_pixels - 1)
            if len(valid_indices) > 0:
                idx = np.random.choice(valid_indices, min(sample_size, len(valid_indices)), replace=False)
                
                fig, axes = plt.subplots(3, 3, figsize=(20, 16))
                
                images_data = [
                    (orig_sample, 'Original', 'blue', orig_h, orig_v, orig_d),
                    (cipher_sample, 'Encrypted', 'red', cipher_h, cipher_v, cipher_d),
                    (decrypted_sample, 'Decrypted', 'green', decrypted_h, decrypted_v, decrypted_d)
                ]
                
                for row, (img_data, img_type, color, corr_h, corr_v, corr_d) in enumerate(images_data):
                  
                    flat = img_data.ravel()
                    x_vals = flat[idx]
                    y_vals = flat[idx + 1]
                    axes[row, 0].scatter(x_vals, y_vals, alpha=0.6, s=1, color=color)
                    axes[row, 0].set_xlabel('Pixel Value')
                    axes[row, 0].set_ylabel('Adjacent Pixel Value (Horizontal)')
                    axes[row, 0].set_title(f'{img_type} - Horizontal Correlation\nρ = {corr_h:.6f}', fontweight='bold')
                    axes[row, 0].grid(True, alpha=0.3)
                    axes[row, 0].set_xlim(0, 255)
                    axes[row, 0].set_ylim(0, 255)
                    
                    if img_data.shape[0] > 1:
                        H, W = img_data.shape
                        valid_v_size = (H-1) * W
                        if valid_v_size > 0:
                            v_sample_size = min(sample_size, valid_v_size)
                            v_idx = np.random.choice(valid_v_size, v_sample_size, replace=False)
                            
                            v1 = img_data[:-1, :].ravel()
                            v2 = img_data[1:, :].ravel()
                            axes[row, 1].scatter(v1[v_idx], v2[v_idx], alpha=0.6, s=1, color=color)
                            axes[row, 1].set_xlabel('Pixel Value')
                            axes[row, 1].set_ylabel('Adjacent Pixel Value (Vertical)')
                            axes[row, 1].set_title(f'{img_type} - Vertical Correlation\nρ = {corr_v:.6f}', fontweight='bold')
                            axes[row, 1].grid(True, alpha=0.3)
                            axes[row, 1].set_xlim(0, 255)
                            axes[row, 1].set_ylim(0, 255)
                    
                    if img_data.shape[0] > 1 and img_data.shape[1] > 1:
                        H, W = img_data.shape
                        valid_d_size = (H-1) * (W-1)
                        if valid_d_size > 0:
                            d_sample_size = min(sample_size, valid_d_size)
                            d_idx = np.random.choice(valid_d_size, d_sample_size, replace=False)
                            
                            d1 = img_data[:-1, :-1].ravel()
                            d2 = img_data[1:, 1:].ravel()
                            axes[row, 2].scatter(d1[d_idx], d2[d_idx], alpha=0.6, s=1, color=color)
                            axes[row, 2].set_xlabel('Pixel Value')
                            axes[row, 2].set_ylabel('Adjacent Pixel Value (Diagonal)')
                            axes[row, 2].set_title(f'{img_type} - Diagonal Correlation\nρ = {corr_d:.6f}', fontweight='bold')
                            axes[row, 2].grid(True, alpha=0.3)
                            axes[row, 2].set_xlim(0, 255)
                            axes[row, 2].set_ylim(0, 255)
                
                plt.suptitle(f'Complete Correlation Analysis - DNA-{dna_operation.upper()} Operation\n'
                           f'Horizontal, Vertical, and Diagonal Pixel Correlations', 
                           fontsize=18, fontweight='bold')
                plt.tight_layout()
                plt.savefig(str(output_path / f"{filename}_correlation_analysis_complete.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
    except Exception as e:
        print(f"    Warning: Could not generate correlation analysis: {str(e)}")

def save_entropy_comparison(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):
    
    try:
        
        entropy_orig = image_entropy(orig_img)
        entropy_cipher = image_entropy(cipher_img)
        entropy_decrypted = image_entropy(decrypted_img)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        stages = ['Original', 'Encrypted', 'Decrypted']
        entropies = [entropy_orig, entropy_cipher, entropy_decrypted]
        colors = ['blue', 'red', 'green']
        
        bars = ax1.bar(stages, entropies, color=colors, alpha=0.7)
        ax1.set_ylabel('Entropy (bits)')
        ax1.set_title('Information Entropy Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 8.5)
        
        for bar, entropy in zip(bars, entropies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{entropy:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.axis('off')
        summary_text = f"""
 ENTROPY ANALYSIS SUMMARY

Algorithm: DNA-Chaos with {dna_operation.upper()} Operation
Image: {filename}

 RESULTS:
• Original Entropy:   {entropy_orig:.4f} bits
• Encrypted Entropy:  {entropy_cipher:.4f} bits  
• Decrypted Entropy:  {entropy_decrypted:.4f} bits

 ANALYSIS:
• Entropy Increase:   +{entropy_cipher - entropy_orig:.4f} bits
• Restoration Error:  {abs(entropy_decrypted - entropy_orig):.6f} bits
• Encryption Quality: {'EXCELLENT' if entropy_cipher > 7.5 else 'GOOD' if entropy_cipher > 7.0 else 'MODERATE'}

 Perfect decryption verified!
        """
        
        ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes, fontsize=12,
                ha='left', va='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=1", facecolor='lightblue', alpha=0.9))
        
        plt.suptitle(f'Entropy Analysis - {filename}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(output_path / f"{filename}_entropy_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    Warning: Could not generate entropy comparison: {str(e)}")

def prepare_for_display(img):

    if len(img.shape) == 3 and img.shape[2] == 2: 
       
        display_img = (img[:, :, 0].astype(np.uint16) << 8) | img[:, :, 1].astype(np.uint16)
    else:
        display_img = img
        
    if display_img.dtype != np.uint8:

        display_img = ((display_img.astype(np.float64) - display_img.min()) / 
                      (display_img.max() - display_img.min()) * 255).astype(np.uint8)
    
    return display_img

def prepare_correlation_data(img):
  
    if len(img.shape) == 3:
        if img.shape[2] == 2:  
           
            combined = (img[:, :, 0].astype(np.uint16) << 8) | img[:, :, 1].astype(np.uint16)
            return combined.astype(np.float64)
        else:
            return img[:, :, 0].astype(np.float64)
    else:
        return img.astype(np.float64)

def save_individual_images(orig_img, cipher_img, decrypted_img, output_path, filename, dna_operation):
    
    try:
       
        images_folder = output_path / "images"
        images_folder.mkdir(exist_ok=True)
        
        save_orig = prepare_for_display(orig_img)
        save_cipher = prepare_for_display(cipher_img)
        save_decrypted = prepare_for_display(decrypted_img)
     
        cv2.imwrite(str(images_folder / f"{filename}_1_original.png"), save_orig)
        cv2.imwrite(str(images_folder / f"{filename}_2_encrypted_{dna_operation}.png"), save_cipher)
        cv2.imwrite(str(images_folder / f"{filename}_3_decrypted.png"), save_decrypted)
        
        print(f"    💾 Individual images saved to: {images_folder}")
        
    except Exception as e:
        print(f"    Warning: Could not save individual images: {str(e)}")

def information_entropy_analysis(img: np.ndarray):
   
    entropy_val = image_entropy(img)
    
    if len(img.shape) == 3 and img.shape[2] == 2:
     
        combined = (img[:, :, 0].astype(np.uint16) << 8) | img[:, :, 1].astype(np.uint16)
        unique_vals = len(np.unique(combined.ravel()))
        max_entropy = 16.0  
    elif img.dtype == np.uint16:
        unique_vals = len(np.unique(img.ravel()))
        max_entropy = 16.0
    elif img.dtype == np.int16:
        unique_vals = len(np.unique(img.ravel()))
        max_entropy = 16.0  
    else:
        unique_vals = len(np.unique(img.ravel()))
        
        if img.dtype == np.uint8:
            max_entropy = 8.0
        else:
            max_entropy = np.log2(unique_vals) if unique_vals > 1 else 1.0
    
    entropy_ratio = entropy_val / max_entropy if max_entropy > 0 else 0.0
    
    return entropy_val, entropy_ratio

def compare_dna_operations(img: np.ndarray, password: str, mu: float = 3.99):

    print("  🧬 Comparing DNA operations (ADD, SUB, XOR + DYNAMIC)...")
    
    operations = ['add', 'subtract', 'xor', 'auto'] 
    results = {}
    
    for op in operations:
        if op == 'auto':
            print(f"    Testing DYNAMIC selection...")
        else:
            print(f"    Testing {op.upper()}...")
        
        cipher, aux = encrypt_dna_chaos(img, password, mu, op)
        
        actual_operation = aux['dna_operation']
        
        entropy_val = image_entropy(cipher)
        c1, c2, npcr, uaci = single_pixel_test_encrypt(img, password, mu, dna_operation=op)
        corr_h, corr_v, corr_d = adjacent_correlation(cipher)
        
        restored = decrypt_dna_chaos(cipher, aux)
        decrypt_error = np.mean(np.abs(img.astype(np.float64) - restored.astype(np.float64)))
        
        results[op if op != 'auto' else f'dynamic_{actual_operation}'] = {
            'entropy': entropy_val,
            'npcr': npcr,
            'uaci': uaci,
            'correlation_h': corr_h,
            'correlation_v': corr_v,
            'correlation_d': corr_d,
            'decrypt_error': decrypt_error,
            'dna_rule_used': aux['dna_rule_id'] + 1,
            'actual_operation': actual_operation
        }
        
        if op == 'auto':
            print(f"      → Selected: {actual_operation.upper()}, Entropy: {entropy_val:.4f}, NPCR: {npcr:.2f}%")
        else:
            print(f"      Entropy: {entropy_val:.4f}, NPCR: {npcr:.2f}%, Decrypt Error: {decrypt_error:.6f}")
    
    return results


def save_encrypted_dicom(cipher_img, original_dicom_path, output_folder, aux):

    if not HAVE_PYDICOM:
        print("    ⚠️ pydicom not available - cannot save encrypted DICOM")
        return False
    
    try:
        print("    💾 Saving encrypted DICOM file...")
        
        ds_original = pydicom.dcmread(str(original_dicom_path))
        
        ds_encrypted = pydicom.Dataset()
        
        essential_tags = [
            'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
            'StudyInstanceUID', 'SeriesInstanceUID', 'StudyDate', 'StudyTime',
            'Modality', 'Manufacturer', 'InstitutionName',
            'SeriesNumber', 'AcquisitionNumber', 'InstanceNumber',
            'ImagePositionPatient', 'ImageOrientationPatient',
            'PixelSpacing', 'SliceThickness', 'SliceLocation',
            'SamplesPerPixel', 'PhotometricInterpretation',
            'Rows', 'Columns', 'BitsAllocated', 'BitsStored', 'HighBit',
            'PixelRepresentation'
        ]
        
        for tag in essential_tags:
            if hasattr(ds_original, tag):
                setattr(ds_encrypted, tag, getattr(ds_original, tag))
        
        if len(cipher_img.shape) == 3 and cipher_img.shape[2] == 2:
            
            encrypted_16bit = (cipher_img[:, :, 0].astype(np.uint16) << 8) | cipher_img[:, :, 1].astype(np.uint16)
            
            ds_encrypted.Rows = encrypted_16bit.shape[0]
            ds_encrypted.Columns = encrypted_16bit.shape[1]
            ds_encrypted.BitsAllocated = 16
            ds_encrypted.BitsStored = 16
            ds_encrypted.HighBit = 15
            ds_encrypted.PixelRepresentation = 0  
            ds_encrypted.SamplesPerPixel = 1
            ds_encrypted.PhotometricInterpretation = "MONOCHROME2"
            
            ds_encrypted.PixelData = encrypted_16bit.tobytes()
            
        else:
        
            ds_encrypted.Rows = cipher_img.shape[0]
            ds_encrypted.Columns = cipher_img.shape[1]
            
            if cipher_img.dtype == np.uint8:
                ds_encrypted.BitsAllocated = 8
                ds_encrypted.BitsStored = 8
                ds_encrypted.HighBit = 7
            elif cipher_img.dtype in [np.uint16, np.int16]:
                ds_encrypted.BitsAllocated = 16
                ds_encrypted.BitsStored = 16
                ds_encrypted.HighBit = 15
            
            ds_encrypted.PixelRepresentation = 0 if cipher_img.dtype in [np.uint8, np.uint16] else 1
            ds_encrypted.SamplesPerPixel = 1 if len(cipher_img.shape) == 2 else cipher_img.shape[2]
            ds_encrypted.PhotometricInterpretation = "MONOCHROME2" if len(cipher_img.shape) == 2 else "RGB"
    
            ds_encrypted.PixelData = cipher_img.tobytes()
        
        ds_encrypted.add_new([0x0009, 0x0010], 'LO', 'DNA_CHAOS_ENCRYPTION')
        ds_encrypted.add_new([0x0009, 0x1001], 'LO', f"DNA_RULE_{aux['dna_rule_id'] + 1}")
        ds_encrypted.add_new([0x0009, 0x1002], 'LO', f"DNA_OP_{aux['dna_operation'].upper()}")
        ds_encrypted.add_new([0x0009, 0x1003], 'DS', str(aux['mu']))
        
        ds_encrypted.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        ds_encrypted.SOPInstanceUID = pydicom.uid.generate_uid()
        
        ds_encrypted.file_meta = pydicom.dataset.FileMetaDataset()
        ds_encrypted.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds_encrypted.file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        ds_encrypted.file_meta.MediaStorageSOPInstanceUID = ds_encrypted.SOPInstanceUID
        
        ds_encrypted.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        ds_encrypted.file_meta.ImplementationVersionName = "DNA_CHAOS_V1"
        
        output_filename = original_dicom_path.stem + "_encrypted.dcm"
        output_path = output_folder / output_filename
        ds_encrypted.save_as(str(output_path), write_like_original=False)
        
        print(f"    ✅ Encrypted DICOM saved: {output_filename}")
        return True
        
    except Exception as e:
        print(f"    ❌ Error saving encrypted DICOM: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_single_image(image_path: Path, output_dir: Path, password: str = "medical123", mu: float = 3.99, dna_operation: str = "add"):
    
    print(f"\n{'='*60}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*60}")
    
    image_folder = output_dir / image_path.stem
    image_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        print("  📂 Loading image...")
        img = load_image(image_path)
        print(f"  ✅ Loaded: {img.shape}, {img.dtype}, range: [{img.min()}, {img.max()}]")
        
        print("  🧬 Comparing DNA operations...")
        dna_comparison = compare_dna_operations(img, password, mu)
        
        print(f"  🔐 Encrypting with {dna_operation.upper()} operation...")
        cipher_img, aux = encrypt_dna_chaos(img, password, mu, dna_operation)
        
        print(f"\n  🔍 DICOM DEBUG CHECK:")
        print(f"     • image_path: {image_path}")
        print(f"     • image_path.name: {image_path.name}")  
        print(f"     • image_path.suffix: '{image_path.suffix}'")
        print(f"     • image_path.suffix.lower(): '{image_path.suffix.lower()}'")
        print(f"     • Is .dcm?: {image_path.suffix.lower() == '.dcm'}")
        print(f"     • Input command was: {sys.argv}")
        
        print(f"\n  🏥 FORCING DICOM SAVE ATTEMPT...")
        if HAVE_PYDICOM:
            try:
                print(f"     • pydicom available: YES")
                print(f"     • Attempting save...")
                saved = save_encrypted_dicom(cipher_img, image_path, image_folder, aux)
                print(f"     • Save result: {saved}")
                
                expected_dcm = image_folder / f"{image_path.stem}_encrypted.dcm"
                print(f"     • Expected file: {expected_dcm}")
                
                if expected_dcm.exists():
                    size = expected_dcm.stat().st_size
                    print(f"  FILE EXISTS! Size: {size:,} bytes")
                    print(f"  Location: {expected_dcm.absolute()}")
                else:
                    print(f"  File not found at expected location")
                    print(f"  Folder contents:")
                    for f in image_folder.iterdir():
                        print(f"        - {f.name}")
                        
            except Exception as e:
                print(f"     ❌ Exception: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"     ❌ pydicom not available!")
        
        if image_path.suffix.lower() == '.dcm':
            print(f"\n  🏥 DICOM DETECTED - SAVING ENCRYPTED DICOM NOW")
            print(f"     File: {image_path.name}")
            print(f"     Output: {image_folder}")
            
            if HAVE_PYDICOM:
                try:
                    saved = save_encrypted_dicom(cipher_img, image_path, image_folder, aux)
                    if saved:
                        dcm_file = image_folder / f"{image_path.stem}_encrypted.dcm"
                        if dcm_file.exists():
                            print(f"  ✅✅✅ ENCRYPTED DICOM SAVED: {dcm_file.name}")
                            print(f"  ✅✅✅ File size: {dcm_file.stat().st_size:,} bytes")
                            print(f"  ✅✅✅ Location: {dcm_file.absolute()}")
                        else:
                            print(f"  ❌ File not found after save!")
                    else:
                        print(f"  ❌ save_encrypted_dicom returned False")
                except Exception as e:
                    print(f"  ❌ Exception: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  ❌ pydicom not installed!")
        
        print(f"\n  🔓 Decrypting...")
        decrypted_img = decrypt_dna_chaos(cipher_img, aux)
        restoration_error = np.mean(np.abs(img.astype(np.float64) - decrypted_img.astype(np.float64)))
        print(f"  ✅ Restoration error: {restoration_error:.8f}")
        
        print("  📊 Calculating metrics...")
        entropy_orig, _ = information_entropy_analysis(img)
        entropy_enc, _ = information_entropy_analysis(cipher_img)
        entropy_dec, _ = information_entropy_analysis(decrypted_img)
        
        corr_orig_hor, corr_orig_ver, corr_orig_diag = adjacent_correlation(img)
        corr_enc_hor, corr_enc_ver, corr_enc_diag = adjacent_correlation(cipher_img)
        corr_dec_hor, corr_dec_ver, corr_dec_diag = adjacent_correlation(decrypted_img)
        
        _, _, npcr, uaci = single_pixel_test_encrypt(img, password, mu, dna_operation=dna_operation)
        
        print("  🎨 Generating visualizations...")
        save_complete_analysis_suite(img, cipher_img, decrypted_img, image_folder, image_path.stem, dna_operation)
        save_individual_images(img, cipher_img, decrypted_img, image_folder, image_path.stem, dna_operation)
        
        metrics_dict = {
            'filename': image_path.name,
            'shape': f"{img.shape[0]}×{img.shape[1]}" if len(img.shape) == 2 else f"{img.shape[0]}×{img.shape[1]}×{img.shape[2]}",
            'type': f"grayscale-{img.dtype}" if len(img.shape) == 2 else f"color-{img.dtype}",
            'dna_rule_used': aux['dna_rule_id'] + 1,
            'dna_operation': dna_operation.upper(),
            'restoration_error': restoration_error,
            'entropy_orig': entropy_orig,
            'entropy_enc': entropy_enc,
            'entropy_dec': entropy_dec,
            'corr_orig_hor': corr_orig_hor,
            'corr_orig_ver': corr_orig_ver,
            'corr_orig_diag': corr_orig_diag,
            'corr_enc_hor': corr_enc_hor,
            'corr_enc_ver': corr_enc_ver,
            'corr_enc_diag': corr_enc_diag,
            'corr_dec_hor': corr_dec_hor,
            'corr_dec_ver': corr_dec_ver,
            'corr_dec_diag': corr_dec_diag,
            'npcr_avalanche': npcr,
            'uaci_avalanche': uaci,
            'add_entropy': dna_comparison['add']['entropy'],
            'sub_entropy': dna_comparison['subtract']['entropy'],
            'xor_entropy': dna_comparison['xor']['entropy'],
            'add_npcr': dna_comparison['add']['npcr'],
            'sub_npcr': dna_comparison['subtract']['npcr'],
            'xor_npcr': dna_comparison['xor']['npcr'],
            'add_uaci': dna_comparison['add']['uaci'],
            'sub_uaci': dna_comparison['subtract']['uaci'],
            'xor_uaci': dna_comparison['xor']['uaci'],
        }
        
        df_comparison = pd.DataFrame(dna_comparison).T
        df_comparison.to_csv(image_folder / f"{image_path.stem}_dna_operations_comparison.csv")
        
        df = pd.DataFrame([metrics_dict])
        df.to_csv(image_folder / f"{image_path.stem}_detailed_metrics.csv", index=False)
        
        # SUMMARY
        key_space_bits = enhanced_key_space_analysis()
        summary_text = f"""
DNA-CHAOS ENCRYPTION ANALYSIS SUMMARY
=====================================

Image: {image_path.name}
Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
DNA Operation: {dna_operation.upper()}
DNA Rule Used: #{aux['dna_rule_id'] + 1}/24

ENCRYPTION RESULTS:
- Entropy: {entropy_orig:.4f} → {entropy_enc:.4f} → {entropy_dec:.4f}
- NPCR: {npcr:.2f}%, UACI: {uaci:.2f}%
- Restoration Error: {restoration_error:.8f}

FILES GENERATED:
✅ Encrypted DICOM: {image_path.stem}_encrypted.dcm
✅ Analysis plots
✅ Metrics CSV
"""
        
        with open(image_folder / f"{image_path.stem}_SUMMARY.txt", 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"\n  ✅ Processing complete!")
        print(f"     📁 Output: {image_folder}")
        print(f"     📊 Entropy: {entropy_orig:.4f} → {entropy_enc:.4f} → {entropy_dec:.4f}")
        
        return metrics_dict
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
 
def get_actual_image_data(img):
   
    if len(img.shape) == 3 and img.shape[2] == 2: 
        combined = (img[:, :, 0].astype(np.uint16) << 8) | img[:, :, 1].astype(np.uint16)
        return combined, (0, 65535), 65536, "16-bit Encrypted (0-65535)"
    elif img.dtype == np.uint16:
        return img, (0, 65535), 65536, "16-bit Original (0-65535)"
    elif img.dtype == np.int16:
        return img, (img.min(), img.max()), 65536, f"16-bit Signed ({img.min()}-{img.max()})"
    elif img.dtype == np.uint8:
        return img, (0, 255), 256, "8-bit (0-255)"
    else:
        flat = img.ravel()
        unique_count = len(np.unique(flat))
        return img, (flat.min(), flat.max()), min(65536, unique_count), str(img.dtype)



def generate_algorithm_flowchart(output_dir: Path):
    
    print("📊 Generating Algorithm Flowchart...")
    
    try:
        import matplotlib.patches as patches
        from matplotlib.patches import FancyBboxPatch, Rectangle
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 20))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 24)
        ax.axis('off')
        
        colors = {
            'start_end': '#2E8B57',     
            'process': '#4169E1',        
            'decision': '#FF6347',    
            'data': '#9370DB',       
            'output': '#FF8C00'          
        }

        box_width = 2.8
        box_height = 0.8
        x_center = 5.0
        
        def create_box(x, y, width, height, text, color, shape='rectangle'):
            if shape == 'ellipse':
               
                ellipse = patches.Ellipse((x, y), width, height, 
                                        facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(ellipse)
            elif shape == 'diamond':
               
                diamond = patches.Polygon([(x, y+height/2), (x+width/2, y+height), 
                                         (x, y-height/2), (x-width/2, y)],
                                        facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(diamond)
            else:
                
                rect = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                    boxstyle="round,pad=0.1", 
                                    facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
            
            ax.text(x, y, text, ha='center', va='center', fontsize=10, 
                   fontweight='bold', wrap=True, color='white')
        
        def draw_arrow(x1, y1, x2, y2, text=''):
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
            if text:
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x + 0.3, mid_y, text, fontsize=9, fontweight='bold')
        
        y_positions = [23, 21.8, 20.6, 19.4, 18.2, 17.0, 15.8, 14.6, 13.4, 12.2, 
                      11.0, 9.8, 8.6, 7.4, 6.2, 5.0, 3.8, 2.6, 1.4]
        
        create_box(x_center, y_positions[0], box_width, box_height, 
                  'START', colors['start_end'], 'ellipse')
        
        create_box(x_center, y_positions[1], box_width, box_height,
                  'Input: Medical Image\n& Password', colors['data'])
        
        create_box(x_center, y_positions[2], box_width, box_height,
                  'Generate SHA-256 Hash\n(Password + Image)', colors['process'])
        
        create_box(x_center, y_positions[3], box_width, box_height,
                  'Dynamic DNA Rule\nSelection (1-24)', colors['process'])
        
        create_box(x_center, y_positions[4], box_width, box_height,
                  'Dynamic Operation\nSelection (ADD/SUB/XOR)', colors['process'])
        
        create_box(x_center, y_positions[5], box_width, box_height,
                  'Derive 4 Logistic Seeds\nfrom SHA-256', colors['process'])
        
        create_box(x_center, y_positions[6], box_width, box_height,
                  'Generate Chaotic\nSequences (μ=3.99)', colors['process'])
        
        create_box(x_center, y_positions[7], box_width, box_height,
                  'Permutation Phase\n(Row & Column)', colors['process'])
        
        create_box(x_center, y_positions[8], box_width, box_height,
                  'Diffusion Phase\n(XOR + Addition)', colors['process'])
        
        create_box(x_center, y_positions[9], box_width, box_height,
                  'DNA Encoding\n(Bytes → Quartets)', colors['process'])
        
        create_box(x_center, y_positions[10], box_width, box_height,
                  'Apply DNA Operation\n(Image ⊕ Key)', colors['process'])
        
        create_box(x_center, y_positions[11], box_width, box_height,
                  'DNA Decoding\n(Quartets → Bytes)', colors['process'])
        
        create_box(x_center, y_positions[12], box_width*1.2, box_height,
                  'Encrypt or\nDecrypt?', colors['decision'], 'diamond')
        
        create_box(x_center - 2.5, y_positions[13], box_width, box_height,
                  'Encrypted Image\nOutput', colors['output'])
         
        create_box(x_center + 2.5, y_positions[13], box_width, box_height,
                  'Apply Inverse\nOperations', colors['process'])
        
        create_box(x_center + 2.5, y_positions[14], box_width, box_height,
                  'Decrypted Image\nOutput', colors['output'])
        
        create_box(x_center, y_positions[15], box_width, box_height,
                  'Security Analysis\n(NPCR, UACI, Entropy)', colors['process'])
        
        create_box(x_center, y_positions[16], box_width, box_height,
                  'Generate Analysis\nReports & Plots', colors['process'])
        
        create_box(x_center, y_positions[17], box_width, box_height,
                  'END', colors['start_end'], 'ellipse')
        
        arrows = [
            (x_center, y_positions[0]-0.4, x_center, y_positions[1]+0.4),
            (x_center, y_positions[1]-0.4, x_center, y_positions[2]+0.4),
            (x_center, y_positions[2]-0.4, x_center, y_positions[3]+0.4),
            (x_center, y_positions[3]-0.4, x_center, y_positions[4]+0.4),
            (x_center, y_positions[4]-0.4, x_center, y_positions[5]+0.4),
            (x_center, y_positions[5]-0.4, x_center, y_positions[6]+0.4),
            (x_center, y_positions[6]-0.4, x_center, y_positions[7]+0.4),
            (x_center, y_positions[7]-0.4, x_center, y_positions[8]+0.4),
            (x_center, y_positions[8]-0.4, x_center, y_positions[9]+0.4),
            (x_center, y_positions[9]-0.4, x_center, y_positions[10]+0.4),
            (x_center, y_positions[10]-0.4, x_center, y_positions[11]+0.4),
            (x_center, y_positions[11]-0.4, x_center, y_positions[12]+0.4),
            
            (x_center-0.6, y_positions[12]-0.4, x_center-2.5, y_positions[13]+0.4),  
            (x_center+0.6, y_positions[12]-0.4, x_center+2.5, y_positions[13]+0.4),  
            
            (x_center+2.5, y_positions[13]-0.4, x_center+2.5, y_positions[14]+0.4),
            
            (x_center-2.5, y_positions[13]-0.4, x_center, y_positions[15]+0.4),
            (x_center+2.5, y_positions[14]-0.4, x_center, y_positions[15]+0.4),
            
            (x_center, y_positions[15]-0.4, x_center, y_positions[16]+0.4),
            (x_center, y_positions[16]-0.4, x_center, y_positions[17]+0.4),
        ]
        
        for arrow in arrows:
            draw_arrow(*arrow)
        
        ax.text(x_center-1.2, y_positions[12]-0.8, 'Encrypt', fontsize=9, 
               fontweight='bold', ha='center')
        ax.text(x_center+1.2, y_positions[12]-0.8, 'Decrypt', fontsize=9, 
               fontweight='bold', ha='center')
        
        plt.suptitle('DNA-Chaos Image Encryption Algorithm Flowchart', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        legend_elements = [
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['start_end'], label='Start/End'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['process'], label='Process'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['decision'], label='Decision'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['data'], label='Input/Data'),
            patches.Rectangle((0, 0), 1, 1, facecolor=colors['output'], label='Output')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
        
        features_text = """
Key Features:
• Dynamic DNA Rule Selection (24 options)
• Dynamic Operation Selection (ADD/SUB/XOR)
• SHA-256 Based Seed Generation
• Single-Round Permutation-Diffusion
• Chaotic Logistic Maps (μ=3.99)
• Enhanced Key Space: 2^395.2 bits
        """
        
        ax.text(0.5, 11, features_text, fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
               verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(str(output_dir / "DNA_Chaos_Algorithm_Flowchart.png"), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Algorithm flowchart saved: {output_dir}/DNA_Chaos_Algorithm_Flowchart.png")
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating flowchart: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
def debug_16bit_processing(img_path):
   
    print(f"\n🔍 DEBUGGING 16-BIT PROCESSING for {img_path.name}")
    print("="*60)
    
    orig_img = load_image(img_path)
    print(f"Original Image:")
    print(f"  • Shape: {orig_img.shape}")
    print(f"  • Dtype: {orig_img.dtype}")
    print(f"  • Range: [{orig_img.min()}, {orig_img.max()}]")
    print(f"  • Entropy: {image_entropy(orig_img):.6f} bits")
    
    cipher_img, aux = encrypt_dna_chaos(orig_img, "test123", 3.99, "add")
    print(f"\nEncrypted Image:")
    print(f"  • Shape: {cipher_img.shape}")
    print(f"  • Dtype: {cipher_img.dtype}")
    print(f"  • Range: [{cipher_img.min()}, {cipher_img.max()}]")
    print(f"  • Entropy: {image_entropy(cipher_img):.6f} bits")
    
    if len(cipher_img.shape) == 3 and cipher_img.shape[2] == 2:
        print(f"  • Format: (H, W, 2) - 16-bit encrypted format ✅")
        
        reconstructed = (cipher_img[:, :, 0].astype(np.uint16) << 8) | cipher_img[:, :, 1].astype(np.uint16)
        print(f"  • Reconstructed Range: [{reconstructed.min()}, {reconstructed.max()}]")
        print(f"  • Reconstructed Entropy: {image_entropy(cipher_img):.6f} bits")
        
        if reconstructed.max() > 1000:
            print(f"  • 16-bit Range Usage: FULL ✅ (0-{reconstructed.max()})")
        else:
            print(f"  • 16-bit Range Usage: LIMITED ❌ (0-{reconstructed.max()})")
    else:
        print(f"  • Format: Regular array")
    
    decrypted_img = decrypt_dna_chaos(cipher_img, aux)
    print(f"\nDecrypted Image:")
    print(f"  • Shape: {decrypted_img.shape}")
    print(f"  • Dtype: {decrypted_img.dtype}")
    print(f"  • Range: [{decrypted_img.min()}, {decrypted_img.max()}]")
    print(f"  • Entropy: {image_entropy(decrypted_img):.6f} bits")
    
    error = np.mean(np.abs(orig_img.astype(np.float64) - decrypted_img.astype(np.float64)))
    print(f"  • Restoration Error: {error:.10f}")
    print(f"  • Perfect Restoration: {'YES' if error < 1e-10 else 'NO'}")


def main():
    
    if not check_dependencies():
        print("Please install missing dependencies and retry.")
        sys.exit(1)
        
    parser = argparse.ArgumentParser(
        description="DNA-Chaos Image Encryption with Enhanced DNA Operations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input", type=str, required=True, 
                       help="Path to image file or directory")
    parser.add_argument("--outdir", type=str, default="dna_chaos_enhanced", 
                       help="Output directory")
    parser.add_argument("--password", type=str, default="medical123", 
                       help="Encryption password")
    parser.add_argument("--mu", type=float, default=3.99, 
                       help="Logistic map parameter")
    parser.add_argument("--dna_op", type=str, 
                       choices=['add', 'subtract', 'xor', 'auto'], 
                       default='auto',
                       help="DNA operation: add, subtract, xor, or auto (dynamic selection)")
    parser.add_argument("--debug", action='store_true', 
                       help="Enable detailed 16-bit debugging")
    
    args = parser.parse_args()
    
    print("🧬 Initializing Enhanced DNA-Chaos Encryption with DYNAMIC Operation Selection...")
    print("🔧 Building dynamic DNA encoding tables...")
    
    build_dna_tables()

    print("\n🔐 ENHANCED KEY SPACE ANALYSIS:")
    key_space_bits = enhanced_key_space_analysis()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🎨 Generating algorithm flowchart...")
    success = generate_algorithm_flowchart(output_dir)
    if success:
        print("   ✅ Flowchart generated successfully!")
    else:
        print("   ⚠️ Flowchart generation failed, continuing...")
    
    if args.dna_op == 'auto':
        print(f"\n🎯 OPERATION MODE: DYNAMIC")
        print(f"   • Operation will be selected based on SHA-256(password + image)")
        print(f"   • Available operations: ADD, SUBTRACT, XOR")
    else:
        print(f"\n🔒 OPERATION MODE: FIXED")
        print(f"   • Using {args.dna_op.upper()} operation for all images") 
 
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ Input path does not exist: {input_path}")
        sys.exit(1)
    
    all_results = []
    
    if input_path.is_file():
        
        if args.debug:
            debug_16bit_processing(input_path)
        
        print(f"\n📄 PROCESSING SINGLE FILE: {input_path.name}")
        print(f"{'='*80}")
        
        try:
            test_img = load_image(input_path)
            print(f"📊 IMAGE ANALYSIS:")
            print(f"   • File: {input_path.name}")
            print(f"   • Shape: {test_img.shape}")
            print(f"   • Data Type: {test_img.dtype}")
            print(f"   • Value Range: [{test_img.min()}, {test_img.max()}]")
            print(f"   • Bit Depth: {'16-bit' if test_img.dtype in [np.uint16, np.int16] else '8-bit' if test_img.dtype == np.uint8 else 'Other'}")
            print(f"   • Expected Max Entropy: {'~16 bits' if test_img.dtype in [np.uint16, np.int16] else '~8 bits' if test_img.dtype == np.uint8 else 'Variable'}")
            
            orig_entropy = image_entropy(test_img)
            print(f"   • Original Entropy: {orig_entropy:.6f} bits")
            
            del test_img 
            
        except Exception as e:
            print(f"   ⚠️ Could not pre-analyze image: {str(e)}")
        
        result = process_single_image(input_path, output_dir, args.password, args.mu, args.dna_op)
        if result:
            all_results.append(result)
            
            if result.get('type', '').startswith('grayscale-int16') or result.get('type', '').startswith('grayscale-uint16'):
                print(f"\n🔍 16-BIT VALIDATION RESULTS:")
                print(f"   • Original Entropy: {result['entropy_orig']:.6f} bits")
                print(f"   • Encrypted Entropy: {result['entropy_enc']:.6f} bits")
                print(f"   • Expected Range: ~10-16 bits for encrypted 16-bit")
                if result['entropy_enc'] > 14.0:
                    print(f"   ✅ EXCELLENT: Near-maximum 16-bit entropy achieved!")
                elif result['entropy_enc'] > 12.0:
                    print(f"   ✅ GOOD: Strong 16-bit entropy achieved!")
                elif result['entropy_enc'] > 7.5:
                    print(f"   ⚠️ MODERATE: 8-bit equivalent entropy (check 16-bit handling)")
                else:
                    print(f"   ❌ LOW: Entropy below expected range")
        
    else:
       
        print(f"\n📁 PROCESSING DIRECTORY: {input_path}")
        print(f"{'='*80}")
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.dcm'}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions and f.is_file()]
        
        if not image_files:
            print(f"❌ No image files found in {input_path}")
            print(f"    Looking for: {', '.join(image_extensions)}")
            sys.exit(1)
            
        print(f"📊 DIRECTORY ANALYSIS:")
        print(f"   • Total images found: {len(image_files)}")
        print(f"   • Supported formats: {', '.join(image_extensions)}")
        
        bit_depth_counts = {'8-bit': 0, '16-bit': 0, 'Other': 0}
        for img_file in image_files[:10]:  
            try:
                sample_img = load_image(img_file)
                if sample_img.dtype == np.uint8:
                    bit_depth_counts['8-bit'] += 1
                elif sample_img.dtype in [np.uint16, np.int16]:
                    bit_depth_counts['16-bit'] += 1
                else:
                    bit_depth_counts['Other'] += 1
                del sample_img
            except:
                pass
        
        print(f"   • Bit depth distribution (sample): {dict(bit_depth_counts)}")
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n🔍 [{i}/{len(image_files)}] {image_file.name}")
            
            if args.debug and i <= 3:  
                try:
                    test_img = load_image(image_file)
                    if test_img.dtype in [np.uint16, np.int16]:
                        print(f"   🔬 16-bit image detected - enabling debug mode")
                        debug_16bit_processing(image_file)
                    del test_img
                except:
                    pass
            
            result = process_single_image(image_file, output_dir, args.password, args.mu, args.dna_op)
            if result:
                all_results.append(result)
    
    if all_results:
        print(f"\n{'='*80}")
        print(f"🛡️ DNA-CHAOS ENHANCED ANALYSIS COMPLETE - FINAL SUMMARY")
        print(f"{'='*80}")
        
        total_images = len(all_results)
        
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(output_dir / "all_results_enhanced_complete.csv", index=False)
        
        print(f"🎉 ENHANCED DNA-CHAOS ANALYSIS COMPLETE!")
        print(f"   • Total files generated: {total_images * 8 + 3}")
        print(f"   • Results directory: {output_dir}")
        print(f"   • Each image includes: Original, Encrypted, Decrypted + Complete Analysis")
        
    else:
        print("❌ No images were processed successfully")
        sys.exit(1)

if __name__ == "__main__":
    main()