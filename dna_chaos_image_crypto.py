import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Chaos: logistic map
# ---------------------------
def logistic_map(mu, x0, n, discard=1000):
    x = x0
    # burn-in
    for _ in range(discard):
        x = mu * x * (1 - x)
    seq = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = mu * x * (1 - x)
        seq[i] = x
    return seq

def chaos_bytes(mu, x0, n, discard=1000):
    seq = logistic_map(mu, x0, n, discard)
    # Map to 0..255
    return np.floor(seq * 256.0).astype(np.uint8)

# ---------------------------
# Permutation using chaos (rows & cols)
# ---------------------------
def permute_image(img, row_seq, col_seq):
    H, W = img.shape[:2]
    row_idx = np.argsort(row_seq[:H])
    col_idx = np.argsort(col_seq[:W])
    if img.ndim == 2:
        out = img[row_idx][:, col_idx]
    else:
        out = img[row_idx][:, col_idx, :]
    return out, row_idx, col_idx

def inv_permute_image(img, row_idx, col_idx):
    H, W = img.shape[:2]
    inv_rows = np.empty_like(row_idx)
    inv_rows[row_idx] = np.arange(len(row_idx))
    inv_cols = np.empty_like(col_idx)
    inv_cols[col_idx] = np.arange(len(col_idx))
    if img.ndim == 2:
        return img[inv_rows][:, inv_cols]
    else:
        return img[inv_rows][:, inv_cols, :]

# ---------------------------
# Diffusion (forward + inverse)
# ---------------------------
def diffuse(img_u8, ks_bytes):
    flat = img_u8.reshape(-1).copy()
    N = flat.size
    # Ensure ks has enough length
    ks = ks_bytes.reshape(-1)
    if ks.size < N:
        ks = np.resize(ks, N)
    # Forward XOR chain
    for i in range(2, N):
        flat[i] = flat[i] ^ flat[i-1] ^ flat[i-2] ^ ks[i]
    return flat.reshape(img_u8.shape)

def inv_diffuse(img_u8, ks_bytes):
    flat = img_u8.reshape(-1).copy()
    N = flat.size
    ks = ks_bytes.reshape(-1)
    if ks.size < N:
        ks = np.resize(ks, N)
    # Reverse XOR chain to undo
    for i in range(N-1, 1, -1):
        flat[i] = flat[i] ^ flat[i-1] ^ flat[i-2] ^ ks[i]
    return flat.reshape(img_u8.shape)

# ---------------------------
# DNA rules (8 bijections preserving complements)
# Encode 2-bit value v in {0,1,2,3} to a base in {A,C,G,T}~{0,1,2,3}
# We'll represent bases as ints: A=0, C=1, G=2, T=3.
# ---------------------------
A, C, G, T = 0, 1, 2, 3
RULES = np.array([
    [A, C, G, T],  # 0
    [A, G, C, T],  # 1
    [C, A, T, G],  # 2
    [C, T, A, G],  # 3
    [G, A, T, C],  # 4
    [G, T, A, C],  # 5
    [T, C, G, A],  # 6
    [T, G, C, A],  # 7
], dtype=np.uint8)

# Inverse maps: base -> 2-bit value under rule r
INV_RULES = np.empty_like(RULES)
for r in range(8):
    inv = np.empty(4, dtype=np.uint8)
    for v in range(4):
        inv[RULES[r, v]] = v
    INV_RULES[r] = inv

def bytes_to_quartets_u2(img_bytes):
    """
    Convert uint8 array of shape (N,) into 2-bit quartets (N,4) with values 0..3
    [b7 b6 | b5 b4 | b3 b2 | b1 b0] -> 4 values (msb pair first)
    """
    bits = np.unpackbits(img_bytes, bitorder='big').reshape(-1, 8)
    q0 = bits[:, 0]*2 + bits[:, 1]
    q1 = bits[:, 2]*2 + bits[:, 3]
    q2 = bits[:, 4]*2 + bits[:, 5]
    q3 = bits[:, 6]*2 + bits[:, 7]
    return np.stack([q0, q1, q2, q3], axis=1).astype(np.uint8)

def quartets_u2_to_bytes(quartets):
    """
    Inverse of bytes_to_quartets_u2. quartets shape (N,4), values 0..3
    """
    q0, q1, q2, q3 = quartets[:,0], quartets[:,1], quartets[:,2], quartets[:,3]
    bits = np.stack([
        (q0>>1)&1, q0&1,
        (q1>>1)&1, q1&1,
        (q2>>1)&1, q2&1,
        (q3>>1)&1, q3&1
    ], axis=1).astype(np.uint8)
    return np.packbits(bits, bitorder='big').astype(np.uint8)

def dna_encode_quartets(quartets_u2, rule_idx):
    """
    Map 2-bit values (0..3) to bases (0..3) using per-byte rule_idx in 0..7.
    quartets_u2: (N,4) uint8 (0..3)
    rule_idx:    (N,)  uint8 (0..7)
    returns bases: (N,4) uint8 (0..3) A/C/G/T
    """
    N = quartets_u2.shape[0]
    bases = np.empty_like(quartets_u2)
    for r in range(8):
        mask = (rule_idx == r)
        if not np.any(mask):
            continue
        vals = quartets_u2[mask]  # (k,4)
        bases[mask] = RULES[r][vals]
    return bases

def dna_decode_quartets(bases, rule_idx):
    """
    Inverse mapping bases->2-bit values using rule_idx.
    """
    N = bases.shape[0]
    quartets = np.empty_like(bases)
    for r in range(8):
        mask = (rule_idx == r)
        if not np.any(mask):
            continue
        vals = bases[mask]
        quartets[mask] = INV_RULES[r][vals]
    return quartets

def dna_op(basesA, basesB, op='xor'):
    """
    DNA operation on bases (A/C/G/T encoded as 0..3), but we define ops
    on their underlying 2-bit numbers for simplicity:
      xor:   v = a ^ b
      add:   v = (a + b) % 4
    Return result as bases (0..3) assuming they still represent 2-bit values.
    """
    if op == 'xor':
        v = np.bitwise_xor(basesA, basesB)  # still in 0..3
    elif op == 'add':
        v = (basesA + basesB) % 4
    else:
        raise ValueError("op must be 'xor' or 'add'")
    return v.astype(np.uint8)

# ---------------------------
# Full pipeline
# ---------------------------
def encrypt_image(img_u8, seed=0.723, mu=3.99, op='xor'):
    """
    img_u8: uint8 array HxW or HxWx3
    Steps:
      1) Chaos → row/col permutation
      2) Diffusion with chaos keystream
      3) DNA encode + op with DNA key (from chaos) + decode
    Returns cipher, aux info for decryption.
    """
    if img_u8.ndim == 3:
        # Work channel-wise for simplicity
        chans = cv2.split(img_u8)
        enc_chans = []
        keys = []
        for ch in chans:
            ciph, key = encrypt_image(ch, seed=seed, mu=mu, op=op)
            enc_chans.append(ciph)
            keys.append(key)
        return cv2.merge(enc_chans), keys

    H, W = img_u8.shape
    N = H*W

    # 1) Permutation keys (rows/cols)
    row_seq_f = logistic_map(mu, seed, max(H, W))    # float
    col_seq_f = logistic_map(mu, seed/2.0, max(H,W)) # another seed
    row_seq_b = (row_seq_f * 1e6).astype(np.int64)
    col_seq_b = (col_seq_f * 1e6).astype(np.int64)

    img_perm, row_idx, col_idx = permute_image(img_u8, row_seq_b, col_seq_b)

    # 2) Diffusion key stream
    ks_bytes = chaos_bytes(mu, seed*1.3 % 0.99, N).reshape(H, W)
    diffed = diffuse(img_perm, ks_bytes)

    # 3) DNA stage
    # Per-pixel rule 0..7 from chaos
    rule_idx = (chaos_bytes(mu, seed*1.7 % 0.99, N) % 8).reshape(-1)
    # DNA key (bases 0..3) for each quartet:
    dna_key_quartets = (chaos_bytes(mu, seed*2.1 % 0.99, N*4) % 4).reshape(-1,4)

    # Flatten for DNA ops on bytes
    flat = diffed.reshape(-1)

    # Encode -> operate -> decode (vectorized per rule)
    q = bytes_to_quartets_u2(flat)                 # (N,4) values 0..3
    bases = dna_encode_quartets(q, rule_idx)       # (N,4) A/C/G/T→0..3
    # DNA operation with key
    bases2 = dna_op(bases, dna_key_quartets, op=op)  # still 0..3
    # Decode back to 2-bit values using SAME rule
    q_back = dna_decode_quartets(bases2, rule_idx)   # (N,4) 0..3
    out_bytes = quartets_u2_to_bytes(q_back).reshape(H, W)

    cipher = out_bytes

    aux = {
        "H": H, "W": W,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "ks_bytes": ks_bytes,
        "rule_idx": rule_idx,
        "dna_key_quartets": dna_key_quartets,
        "seed": seed,
        "mu": mu,
        "op": op,
    }
    return cipher, aux

def decrypt_image(cipher_u8, aux):
    if cipher_u8.ndim == 3:
        chans = cv2.split(cipher_u8)
        dec_chans = []
        for ch, key in zip(chans, aux):
            dec_chans.append(decrypt_image(ch, key))
        return cv2.merge(dec_chans)

    H, W = aux["H"], aux["W"]
    row_idx = aux["row_idx"]
    col_idx = aux["col_idx"]
    ks_bytes = aux["ks_bytes"]
    rule_idx = aux["rule_idx"]
    dna_key_quartets = aux["dna_key_quartets"]
    op = aux["op"]

    # Inverse DNA stage (reverse operation)
    # Forward did: encode -> OP with key -> decode
    # Reverse:     encode -> INV_OP with same key -> decode
    flat = cipher_u8.reshape(-1)
    q = bytes_to_quartets_u2(flat)
    bases = dna_encode_quartets(q, rule_idx)
    if op == 'xor':
        bases2 = dna_op(bases, dna_key_quartets, op='xor')  # XOR is its own inverse
    elif op == 'add':
        # inverse of addition mod 4 is subtraction mod 4 -> (a - b) % 4 == (a + (4-b)) % 4
        bases2 = (bases - dna_key_quartets) % 4
    else:
        raise ValueError("op must be 'xor' or 'add'")
    q_back = dna_decode_quartets(bases2, rule_idx)
    after_dna = quartets_u2_to_bytes(q_back).reshape(H, W)

    # Inverse diffusion
    undiff = inv_diffuse(after_dna, ks_bytes)

    # Inverse permutation
    plain = inv_permute_image(undiff, row_idx, col_idx)
    return plain

# ---------------------------
# Demo / CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None, help="Path to image (png/jpg). If omitted, a synthetic image is used.")
    ap.add_argument("--op", type=str, default="xor", choices=["xor","add"], help="DNA op to use")
    ap.add_argument("--seed", type=float, default=0.723, help="Initial x0 for logistic map (0<x0<1)")
    ap.add_argument("--mu", type=float, default=3.99, help="Chaos parameter (3.57..4 gives chaos)")
    args = ap.parse_args()

    # Load or generate image
    if args.input and os.path.exists(args.input):
        img = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to read image.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # show in RGB for plotting
    else:
    # synthetic RGB test pattern (safe uint8 handling)
        H, W = 256, 256
        x = np.arange(W, dtype=np.uint8)
        y = np.arange(H, dtype=np.uint8)
        X, Y = np.meshgrid(x, y)
        Z = ((X.astype(np.uint16)//2 + Y.astype(np.uint16)//2) % 256).astype(np.uint8)
        img = np.dstack([X, Y, Z])

    # Encrypt & decrypt
    cipher, aux = encrypt_image(img, seed=args.seed, mu=args.mu, op=args.op)
    plain = decrypt_image(cipher, aux)

    # Verify
    ok = np.array_equal(img, plain)
    print("Decryption exact match:", ok)

    # Show
    fig = plt.figure(figsize=(12,4))
    ax1 = plt.subplot(1,3,1); ax1.imshow(img); ax1.set_title("Original"); ax1.axis("off")
    ax2 = plt.subplot(1,3,2); ax2.imshow(cipher); ax2.set_title("Encrypted"); ax2.axis("off")
    ax3 = plt.subplot(1,3,3); ax3.imshow(plain); ax3.set_title(f"Decrypted (ok={ok})"); ax3.axis("off")
    # Save results to files
    cv2.imwrite("original.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite("encrypted.png", cv2.cvtColor(cipher, cv2.COLOR_RGB2BGR))
    cv2.imwrite("decrypted.png", cv2.cvtColor(plain, cv2.COLOR_RGB2BGR))

    print("Images saved as: original.png, encrypted.png, decrypted.png")

    plt.tight_layout()
    plt.show()   # still show the popup if possible


if __name__ == "__main__":
    main()
